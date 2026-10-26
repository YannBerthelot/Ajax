"""Upside-Down RL (Schmidhuber, 2019; Srivastava et al., 2019): RL as
supervised learning of a command-conditioned policy pi(a | s, c), the
command c = (d_r, d_h) a desired return and horizon.

One iteration on the own-iteration family's loop (OwnIterationAgent), each
line a method or function below::

    roll out n_steps per env, the actor reading c scaled           UDRL.collect_step
        c <- (d_r - r, max(d_h - 1, 1)) within an episode,             update_command
        a fresh command at an episode's end:
        d_r ~ U(R_target, R_target + R_std), d_h = H_target
    append the segment to the replay of buffer_capacity segments    (buffer.add_segment)
    R_target, R_std, H_target <- the top-K episodes' mean return, its spread and
        their mean horizon, smoothed by command_target_tau     (buffer.topk_command_stats)
    for each of n_updates_per_iter batches of (s, a, d_r, d_h), d_r and d_h
        realised over a sampled interval of an episode     (buffer.sample_training_batch):
        one step of the actor down -log pi(a | s, c)                  compute_actor_loss

The replay is what makes UDRL bootstrap: its segments span many policies
and sub-trajectory commands, so the actor learns p(a | s, c) over a wide
range of c rather than the current rollout's. Commands, top-K statistics
and training targets are undiscounted, as in the paper; gamma discounts
only the logged Train/mean_rtg. The actor reads the command scaled
(scale_command), the replay keeps it raw. No critic is trained.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Union

import distrax
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.agents.families import (
    ExpertGuided,
    GradientClipping,
    ImageObservations,
    ObservationNormalization,
    OwnIterationAgent,
    RewardNormalization,
    TrainContext,
)
from ajax.agents.loop import with_obs_norm
from ajax.agents.UDRL.buffer import (
    SegmentBuffer,
    add_segment,
    episode_stream,
    init_buffer,
    sample_training_batch,
    topk_command_stats,
)
from ajax.agents.update_steps import apply_post_update
from ajax.controller import (
    ActingState,
    feature_width,
    find_expert_action_in_choice_space,
    make_run_controller,
    policy_action_dim,
)
from ajax.environments.interaction import (
    ActionPipelineResult,
    executed_action,
    expert_collect_kwargs,
    get_final_obs,
    get_pi,
    get_raw_final_obs,
    get_raw_obs,
    step,
    step_features,
    update_episodic_return,
)
from ajax.environments.obs_norm import update_obs_norm
from ajax.environments.utils import agent_action_to_env, env_keys, raw_obs_columns
from ajax.evaluate import evaluate
from ajax.log import eval_norm_info
from ajax.networks.distributions import Squash
from ajax.state import BaseAgentState, LoadedTrainState, Transition
from ajax.types import Architecture
from ajax.wrappers import RAW_REWARD_KEY


@dataclass(kw_only=True, eq=False, repr=False)
class UDRL(
    GradientClipping,
    ObservationNormalization,
    RewardNormalization,
    ImageObservations,
    ExpertGuided,
    OwnIterationAgent,
):
    """Upside-Down RL (Schmidhuber, 2019): command-conditioned supervised
    policy learning on a replay of rollout segments (the module's program).

    With an image (cnn_image_shape), the (d_r, d_h) command joins the CNN's
    embedding after the convolutions, beside cnn_extra_obs_dim features. With
    an expert_policy, a residual policy or gain mode acts through it, and
    each evaluation logs its return.
    """

    actor_learning_rate: float = 3e-4
    actor_architecture: Architecture = ("128", "relu", "128", "relu")
    #: Discounts only the Train/mean_rtg diagnostic: commands, top-K
    #: statistics and training targets are undiscounted, as in the paper.
    gamma: float = 1.0
    #: The steps each env collects per iteration (a segment's length).
    n_steps: int = 64
    batch_size: int = 64
    command_return_init: float = 1.0
    command_horizon_init: float = 100.0
    #: Every iteration the return target is the top-K episode returns' mean
    #: times command_return_boost and its spread their std: exploration draws
    #: the return command uniformly in [target, target + std] (UDRL paper
    #: Sec. 4.2); the horizon target is the same top-K's mean horizon.
    command_topk: int = 32
    command_return_boost: float = 1.0
    #: Smoothing of the targets' update: (1 - tau) old + tau proposal; 1 replaces.
    command_target_tau: float = 1.0
    #: The scales of (d_r, d_h) the actor reads (the paper's LunarLander
    #: 0.02 and 0.01): the network learns more cleanly on O(1) commands than
    #: O(100); the replay and the command's decay stay in raw units.
    command_scale_r: float = 0.02
    command_scale_h: float = 0.01
    #: The replay's capacity, in segments: with n_envs envs it holds about
    #: buffer_capacity * n_steps * n_envs / mean episode length episodes.
    buffer_capacity: int = 64
    #: The gradient steps per iteration, each on a batch from the replay
    #: (Algorithm 3).
    n_updates_per_iter: int = 64

    def actor_critic_args(self) -> dict[str, Any]:
        # The critic built beside the actor is never trained. The LayerNorm
        # on the policy encoder's output is kept from Ajax's earlier
        # networks (the shared default is now a plain MLP).
        return {
            "critic_learning_rate": self.actor_learning_rate,
            "critic_architecture": self.actor_architecture,
            "encoder_layer_norm": True,
        }

    def count_iteration_steps(self, context: TrainContext) -> tuple[int, str]:
        return context.env_args.n_envs * self.n_steps, "n_envs x n_steps"

    def init(self, key: jax.Array, context: TrainContext) -> UDRLState:
        """The actor (its input the observation and the command), the
        collector, whose observations carry the initial command, and the
        empty replay of segments."""
        training_key, init_key, collector_key, evaluation_key = jax.random.split(key, 4)
        env_args = context.env_args
        continuous = env_args.continuous
        actor_state, critic_state = context.make_actor_critic(
            init_key,
            continuous=continuous,
            action_value=False,
            squash=continuous,
            num_critics=1,
            extra_obs_dim=2,  # the (d_r, d_h) command is appended to obs
        )
        # Under normalize_observations the collector's statistics cover the
        # env's columns: the command appended to them stays raw, then scaled.
        collector_state = context.make_collector(collector_key)
        initial_command = jnp.tile(
            jnp.array(
                [self.command_return_init, self.command_horizon_init], jnp.float32
            ),
            (env_args.n_envs, 1),
        )
        last_obs = jnp.concatenate([collector_state.last_obs, initial_command], -1)
        collector_state = collector_state.replace(last_obs=last_obs)
        # The replay stores actions with a trailing axis: action_dim for
        # continuous actions (gain mode's gains), 1 for a discrete index.
        if continuous:
            action_dim, action_dtype = policy_action_dim(env_args), jnp.float32
        else:
            action_dim, action_dtype = 1, jnp.int32
        buffer = init_buffer(
            capacity=self.buffer_capacity,
            segment_length=self.n_steps,
            n_envs=env_args.n_envs,
            obs_dim=last_obs.shape[-1],
            action_dim=action_dim,
            action_dtype=action_dtype,
        )
        return UDRLState(
            rng=training_key,
            eval_rng=evaluation_key,
            actor_state=actor_state,
            critic_state=critic_state,
            collector_state=collector_state,
            n_updates=0,
            command_target_return=jnp.asarray(self.command_return_init, jnp.float32),
            command_target_return_std=jnp.asarray(0.0, jnp.float32),
            command_target_horizon=jnp.asarray(self.command_horizon_init, jnp.float32),
            buffer=buffer,
        )

    def make_acting_pipeline(
        self, context: TrainContext, keeps_choice: bool = False
    ) -> Any:
        """UDRL's action pipeline (interaction.ActionPipelineResult's): a
        sample of the actor on the observation with its command scaled,
        executed through the run's residual transform where it has one. With
        keeps_choice (an action extension in the stack), it gives the action
        stage the choice that executes the expert's action, in the
        segment's layout: the expert's own, or a residual of 0."""
        transform = next(iter(context.env_args.transforms), None)
        scale_r, scale_h = self.command_scale_r, self.command_scale_h

        def pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key):
            del rng, uniform, mix_key
            # Scaled before the actor's forward pass (paper, appendix); the
            # replay keeps the raw command.
            obs_for_actor = scale_command(acting.obs, scale_r, scale_h)
            actor = agent_state.actor_state
            pi, _ = get_pi(
                actor_state=actor, actor_params=actor.params, obs=obs_for_actor
            )
            action, log_probs = pi.sample_and_log_prob(seed=action_key)
            for_expert = None
            if keeps_choice:
                for_expert = find_expert_action_in_choice_space(
                    transform, True, acting.expert_action, action
                )
                for_expert = for_expert.reshape(action.shape).astype(action.dtype)
            return ActionPipelineResult(
                env_action=executed_action(transform, action, acting, raw_obs),
                policy_action=action,
                log_probs=log_probs,
                policy_action_for_expert=for_expert,
            )

        return pipeline

    def collect_step(
        self, agent_state: UDRLState, context: TrainContext
    ) -> tuple[UDRLState, Transition]:
        """One env step: the actor reads the observation and its command
        (the last two columns of last_obs) scaled, the env steps, and the
        next observation carries the next command. With an expert, called
        first as every collector calls it (expert_collect_kwargs), the action
        is executed through the run's transform (residual RL, gain mode) and
        the tracked expert; the segment keeps the policy's own action, which
        the actor fits."""
        env_args, n_envs = context.env_args, context.env_args.n_envs
        mode = env_args.mode
        collector = agent_state.collector_state
        # An action extension (JSRL, EDGE) acts in the action stage, its gates
        # drawing on a fourth key: split(k, 4)[:3] is split(k, 3), so the
        # agent's own draws are unchanged.
        acts = any(
            "action" in extension.implemented_phases()
            for extension in context.extensions
        )
        gate_key = None
        if acts:
            keys = jax.random.split(agent_state.rng, 4)
            training_key, action_key, step_key, gate_key = keys
        else:
            training_key, action_key, step_key = jax.random.split(agent_state.rng, 3)
        step_keys = env_keys(step_key, mode, n_envs)
        last_obs = collector.last_obs  # (n_envs, env's columns + 2)
        # The observation statistics take each observation once, when it is
        # collected and before the actor reads it (as collect_experience).
        if collector.obs_norm_info is not None:
            _, stats = update_obs_norm(
                last_obs[:, :-2], collector.obs_norm_info, raw_obs_columns(env_args.env)
            )
            collector = collector.replace(obs_norm_info=stats)
            agent_state = with_obs_norm(agent_state.replace(collector_state=collector))
        transform = next(iter(env_args.transforms), None)
        pipeline = expert_collect_kwargs(
            self.expert,
            self.make_acting_pipeline(context, keeps_choice=acts),
            False,
            n_envs,
            transform,
            extensions=context.extensions if acts else (),
            total_timesteps=context.total_timesteps,
        )["action_pipeline"]
        acting = ActingState(
            obs=last_obs,
            done=jnp.logical_or(collector.last_terminated, collector.last_truncated),
            timestep=jnp.asarray(collector.timestep),
            expert_state=collector.expert_state,
        )
        # The env's own observation, which only the expert reads.
        raw = None
        if self.expert is not None:
            raw = get_raw_obs(
                env_state=collector.env_state, env=env_args.env, mode=mode
            )
        result = pipeline(agent_state, acting, raw, gate_key, False, None, action_key)
        log_probs, env_action = result.log_probs, result.env_action
        # The segment keeps the choice the actor fits: the one executing the
        # expert's action where the stage handed it the step, unless the
        # run's replay chooser (StorePolicyAction) picks the policy's own.
        action = result.policy_action
        if result.stored_policy_action is not None:
            action = result.stored_policy_action
        for chooser in env_args.replay_action_choosers:
            action = chooser.choose_replay_action(action, result.policy_action)
        expert_state = collector.expert_state
        if result.new_expert_state is not None:
            expert_state = result.new_expert_state
        new_env_obs, new_env_state, reward, terminated, truncated, info = (
            jax.lax.stop_gradient(
                step(
                    step_keys,
                    collector.env_state,
                    agent_action_to_env(env_action, env_args.env, env_args.env_params),
                    env_args.env,
                    mode,
                    env_args.env_params,
                )
            )
        )
        done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool))
        done = done.astype(jnp.float32)
        if env_args.observers:
            # The next observation's features (the reset one's where an
            # episode ended), from the expert's state after this step's call,
            # before the command joins them.
            final_obs = get_final_obs(info, new_env_obs).astype(new_env_obs.dtype)
            next_columns, _, observer_states = step_features(
                env_args,
                mode,
                self.expert,
                collector.observer_states,
                new_env_state,
                get_raw_final_obs(info, final_obs),
                expert_state,
                done > 0,
                timestep=collector.timestep,
                horizon=collector.max_timesteps,
            )
            new_env_obs = jnp.concatenate([new_env_obs, next_columns], axis=-1)
            collector = collector.replace(observer_states=observer_states)
        # A fresh exploratory command per env at an episode's end
        # (Algorithm 5): d_r ~ U(R_target, R_target + R_std), d_h = H_target.
        training_key, explore_key = jax.random.split(training_key)
        explore_noise = jax.random.uniform(
            explore_key, shape=(n_envs,), minval=0.0, maxval=1.0
        )
        initial_return = (
            agent_state.command_target_return
            + explore_noise * agent_state.command_target_return_std
        )
        initial_horizon = jnp.broadcast_to(
            agent_state.command_target_horizon, (n_envs,)
        )
        # Envs that just ended restart from the drawn command; the others
        # decay theirs.
        desired_return, desired_horizon = update_command(
            last_obs[:, -2],
            last_obs[:, -1],
            reward,
            done,
            initial_return,
            initial_horizon,
        )
        new_command = jnp.stack([desired_return, desired_horizon], axis=-1)
        new_last_obs = jnp.concatenate([new_env_obs, new_command], axis=-1)
        transition = Transition(
            obs=last_obs,
            action=action,
            reward=reward[:, None],
            terminated=terminated[:, None],
            truncated=truncated[:, None],
            next_obs=new_last_obs,
            log_prob=log_probs,
        )
        # Episodes span segments: their returns are summed across iterations,
        # in the env's own units (a reward normaliser keeps them in info), as
        # collect_experience does.
        paid = (
            info[RAW_REWARD_KEY].astype(jnp.float32)
            if RAW_REWARD_KEY in info
            else reward
        )
        episodic_return_state, episodic_mean_return = update_episodic_return(
            collector.episodic_return_state, paid, done > 0
        )
        collector = collector.replace(
            rng=training_key,
            _env_state=new_env_state,
            last_obs=new_last_obs,
            timestep=collector.timestep + n_envs,
            last_terminated=terminated,
            last_truncated=truncated,
            episodic_return_state=episodic_return_state,
            episodic_mean_return=episodic_mean_return,
        )
        if expert_state is not None:
            collector = collector.replace(expert_state=expert_state)
        # Each env's step in its episode (JSRL's horizon), and the share of
        # envs the stage handed the expert, as collect_experience keeps them.
        if collector.step_in_episode is not None:
            steps = jnp.where(done > 0, 0, collector.step_in_episode + 1)
            collector = collector.replace(step_in_episode=steps)
        if result.is_expert_flag is not None:
            collector = collector.replace(
                last_expert_frac=jnp.mean(result.is_expert_flag)
            )
        agent_state = agent_state.replace(collector_state=collector, rng=training_key)
        return agent_state, transition

    def iterate(
        self, agent_state: UDRLState, context: TrainContext
    ) -> tuple[UDRLState, UDRLAuxiliaries]:
        """A segment, appended to the replay; the command targets from the
        replay's top-K episodes; n_updates_per_iter fits of the actor."""
        extensions, total_timesteps = context.extensions, context.total_timesteps
        agent_state, rollout = jax.lax.scan(
            lambda agent_state, _: self.collect_step(agent_state, context),
            agent_state,
            xs=None,
            length=self.n_steps,
        )
        dones = jnp.logical_or(
            rollout.terminated.astype(bool), rollout.truncated.astype(bool)
        ).astype(jnp.float32)
        assert agent_state.buffer is not None
        buffer = add_segment(
            agent_state.buffer,
            obs=rollout.obs,
            actions=rollout.action,
            rewards=rollout.reward,
            dones=dones,
        )
        agent_state = agent_state.replace(buffer=buffer)
        # The targets move towards the top-K statistics (Algorithm 5), each
        # kept while the replay holds no whole episode (NaN statistics).
        stream = episode_stream(buffer)
        top_return, top_return_std, top_horizon = topk_command_stats(
            stream, k=self.command_topk
        )
        tau = self.command_target_tau

        def refresh(old: Union[jax.Array, float], proposal: jax.Array) -> jax.Array:
            return (1.0 - tau) * old + tau * jnp.where(
                jnp.isnan(proposal), old, proposal
            )

        new_target_return = refresh(
            agent_state.command_target_return, top_return * self.command_return_boost
        )
        new_target_return_std = refresh(
            agent_state.command_target_return_std, top_return_std
        )
        new_target_horizon = refresh(agent_state.command_target_horizon, top_horizon)
        # A third key for the extensions: split(k, 3)[:2] is split(k) under
        # JAX's default partitionable threefry, so the agent's own draws are
        # unchanged.
        next_key, fit_key, extensions_key = jax.random.split(agent_state.rng, 3)
        agent_state = agent_state.replace(rng=next_key)
        env_width = buffer.obs.shape[-1] - 2 - feature_width(context.env_args)

        def fit_batch(
            agent_state: UDRLState, keys: tuple[jax.Array, jax.Array]
        ) -> tuple[UDRLState, tuple[jax.Array, dict]]:
            sample_key, extension_key = keys
            observations, actions = sample_training_batch(
                sample_key, buffer, stream, self.batch_size
            )
            # The command scaled as the actor reads it when acting.
            observations = scale_command(
                observations, self.command_scale_r, self.command_scale_h
            )
            actor = agent_state.actor_state

            def compute_loss(params: FrozenDict) -> tuple[jax.Array, dict]:
                loss = compute_actor_loss(params, actor, observations, actions)
                batch = {
                    "observations": observations,
                    "actions": actions,
                    "actor_params": params,
                    "actor_state": actor,
                }
                if extensions:
                    # For the extensions: the env's own observations at the
                    # batch's rows (the replay holds them raw, then their
                    # features and the command), and the policy's mean action
                    # (None for a discrete policy).
                    pi, _ = get_pi(actor, params, observations)
                    discrete = isinstance(pi, distrax.Categorical)
                    batch["raw_observations"] = observations[..., :env_width]
                    batch["pi_mean"] = None if discrete else pi.mean()
                extra, metrics = extensions.compute_extensions_actor_loss(
                    agent_state,
                    batch,
                    agent_state.collector_state.timestep,
                    extension_key,
                    total_timesteps,
                )
                return loss + extra, metrics

            (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(
                actor.params
            )
            actor = actor.apply_gradients(grads=grads)
            return agent_state.replace(actor_state=actor), (loss, metrics)

        n_updates = self.n_updates_per_iter
        keys = (
            jax.random.split(fit_key, n_updates),
            jax.random.split(extensions_key, n_updates),
        )
        agent_state, (losses, update_metrics) = jax.lax.scan(
            fit_batch, agent_state, keys
        )
        agent_state = apply_post_update(
            extensions,
            agent_state,
            agent_state.collector_state.timestep,
            total_timesteps,
        )
        # The realised return-to-go and horizon over the segment, logged only.
        returns_to_go, horizons = compute_returns_to_go_horizons(
            rollout.reward, dones, gamma=self.gamma
        )
        aux = UDRLAuxiliaries(
            actor_loss=losses.mean(),
            mean_rtg=returns_to_go.mean(),
            mean_horizon=horizons.mean(),
            episodic_return=agent_state.collector_state.episodic_mean_return,
            n_completed_episodes=dones.sum(),
            timestep=agent_state.collector_state.timestep,
            extensions=jax.tree.map(jnp.mean, update_metrics),
        )
        agent_state = agent_state.replace(
            n_updates=agent_state.n_updates + 1,
            command_target_return=new_target_return,
            command_target_return_std=new_target_return_std,
            command_target_horizon=new_target_horizon,
        )
        return agent_state, aux

    def report(self, agent_state: UDRLState, aux: UDRLAuxiliaries) -> dict:
        """The iteration's segment and updates."""
        del agent_state
        metrics = {
            "timestep": aux.timestep,
            "Train/episodic mean reward": aux.episodic_return,
            "Train/actor_loss": aux.actor_loss,
            "Train/mean_rtg": aux.mean_rtg,
            "Train/mean_horizon": aux.mean_horizon,
            "Train/n_completed_episodes": aux.n_completed_episodes,
        }
        metrics.update(aux.extensions)
        return metrics

    def evaluate(
        self,
        agent_state: UDRLState,
        key: jax.Array,
        context: TrainContext,
        num_episode_test: int,
    ) -> dict:
        """The deterministic policy's return, entropy and length on
        num_episode_test episodes of the shared evaluation, commanded by
        EvalCommand (acting through the run's transform and the expert), and
        the return and horizon each episode is commanded with at its start;
        the expert's return on the same episodes, eval_expert_policy's when
        given (read in agent units, as the family evaluation reads it)."""
        env_args = context.env_args
        returns, entropy, _, _, length, expert_returns = evaluate(
            env_args.env,
            agent_state.actor_state,
            num_episode_test,
            key,
            env_args.env_params,
            norm_info=eval_norm_info(agent_state, env_args.mode),
            agent_state=agent_state,
            command=EvalCommand(self.command_scale_r, self.command_scale_h),
            expert_policy=self.expert,
            eval_expert_policy=self._in_agent_units(self.eval_expert_policy),
            controller=make_run_controller(env_args),
        )
        return {
            "Eval/episodic mean reward": returns,
            "Eval/episodic entropy": entropy,
            "Eval/mean episodic length": length,
            "Eval/episodic mean expert reward": expert_returns.mean(),
            "Eval/expert bias": jnp.mean(returns) - expert_returns.mean(),
            "Eval/commanded return": agent_state.command_target_return,
            "Eval/commanded horizon": agent_state.command_target_horizon,
        }

    # -- What extensions may rely on (ajax.extensions.capabilities) ----------
    name = "UDRL"
    action_space = "any"
    capabilities = frozenset(
        {
            "actor",
            "obs_networks",
            "off_policy",
            "policy_density",
            "replay",
            "reward_learning",
        }
    )
    # Its collector runs the action stage (action), acts through the run's
    # transform (transform_action, actuate), observes the features (observe)
    # and stores what the run's chooser picks (choose_replay_action) itself.
    supported_extension_phases = OwnIterationAgent.supported_extension_phases.union(
        {
            "actor_loss",
            "transform_action",
            "actuate",
            "observe",
            "action",
            "choose_replay_action",
        }
    )
    # Unannotated, as the whole contract, so the dataclass makes it no field.
    phase_batch_keys = {  # noqa: RUF012
        "actor_loss": frozenset(
            {"actor_state", "observations", "raw_observations", "pi_mean", "actions"}
        ),
    }
    # It learns from the segments of up to buffer_capacity past iterations.
    replays_transitions = True


@dataclasses.dataclass(frozen=True)
class EvalCommand:
    """The command UDRL is evaluated with (ajax.evaluate.evaluate's
    command): each episode starts from the current target return and
    horizon, without the exploration noise collection adds, then decays as
    in collection (update_command; past an episode's end the evaluation
    counts nothing), the actor reading it scaled. Under reward
    normalisation the targets are in normalised units, as the rewards
    collection decays the command by: evaluate() hands update the rewards
    normalised with the training env's frozen statistics."""

    scale_r: float
    scale_h: float

    def initial(self, agent_state: UDRLState, n_envs: int) -> jax.Array:
        target = (agent_state.command_target_return, agent_state.command_target_horizon)
        return jnp.tile(jnp.stack(target), (n_envs, 1))

    def update(
        self, command: jax.Array, reward: jax.Array, done: jax.Array
    ) -> jax.Array:
        desired_return, desired_horizon = command[:, 0], command[:, 1]
        desired_return, desired_horizon = update_command(
            desired_return,
            desired_horizon,
            reward,
            done.astype(jnp.float32),
            desired_return,
            desired_horizon,
        )
        return jnp.stack([desired_return, desired_horizon], axis=-1)

    def actor_input(self, obs: jax.Array, command: jax.Array) -> jax.Array:
        obs = jnp.concatenate([obs, command.astype(obs.dtype)], axis=-1)
        return scale_command(obs, self.scale_r, self.scale_h)


# -- UDRL's equations, in the order an iteration calls them -------------------
def scale_command(obs: jax.Array, scale_r: float, scale_h: float) -> jax.Array:
    """The observation with its last two columns, the (d_r, d_h) command,
    scaled as the actor reads them (paper, appendix)."""
    base = obs[..., :-2]
    command = obs[..., -2:] * jnp.asarray([scale_r, scale_h], dtype=obs.dtype)
    return jnp.concatenate([base, command], axis=-1)


def update_command(
    prev_d_r: jax.Array,
    prev_d_h: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    return_init: jax.Array,
    horizon_init: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """One step of the (desired_return, desired_horizon) command.

    Within an episode d_r decreases by the realised reward and d_h by 1,
    floored at 1 so an episode running past its desired horizon is not
    conditioned on 0 or less; at an episode's end (done 1) both restart from
    their initial values.
    """
    decayed_d_r = prev_d_r - reward
    decayed_d_h = jnp.maximum(prev_d_h - 1.0, 1.0)
    new_d_r = jnp.where(done > 0.0, return_init, decayed_d_r)
    new_d_h = jnp.where(done > 0.0, horizon_init, decayed_d_h)
    return new_d_r, new_d_h


def compute_actor_loss(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    obs_with_realised_command: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """-log pi(a | s, c), on the realised command: the negative
    log-likelihood of a categorical policy, or of a squashed Gaussian at the
    action clipped inside its open interval."""
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=obs_with_realised_command,
    )
    if isinstance(pi, distrax.Categorical):
        # gymnax discrete actions arrive as (B,); other paths may carry a
        # trailing length-1 dim. Strip only that case so log_prob always
        # sees integer indices of shape (B,).
        if actions.ndim > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        return -pi.log_prob(actions).mean()
    if isinstance(pi, Squash):
        # Inside the open interval, where the squashed density is finite.
        margin = 1e-3
        target = jnp.clip(actions, -1.0 + margin, 1.0 - margin)
        return -pi.log_prob(target).sum(-1).mean()
    return -pi.log_prob(actions).sum(-1).mean()


def compute_returns_to_go_horizons(
    rewards: jax.Array,
    dones: jax.Array,
    gamma: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Each step's return-to-go and steps until done over a rollout.

    rewards, dones: time along the first axis (T, ...); dones[t] is 1 where
    the episode ended at step t. The reverse scan restarts at each end, so
    within each piece of an episode the return-to-go covers the rewards
    from t to the ending step, included.
    """

    def step_back(
        following: tuple[jax.Array, jax.Array], step: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        next_return, next_horizon = following
        reward, done = step
        not_done = 1.0 - done
        return_to_go = reward + gamma * next_return * not_done
        horizon = 1.0 + next_horizon * not_done
        return (return_to_go, horizon), (return_to_go, horizon)

    last = (jnp.zeros_like(rewards[0]), jnp.zeros_like(rewards[0]))
    _, (returns_to_go, horizons) = jax.lax.scan(
        step_back, last, (rewards[::-1], dones[::-1])
    )
    return returns_to_go[::-1], horizons[::-1]


# -- State and logged metrics -------------------------------------------------
@struct.dataclass
class UDRLState(BaseAgentState):
    """UDRL's state adds the command targets tracked across iterations and
    the replay of episode segments (Algorithm 3).

    UDRL bootstraps only while the rollout command (d_r, d_h) stays near
    what the current policy achieves; an initial command too far from the
    policy's returns leaves the (s, c) -> a mapping untrained at c, so
    rollouts commanded by c act at random. Each iteration refreshes the
    targets from the replay's top-K episode returns (Algorithm 5).
    """

    command_target_return: Union[float, jax.Array] = 1.0
    command_target_return_std: Union[float, jax.Array] = 0.0
    command_target_horizon: Union[float, jax.Array] = 100.0
    buffer: Optional[SegmentBuffer] = None


@struct.dataclass
class UDRLAuxiliaries:
    actor_loss: jax.Array
    mean_rtg: jax.Array
    mean_horizon: jax.Array
    # The house training return (update_episodic_return): each env's
    # rolling mean of whole-episode returns, averaged over envs.
    episodic_return: Union[float, jax.Array]
    # Episodes that ended in the iteration's segment.
    n_completed_episodes: jax.Array
    # The env steps collected up to and including this iteration.
    timestep: Union[int, jax.Array]
    # The extensions' training metrics, averaged over the iteration's updates.
    extensions: dict
