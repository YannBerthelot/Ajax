"""PPO (Schulman et al., 2017), with brax's minibatch geometry as an option.

Each iteration collects an ``(n_steps, n_envs)`` rollout, then runs
``n_epochs`` passes of a critic and an actor step on every minibatch of it
(the clipped surrogate: :mod:`ajax.agents.PPO.core`). The rollout's
geometry (:class:`ajax.agents.minibatch.Geometry`) decides how it is
minibatched and where GAE is computed.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from ajax.agents.diagnostics import approximate_kl, explained_variance, mean_policy_std
from ajax.agents.loop import TrainLoop, extension_key
from ajax.agents.minibatch import (
    get_minibatches_from_batch,
    get_minibatches_preserving_time,
    minibatch_geometry,
    rollout_actions,
    run_epochs,
    sequence_minibatches,
)
from ajax.agents.PPO.core import (
    AuxiliaryLogs,
    PolicyAuxiliaries,
    ValueAuxiliaries,
    clipped_surrogate,
    policy_entropy,
    recompute_log_prob,
    resolve_clip_coef,
)
from ajax.agents.PPO.state import PPOConfig, PPOState
from ajax.agents.PPO.utils import _compute_gae
from ajax.agents.recurrent import rollout_sequence_values
from ajax.agents.steps import joint_step
from ajax.deprecation import moved_names
from ajax.environments.interaction import (
    get_pi,
    get_pi_sequence,
    init_collector_state,
)
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
)
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import (
    get_initialized_actor_critic,
    predict_value,
    predict_value_sequence,
    state_value,
)
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)


def init_PPO(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    expert_policy: Optional[Callable] = None,
) -> PPOState:
    """The initial actor, critic and collector (holding ``expert_policy``'s
    initial state when it is stateful)."""
    rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)

    continuous = check_if_environment_has_continuous_actions(
        env_args.env, env_params=env_args.env_params
    )
    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=continuous,
        action_value=False,
        squash=network_args.squash,
        num_critics=1,
    )
    mode = env_args.mode
    collector_state = init_collector_state(
        collector_key, env_args=env_args, mode=mode, expert_policy=expert_policy
    )

    return PPOState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
        n_updates=0,
    )


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    recurrent: bool,
    vf_coef: float = 1.0,
    extra_loss_fn: Optional[Callable] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[ValueAuxiliaries, dict]]:
    """``vf_coef * 0.5 * MSE(V(obs), targets)`` plus ``extra_loss_fn``'s
    term, with the metrics and ``extra_loss_fn``'s (``{}`` without one). A
    recurrent critic replays the sequence from ``initial_hidden``,
    ``resets`` flagging the observations that start an episode."""
    if recurrent:
        assert resets is not None and initial_hidden is not None
        v_preds, _ = predict_value_sequence(
            critic_state=critic_states,
            critic_params=critic_params,
            x=observations,
            resets=resets,
            initial_hidden=initial_hidden,
        )
    else:
        v_preds = predict_value(
            critic_state=critic_states,
            critic_params=critic_params,
            x=observations,
        )
    v_preds = v_preds.squeeze(0)  # the single critic keeps the ensemble axis

    loss = vf_coef * 0.5 * jnp.mean((v_preds - value_targets) ** 2)
    metrics: dict = {}
    if extra_loss_fn is not None:
        extra, metrics = extra_loss_fn(
            critic_params, critic_states, observations, value_targets
        )
        loss = loss + extra

    aux = ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
        explained_variance=explained_variance(v_preds, value_targets),
    )
    return loss, (aux, metrics)


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    recurrent: bool,
    clip_coef: float,
    ent_coef: float,
    advantage_normalization: bool,
    extra_loss_fn: Optional[Callable] = None,
    raw_actions: Optional[jax.Array] = None,
    entropy_rng: Optional[jax.Array] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[PolicyAuxiliaries, dict]]:
    """The clipped surrogate minus ``ent_coef`` times the entropy bonus,
    plus ``extra_loss_fn``'s term; with ``advantage_normalization`` the
    advantages are normalised per minibatch. A recurrent actor replays the
    sequence from ``initial_hidden`` (the live carry on ``actor_state`` has
    advanced past this rollout), ``resets`` flagging episode starts."""
    if recurrent:
        assert resets is not None
        pi, _ = get_pi_sequence(
            actor_state=actor_state,
            actor_params=actor_params,
            obs=observations,
            resets=resets,
            initial_hidden=initial_hidden,
        )
    else:
        pi, _ = get_pi(actor_state, actor_params, observations)

    new_log_probs = recompute_log_prob(pi, actions, raw_actions)
    ratio = jnp.exp(new_log_probs - log_probs)

    if advantage_normalization:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    surrogate, clip_fraction = clipped_surrogate(ratio, gae, clip_coef)
    loss_actor = surrogate.mean()
    entropy = policy_entropy(pi, entropy_rng).mean()

    total_loss = loss_actor - ent_coef * entropy
    metrics: dict = {}
    if extra_loss_fn is not None:
        # A discrete policy has no mean action.
        pi_mean = None if isinstance(pi, distrax.Categorical) else pi.mean()
        extra, metrics = extra_loss_fn(actor_params, actor_state, observations, pi_mean)
        total_loss = total_loss + extra

    aux = PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy,
        approx_kl=approximate_kl(new_log_probs - log_probs),
        policy_std=mean_policy_std(pi),
    )
    return total_loss, (aux, metrics)


def _stack_critic_loss(
    extension_stack: ExtensionStack,
    agent_state: PPOState,
    key: jax.Array,
    total_timesteps: int,
) -> Optional[Callable]:
    """The stack's additive critic-loss term on ``key`` as a
    ``value_loss_function`` ``extra_loss_fn``: ``(critic_params,
    critic_states, observations, value_targets) -> (term, metrics)``;
    ``None`` for an empty stack. The extensions read their state from
    ``agent_state``.
    """
    if not extension_stack:
        return None

    def critic_loss(critic_params, critic_states, observations, value_targets):
        _batch = {
            "observations": observations,
            "targets": value_targets,
            "critic_params": critic_params,
            "critic_state": critic_states,
        }
        return extension_stack.fold_critic_loss(
            agent_state,
            _batch,
            agent_state.collector_state.timestep,
            key,
            total_timesteps,
        )

    return critic_loss


def _stack_actor_loss(
    extension_stack: ExtensionStack,
    agent_state: PPOState,
    key: jax.Array,
    total_timesteps: int,
    raw_observations: Optional[jax.Array] = None,
) -> Optional[Callable]:
    """The stack's additive actor-loss term on ``key`` as a
    ``policy_loss_function`` ``extra_loss_fn``: ``(actor_params,
    actor_state, observations, pi_mean) -> (term, metrics)``, the minibatch's
    ``raw_observations`` beside them; ``None`` for an empty stack."""
    if not extension_stack:
        return None

    def actor_loss(actor_params, actor_state, observations, pi_mean):
        _batch = {
            "observations": observations,
            "raw_observations": raw_observations,
            "pi_mean": pi_mean,
            "actor_params": actor_params,
            "actor_state": actor_state,
        }
        return extension_stack.fold_actor_loss(
            agent_state,
            _batch,
            agent_state.collector_state.timestep,
            key,
            total_timesteps,
        )

    return actor_loss


def update_agent(
    agent_state: PPOState,
    rollout: Transition,
    start: PPOState,
    agent_config: PPOConfig,
    recurrent: bool,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> tuple[PPOState, AuxiliaryLogs]:
    """One PPO update on the ``(n_steps, n_envs)`` ``rollout`` collected
    from ``start``: ``n_epochs`` passes over its minibatches."""
    n_steps, n_envs = rollout.obs.shape[:2]
    geometry = minibatch_geometry(agent_config, n_steps, n_envs, recurrent)
    actions, log_probs = rollout_actions(rollout)
    # The update's key: it shuffles the rollout, and the rollout's folds draw
    # on keys derived from it.
    agent_state, shuffle_key = agent_state.next_keys()
    reward_key, target_key = jax.random.split(extension_key(shuffle_key))
    rewards, reward_metrics = rollout.reward, dict[str, Any]()
    if "reward" in extension_stack.implemented_phases():
        # The extensions' rewards replace the env's (GAIL's discriminator).
        agent_state, rewards, reward_metrics = extension_stack.fold_reward(
            agent_state,
            {
                "observations": rollout.obs,
                "raw_observations": rollout.raw_obs,
                "actions": actions,
            },
            rewards,
            agent_state.collector_state.timestep,
            reward_key,
            total_timesteps,
        )
    batch = {
        "obs": rollout.obs,
        "actions": actions,
        "log_probs": log_probs,
        "raw_actions": rollout.raw_action,
    }
    if extension_stack:
        # The environment's own observations, which an expert reads.
        batch["raw_obs"] = rollout.raw_obs
    rollout_metrics: dict = {}  # the extensions' on the rollout's targets

    if geometry.kind == "time":
        # GAE in each minibatch, with the critic of that step.
        batch |= {
            "next_obs": rollout.next_obs,
            "terminated": rollout.terminated,
            "truncated": rollout.truncated,
            "rewards": rewards,
        }
    else:
        if recurrent:
            agent_state, values, next_values, resets, carries = rollout_sequence_values(
                agent_state, rollout, start, geometry.length
            )
            batch["resets"] = resets
        else:
            values = predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.critic_state.params,
                x=rollout.obs,
            ).squeeze(0)
            next_values = predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.critic_state.params,
                x=rollout.next_obs,
            ).squeeze(0)
        gae, value_targets = _compute_gae(
            values=values,
            next_values=next_values,
            rewards=rewards,
            terminateds=rollout.terminated,
            truncateds=rollout.truncated,
            gamma=agent_config.gamma,
            gae_lambda=agent_config.gae_lambda,
        )
        if extension_stack:
            target_batch = {
                "observations": rollout.obs,
                "next_observations": rollout.next_obs,
                "rewards": rewards,
                "terminated": rollout.terminated,
                "truncated": rollout.truncated,
                "gae": gae,
                "values": values,
                "next_values": next_values,
                "gamma": agent_config.gamma,
            }
            value_targets, rollout_metrics = extension_stack.fold_on_target(
                agent_state,
                target_batch,
                value_targets,
                agent_state.collector_state.timestep,
                target_key,
                total_timesteps,
            )
        batch |= {"targets": value_targets, "gae": gae}

    k, length = geometry.num_minibatches, geometry.length

    def minibatches(key: jax.Array) -> dict:
        """One epoch's partition of the rollout."""
        if geometry.kind == "time":
            return get_minibatches_preserving_time(batch, key, k, length)
        if geometry.kind == "recurrent":
            return sequence_minibatches(batch, carries, key, k, length)
        return get_minibatches_from_batch(batch, key, k)

    def minibatch_step(agent_state: PPOState, mb: dict) -> tuple[PPOState, Any]:
        """A critic and an actor step, both from the state before either."""
        agent_state, ent_rng = agent_state.next_keys()
        target_metrics: dict = {}
        critic, actor = agent_state.critic_state, agent_state.actor_state
        # The extensions' target and loss terms, each on a key of its own.
        target_key, critic_key, actor_key = jax.random.split(extension_key(ent_rng), 3)
        critic_extra = _stack_critic_loss(
            extension_stack, agent_state, critic_key, total_timesteps
        )
        actor_extra = _stack_actor_loss(
            extension_stack, agent_state, actor_key, total_timesteps, mb.get("raw_obs")
        )

        if geometry.kind == "time":
            # brax's compute_ppo_loss: GAE from the current critic, held
            # constant (no gradient reaches the critic through it).
            values = jax.lax.stop_gradient(
                predict_value(
                    critic_state=critic, critic_params=critic.params, x=mb["obs"]
                ).squeeze(0)
            )
            next_values = jax.lax.stop_gradient(
                predict_value(
                    critic_state=critic,
                    critic_params=critic.params,
                    x=mb["next_obs"],
                ).squeeze(0)
            )
            gae, value_targets = _compute_gae(
                values=values,
                next_values=next_values,
                rewards=mb["rewards"],
                terminateds=mb["terminated"],
                truncateds=mb["truncated"],
                gamma=agent_config.gamma,
                gae_lambda=agent_config.gae_lambda,
            )
            gae = jax.lax.stop_gradient(gae)
            value_targets = jax.lax.stop_gradient(value_targets)
            if extension_stack:
                target_batch = {
                    "observations": mb["obs"],
                    "next_observations": mb["next_obs"],
                    "rewards": mb["rewards"],
                    "terminated": mb["terminated"],
                    "truncated": mb["truncated"],
                    "gae": gae,
                    "values": values,
                    "next_values": next_values,
                    "gamma": agent_config.gamma,
                }
                value_targets, target_metrics = extension_stack.fold_on_target(
                    agent_state,
                    target_batch,
                    value_targets,
                    agent_state.collector_state.timestep,
                    target_key,
                    total_timesteps,
                )
        else:
            gae, value_targets = mb["gae"], mb["targets"]

        resets = mb.get("resets")
        (_, (v_aux, v_metrics)), v_grads = jax.value_and_grad(
            value_loss_function, has_aux=True
        )(
            critic.params,
            critic,
            mb["obs"],
            value_targets,
            recurrent,
            vf_coef=agent_config.vf_coef,
            extra_loss_fn=critic_extra,
            resets=resets,
            initial_hidden=mb.get("critic_hidden"),
        )
        (_, (p_aux, p_metrics)), p_grads = jax.value_and_grad(
            policy_loss_function, has_aux=True
        )(
            actor.params,
            actor,
            mb["obs"],
            mb["actions"],
            mb["log_probs"],
            gae,
            recurrent,
            resolve_clip_coef(agent_config, agent_state.collector_state.timestep),
            agent_config.ent_coef,
            agent_config.normalize_advantage,
            extra_loss_fn=actor_extra,
            raw_actions=mb["raw_actions"],
            entropy_rng=ent_rng,
            resets=resets,
            initial_hidden=mb.get("actor_hidden"),
        )
        actor, critic = joint_step(
            actor, critic, p_grads, v_grads, agent_config.max_grad_norm
        )
        agent_state = agent_state.replace(critic_state=critic, actor_state=actor)
        # Each network's norm before the joint clip.
        p_aux = p_aux.replace(grad_norm=optax.global_norm(p_grads))
        v_aux = v_aux.replace(grad_norm=optax.global_norm(v_grads))
        metrics = merge_metrics(target_metrics, v_metrics, p_metrics)
        return agent_state, AuxiliaryLogs(policy=p_aux, value=v_aux, extensions=metrics)

    agent_state, aux = run_epochs(
        agent_state, shuffle_key, minibatches, minibatch_step, agent_config.n_epochs
    )
    # The flat and recurrent updates fold on_target once, over the rollout.
    aux = aux.replace(
        extensions=merge_metrics(reward_metrics, rollout_metrics, aux.extensions)
    )
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)

    return agent_state, aux


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: PPOConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    expert_policy: Optional[Callable] = None,
    extensions: Sequence = (),
):
    """PPO's train function: an ``n_steps`` rollout per env, then one
    update, per iteration."""
    recurrent = network_args.memory is not None
    if recurrent and (extensions or expert_policy is not None):
        raise NotImplementedError(
            "Recurrent PPO does not support extensions or an expert_policy yet."
        )
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        expert_policy=expert_policy,
    )

    def init(key: jax.Array) -> PPOState:
        return init_PPO(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            expert_policy,
        )

    def update(
        agent_state: PPOState, rollout: Transition, start: PPOState
    ) -> tuple[PPOState, AuxiliaryLogs]:
        return update_agent(
            agent_state,
            rollout,
            start,
            agent_config,
            recurrent,
            loop.stack,
            total_timesteps,
        )

    return loop.on_policy(
        init,
        update,
        agent_config.n_steps,
        recurrent=recurrent,
        expose_rollout=agent_config.expose_recent_rollout,
        forced_resets=(agent_config.num_resets_per_eval, agent_config.num_evals),
        eval_kwargs={
            "value_estimate": ValueEstimate(state_value, agent_config.gamma),
            "expert_policy": expert_policy,
        },
    )


__getattr__ = moved_names(
    __name__,
    dict.fromkeys(("Geometry", "default_bptt_length"), "ajax.agents.minibatch")
    | {"clip_global_norm": "ajax.agents.steps"},
)
