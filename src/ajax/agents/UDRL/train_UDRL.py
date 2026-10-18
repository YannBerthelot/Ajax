"""Training loop for Upside-Down RL (UDRL), faithful to Schmidhuber 2019.

UDRL re-frames RL as supervised learning of a command-conditioned policy
pi(a | s, command), with command = (desired_return, desired_horizon).
At each iteration we:
  - collect one rollout segment with the current policy (Algorithm 4);
  - append the segment to a fixed-size replay buffer of segments;
  - refresh the dynamic rollout command from top-K observed episode returns
    in the buffer (Algorithm 5);
  - sample (state, action, dr, dh) tuples from the buffer with paper-style
    interval RTG sampling (Algorithm 3) and fit the actor by NLL/MSE.

The replay buffer is what makes UDRL bootstrap: the training data spans
many policies and many sub-trajectory commands, so the actor learns the
conditional p(a | s, c) over a wide range of c rather than collapsing to
the current rollout's narrow distribution.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any, Optional, Tuple, Union

import distrax
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from jax.tree_util import Partial as partial

from ajax.agents.loop import Evaluation, TrainLoop, fold_post_update, with_obs_norm
from ajax.agents.UDRL.buffer import (
    add_segment,
    episode_stream,
    init_buffer,
    sample_training_batch,
    topk_command_stats,
)
from ajax.agents.UDRL.state import UDRLConfig, UDRLState
from ajax.agents.UDRL.utils import compute_returns_to_go_horizons, update_command
from ajax.environments.interaction import (
    get_pi,
    init_collector_state,
    step,
    update_episodic_return,
)
from ajax.environments.obs_norm import update_obs_norm
from ajax.environments.utils import (
    agent_action_to_env,
    check_if_environment_has_continuous_actions,
    env_keys,
    raw_obs_columns,
)
from ajax.evaluate import evaluate
from ajax.extensions.base import ExtensionStack
from ajax.log import eval_norm_info
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.distributions import Squash
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.wrappers import RAW_REWARD_KEY


@struct.dataclass
class UDRLAuxiliaries:
    actor_loss: jnp.ndarray
    mean_rtg: jnp.ndarray
    mean_horizon: jnp.ndarray
    # The house training return (update_episodic_return): each env's
    # rolling mean of whole-episode returns, averaged over envs.
    episodic_return: Union[float, jax.Array]
    # Episodes that ended in the iteration's segment.
    n_completed_episodes: jnp.ndarray
    # Total env timesteps consumed up to and including this iter.
    timestep: Union[int, jax.Array]


def init_UDRL(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: UDRLConfig,
) -> UDRLState:
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
        squash=continuous,
        num_critics=1,
        extra_obs_dim=2,  # the (d_r, d_h) command is appended to obs
    )
    # Under normalize_observations the collector's statistics cover the
    # env's columns: the command appended to them stays raw, then scaled.
    collector_state = init_collector_state(
        collector_key, env_args=env_args, mode=env_args.mode
    )

    n_envs = env_args.n_envs
    init_cmd = jnp.tile(
        jnp.array(
            [agent_config.command_return_init, agent_config.command_horizon_init],
            dtype=jnp.float32,
        ),
        (n_envs, 1),
    )
    new_last_obs = jnp.concatenate([collector_state.last_obs, init_cmd], axis=-1)
    collector_state = collector_state.replace(last_obs=new_last_obs)

    obs_dim = collector_state.last_obs.shape[-1]
    # The buffer stores actions with a trailing dim (1 for discrete, action_dim
    # for continuous).
    from ajax.environments.utils import get_action_dim

    if continuous:
        action_dim = get_action_dim(env_args.env, env_args.env_params)
        action_dtype = jnp.float32
    else:
        action_dim = 1
        action_dtype = jnp.int32
    buffer = init_buffer(
        capacity=agent_config.buffer_capacity,
        segment_length=agent_config.n_steps,
        n_envs=env_args.n_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_dtype=action_dtype,
    )

    return UDRLState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
        n_updates=0,
        command_target_return=jnp.asarray(
            agent_config.command_return_init, dtype=jnp.float32
        ),
        command_target_return_std=jnp.asarray(0.0, dtype=jnp.float32),
        command_target_horizon=jnp.asarray(
            agent_config.command_horizon_init, dtype=jnp.float32
        ),
        buffer=buffer,
    )


def _udrl_collect_step(
    agent_state: UDRLState,
    _: Any,
    *,
    env_args: EnvironmentConfig,
    mode: str,
    agent_config: UDRLConfig,
):
    """One env step. Reads (d_r, d_h) from the trailing 2 dims of last_obs,
    forwards the actor on the augmented obs, steps the env, then writes the
    next augmented obs (raw_next_obs ++ updated_command) back into
    collector_state.last_obs.
    """
    cs = agent_state.collector_state
    rng, action_key, step_key = jax.random.split(agent_state.rng, 3)
    rng_step = env_keys(step_key, mode, env_args.n_envs)

    last_obs = cs.last_obs  # (n_envs, raw_obs_dim + 2)
    # The observation statistics take each observation once, when it is
    # collected and before the actor reads it (as collect_experience).
    if cs.obs_norm_info is not None:
        _, stats = update_obs_norm(
            last_obs[:, :-2], cs.obs_norm_info, raw_obs_columns(env_args.env)
        )
        cs = cs.replace(obs_norm_info=stats)
        agent_state = with_obs_norm(agent_state.replace(collector_state=cs))
    # Scale (dr, dh) before the actor's forward pass (paper §appendix). The
    # underlying buffer keeps raw values; the actor input is the scaled view.
    last_obs_for_actor = _scale_obs_command(
        last_obs, agent_config.command_scale_r, agent_config.command_scale_h
    )
    pi, new_actor_state = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=last_obs_for_actor,
    )
    action, log_probs = pi.sample_and_log_prob(seed=action_key)

    new_env_obs, new_env_state, reward, terminated, truncated, info = (
        jax.lax.stop_gradient(
            step(
                rng_step,
                cs.env_state,
                agent_action_to_env(action, env_args.env, env_args.env_params),
                env_args.env,
                mode,
                env_args.env_params,
            )
        )
    )
    done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool)).astype(
        jnp.float32
    )

    # Sample fresh exploratory commands per env at episode reset (Algorithm 5):
    # dr ~ Uniform(target_R, target_R + target_R_std), dh = target_H.
    rng, k_explore = jax.random.split(rng)
    explore_noise = jax.random.uniform(
        k_explore,
        shape=(env_args.n_envs,),
        minval=0.0,
        maxval=1.0,
    )
    sampled_init_R = (
        agent_state.command_target_return
        + explore_noise * agent_state.command_target_return_std
    )
    sampled_init_H = jnp.broadcast_to(
        agent_state.command_target_horizon, (env_args.n_envs,)
    )
    # Envs that just hit done restart from the sampled command; envs
    # mid-episode get the decayed (prev - reward, max(prev - 1, 1)).
    new_d_r, new_d_h = update_command(
        last_obs[:, -2], last_obs[:, -1], reward, done, sampled_init_R, sampled_init_H
    )
    new_command = jnp.stack([new_d_r, new_d_h], axis=-1)
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
        info[RAW_REWARD_KEY].astype(jnp.float32) if RAW_REWARD_KEY in info else reward
    )
    episodic_return_state, episodic_mean_return = update_episodic_return(
        cs.episodic_return_state, paid, done > 0
    )
    new_cs = cs.replace(
        rng=rng,
        _env_state=new_env_state,
        last_obs=new_last_obs,
        timestep=cs.timestep + env_args.n_envs,
        last_terminated=terminated,
        last_truncated=truncated,
        episodic_return_state=episodic_return_state,
        episodic_mean_return=episodic_mean_return,
    )
    return (
        agent_state.replace(
            collector_state=new_cs, actor_state=new_actor_state, rng=rng
        ),
        transition,
    )


def _scale_obs_command(obs: jax.Array, scale_r: float, scale_h: float) -> jax.Array:
    """Multiply the trailing 2 obs dims (the (dr, dh) command) by per-axis
    scaling factors. The buffer / decay logic uses raw values; the actor
    sees the scaled view at every forward pass (paper §appendix)."""
    base = obs[..., :-2]
    cmd = obs[..., -2:] * jnp.asarray([scale_r, scale_h], dtype=obs.dtype)
    return jnp.concatenate([base, cmd], axis=-1)


def actor_loss_fn(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    obs_with_realised_command: jax.Array,
    actions: jax.Array,
) -> jax.Array:
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
        log_p = pi.log_prob(actions)
        return -log_p.mean()

    if isinstance(pi, Squash):
        eps = 1e-3
        target = jnp.clip(actions, -1.0 + eps, 1.0 - eps)
        return -pi.log_prob(target).sum(-1).mean()
    return -pi.log_prob(actions).sum(-1).mean()


def training_iteration(
    agent_state: UDRLState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    agent_config: UDRLConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int = 1,
) -> Tuple[UDRLState, UDRLAuxiliaries]:
    # 1. Collect one rollout segment (Algorithm 4).
    collect_fn = partial(
        _udrl_collect_step, env_args=env_args, mode=mode, agent_config=agent_config
    )
    agent_state, rollout = jax.lax.scan(
        collect_fn, agent_state, xs=None, length=agent_config.n_steps
    )

    dones = jnp.logical_or(
        rollout.terminated.astype(bool), rollout.truncated.astype(bool)
    ).astype(jnp.float32)

    # 2. Append the segment to the replay buffer.
    assert agent_state.buffer is not None
    new_buffer = add_segment(
        agent_state.buffer,
        obs=rollout.obs,
        actions=rollout.action,
        rewards=rollout.reward,
        dones=dones,
    )
    agent_state = agent_state.replace(buffer=new_buffer)

    # 3. Refresh the rollout command target from buffer top-K (Algorithm 5):
    #    an EMA towards the top-K statistics, keeping the old target while
    #    the buffer holds no completed episode (NaN statistics).
    stream = episode_stream(new_buffer)
    topk_r, topk_std, topk_h = topk_command_stats(stream, k=agent_config.command_topk)
    tau = agent_config.command_target_tau

    def _refresh(old: jax.Array | float, proposal: jax.Array) -> jax.Array:
        return (1.0 - tau) * old + tau * jnp.where(jnp.isnan(proposal), old, proposal)

    new_target_r = _refresh(
        agent_state.command_target_return, topk_r * agent_config.command_return_boost
    )
    new_target_std = _refresh(agent_state.command_target_return_std, topk_std)
    new_target_h = _refresh(agent_state.command_target_horizon, topk_h)

    # 4. Train: sample (s, a, dr, dh) from the buffer and fit the actor with
    #    a fixed number of gradient updates per iteration (Algorithm 3).
    # A third key for the extensions: split(k, 3)[:2] is split(k) under JAX's
    # default partitionable threefry, so the agent's own draws are unchanged.
    rng, train_rng, ext_rng = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=rng)

    def update_step(carry, keys):
        a_state, (key, ext_key) = carry, keys
        obs_b, act_b = sample_training_batch(
            key, new_buffer, stream, agent_config.batch_size
        )
        # Scale the (dr, dh) portion of obs to match what the actor sees
        # at rollout time.
        obs_b = _scale_obs_command(
            obs_b, agent_config.command_scale_r, agent_config.command_scale_h
        )

        def _loss(params):
            loss = actor_loss_fn(params, a_state.actor_state, obs_b, act_b)
            _al_batch = {
                "observations": obs_b,
                "actions": act_b,
                "actor_params": params,
                "actor_state": a_state.actor_state,
            }
            if extension_stack:
                # For the extensions: the env's own observations at the
                # batch's rows (the buffer holds them raw), and the policy's
                # mean action (None for a discrete policy, which has none).
                pi, _ = get_pi(a_state.actor_state, params, obs_b)
                discrete = isinstance(pi, distrax.Categorical)
                _al_batch["raw_observations"] = obs_b[..., :-2]
                _al_batch["pi_mean"] = None if discrete else pi.mean()
            return loss + extension_stack.fold_actor_loss(
                a_state,
                _al_batch,
                a_state.collector_state.timestep,
                ext_key,
                total_timesteps,
            )

        loss, grads = jax.value_and_grad(_loss)(a_state.actor_state.params)
        new_actor_state = a_state.actor_state.apply_gradients(grads=grads)
        return a_state.replace(actor_state=new_actor_state), loss

    n_updates = agent_config.n_updates_per_iter
    update_keys = jax.random.split(train_rng, n_updates)
    ext_keys = jax.random.split(ext_rng, n_updates)
    agent_state, update_losses = jax.lax.scan(
        update_step, agent_state, (update_keys, ext_keys)
    )

    agent_state = fold_post_update(
        extension_stack,
        agent_state,
        agent_state.collector_state.timestep,
        total_timesteps,
    )

    # Realised RTG / horizon over the segment, only used for diagnostics.
    rtg, horizon = compute_returns_to_go_horizons(
        rollout.reward, dones, gamma=agent_config.gamma
    )

    aux = UDRLAuxiliaries(
        actor_loss=update_losses.mean(),
        mean_rtg=rtg.mean(),
        mean_horizon=horizon.mean(),
        episodic_return=agent_state.collector_state.episodic_mean_return,
        n_completed_episodes=dones.sum(),
        timestep=agent_state.collector_state.timestep,
    )
    agent_state = agent_state.replace(
        n_updates=agent_state.n_updates + 1,
        command_target_return=new_target_r,
        command_target_return_std=new_target_std,
        command_target_horizon=new_target_h,
    )
    return agent_state, aux


@dataclasses.dataclass(frozen=True)
class EvalCommand:
    """The command UDRL is evaluated with (:func:`ajax.evaluate.evaluate`'s
    ``command``): each episode starts from the current target return and
    horizon, without the exploration noise collection adds, then decays as
    in collection (:func:`update_command`; past an episode's end the
    evaluation counts nothing), the actor reading it scaled. Under reward
    normalisation the targets are in normalised units, as the rewards
    collection decays the command by: evaluate() hands ``update`` the
    rewards normalised with the training env's frozen statistics."""

    scale_r: float
    scale_h: float

    def initial(self, agent_state: UDRLState, n_envs: int) -> jax.Array:
        target = (agent_state.command_target_return, agent_state.command_target_horizon)
        return jnp.tile(jnp.stack(target), (n_envs, 1))

    def update(self, command: jax.Array, reward: jax.Array, done: jax.Array):
        d_r, d_h = command[:, 0], command[:, 1]
        d_r, d_h = update_command(d_r, d_h, reward, done.astype(jnp.float32), d_r, d_h)
        return jnp.stack([d_r, d_h], axis=-1)

    def actor_input(self, obs: jax.Array, command: jax.Array) -> jax.Array:
        obs = jnp.concatenate([obs, command.astype(obs.dtype)], axis=-1)
        return _scale_obs_command(obs, self.scale_r, self.scale_h)


def evaluate_UDRL(
    agent_state: UDRLState,
    key: jax.Array,
    env_args: EnvironmentConfig,
    agent_config: UDRLConfig,
    num_episode_test: int,
) -> dict:
    """The deterministic policy's return, entropy and length on
    ``num_episode_test`` episodes of the shared evaluation, commanded by
    :class:`EvalCommand`, and the return and horizon each episode is
    commanded with at its start."""
    returns, entropy, _, _, length, _ = evaluate(
        env_args.env,
        agent_state.actor_state,
        num_episode_test,
        key,
        env_args.env_params,
        norm_info=eval_norm_info(agent_state, env_args.mode),
        agent_state=agent_state,
        command=EvalCommand(agent_config.command_scale_r, agent_config.command_scale_h),
    )
    return {
        "Eval/episodic mean reward": returns,
        "Eval/episodic entropy": entropy,
        "Eval/mean episodic length": length,
        "Eval/commanded return": agent_state.command_target_return,
        "Eval/commanded horizon": agent_state.command_target_horizon,
    }


def train_metrics(agent_state: UDRLState, aux: UDRLAuxiliaries) -> dict:
    """The logged training metrics: the iteration's segment and updates."""
    del agent_state
    return {
        "timestep": aux.timestep,
        "Train/episodic mean reward": aux.episodic_return,
        "Train/actor_loss": aux.actor_loss,
        "Train/mean_rtg": aux.mean_rtg,
        "Train/mean_horizon": aux.mean_horizon,
        "Train/n_completed_episodes": aux.n_completed_episodes,
    }


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: UDRLConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    extensions: Sequence = (),
):
    """UDRL's train function: its own iteration (a segment, the replay of
    segments, the command statistics, the updates) on :meth:`TrainLoop.train`.

    With a logging config every ``log_frequency`` env steps it logs the
    iteration's training metrics, its policy's evaluation
    (:func:`evaluate_UDRL`) and the extensions' ``eval_metrics``, whether or
    not a backend records them, and ``train`` returns them.
    """
    mode = env_args.mode
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
    )
    per_iter = env_args.n_envs * agent_config.n_steps
    loop.warn_if_partial(per_iter, "n_envs x n_steps")
    num_iterations = max(loop.budget // per_iter, 1)
    evaluation = Evaluation(
        metrics=train_metrics,
        every=loop.log_frequency and max(loop.log_frequency // per_iter, 1),
        evaluate=partial(
            evaluate_UDRL,
            env_args=env_args,
            agent_config=agent_config,
            num_episode_test=num_episode_test,
        ),
    )

    def init(key: jax.Array) -> UDRLState:
        return init_UDRL(
            key=key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            agent_config=agent_config,
        )

    def iteration(agent_state: UDRLState, index: Any, tick: Any) -> tuple:
        agent_state, aux = training_iteration(
            agent_state,
            None,
            env_args=env_args,
            mode=mode,
            agent_config=agent_config,
            extension_stack=loop.stack,
            total_timesteps=total_timesteps,
        )
        return loop.evaluate_every(agent_state, aux, index, tick, evaluation)

    return loop.train(
        init, iteration, num_iterations, evaluations=evaluation.count(num_iterations)
    )
