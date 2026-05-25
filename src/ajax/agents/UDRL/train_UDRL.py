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

from collections.abc import Sequence
from typing import Any, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from jax.tree_util import Partial as partial

from ajax.agents.SAC.utils import SquashedNormal
from ajax.agents.UDRL.buffer import (
    add_segment,
    init_buffer,
    sample_training_batch,
    topk_command_stats,
)
from ajax.agents.UDRL.state import UDRLConfig, UDRLState
from ajax.agents.UDRL.utils import (
    compute_returns_to_go_horizons,
)
from ajax.environments.interaction import (
    get_pi,
    init_collector_state,
    step,
)
from ajax.environments.utils import (
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
)
from ajax.extensions.base import ExtensionStack
from ajax.networks.networks import get_initialized_actor_critic
from ajax.perf_utils import train_jit
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)


@struct.dataclass
class UDRLAuxiliaries:
    actor_loss: jnp.ndarray
    mean_rtg: jnp.ndarray
    mean_horizon: jnp.ndarray
    # Mean completed-episode return within the rollout segment (0 if no
    # episode finished). Plotted as the training curve.
    episodic_return: jnp.ndarray
    # Number of completed episodes in the segment (per iter).
    n_completed_episodes: jnp.ndarray
    # Total env timesteps consumed up to and including this iter.
    timestep: jnp.ndarray


def init_UDRL(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: UDRLConfig,
    window_size: int = 10,
    cnn_image_shape: Optional[Tuple[int, int, int]] = None,
) -> UDRLState:
    rng, init_key, collector_key = jax.random.split(key, 3)

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
        cnn_image_shape=cnn_image_shape,
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        window_size=window_size,
        expert_state_aug_dim=2,  # reserves 2 trailing slots in last_obs for command
    )

    # Seed last_obs's command slots with (return_init, horizon_init).
    n_envs = env_args.n_envs
    init_cmd = jnp.tile(
        jnp.array(
            [agent_config.command_return_init, agent_config.command_horizon_init],
            dtype=collector_state.last_obs.dtype,
        ),
        (n_envs, 1),
    )
    new_last_obs = jnp.concatenate(
        [collector_state.last_obs[:, :-2], init_cmd], axis=-1
    )
    collector_state = collector_state.replace(last_obs=new_last_obs)

    obs_dim = collector_state.last_obs.shape[-1]
    # The buffer stores actions with a trailing dim (1 for discrete, action_dim
    # for continuous). Read it from the freshly-built actor's expected action
    # space dim.
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
        eval_rng=rng,
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
    recurrent: bool,
):
    """One env step. Reads (d_r, d_h) from the trailing 2 dims of last_obs,
    forwards the actor on the augmented obs, steps the env, then writes the
    next augmented obs (raw_next_obs ++ updated_command) back into
    collector_state.last_obs.
    """
    cs = agent_state.collector_state
    rng, action_key, step_key = jax.random.split(agent_state.rng, 3)
    rng_step = (
        jax.random.split(step_key, env_args.n_envs) if mode == "gymnax" else step_key
    )

    last_obs = cs.last_obs  # (n_envs, raw_obs_dim + 2)
    # Scale (dr, dh) before the actor's forward pass (paper §appendix). The
    # underlying buffer keeps raw values; the actor input is the scaled view.
    last_obs_for_actor = _scale_obs_command(
        last_obs, agent_config.command_scale_r, agent_config.command_scale_h
    )
    pi, new_actor_state = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=last_obs_for_actor,
        recurrent=recurrent,
    )
    action, log_probs = pi.sample_and_log_prob(seed=action_key)

    new_env_obs, new_env_state, reward, terminated, truncated, _ = (
        jax.lax.stop_gradient(
            step(
                rng_step,
                cs.env_state,
                action,
                env_args.env,
                mode,
                env_args.env_params,
            )
        )
    )
    done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool)).astype(
        jnp.float32
    )

    prev_d_r = last_obs[:, -2]
    prev_d_h = last_obs[:, -1]
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
    # update_command uses sampled_init_R/H for envs that just hit done; envs
    # mid-episode get the decayed (prev - reward, prev - 1).
    decayed_r = prev_d_r - reward
    decayed_h = jnp.maximum(prev_d_h - 1.0, 1.0)
    new_d_r = jnp.where(done > 0.0, sampled_init_R, decayed_r)
    new_d_h = jnp.where(done > 0.0, sampled_init_H, decayed_h)
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

    new_cs = cs.replace(
        rng=rng,
        _env_state=new_env_state,
        last_obs=new_last_obs,
        timestep=cs.timestep + env_args.n_envs,
        last_terminated=terminated,
        last_truncated=truncated,
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
    bc_loss_type: str,
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

    if bc_loss_type == "mse":
        if hasattr(pi, "unsquashed_mean"):
            mu = pi.unsquashed_mean()
        else:
            mu = pi.mean()
        target = jnp.arctanh(jnp.clip(actions, -0.999, 0.999))
        return jnp.mean((mu - target) ** 2)

    # NLL on the squashed-normal / normal
    if isinstance(pi, SquashedNormal):
        eps = 1e-3
        target = jnp.clip(actions, -1.0 + eps, 1.0 - eps)
        return -pi.log_prob(target).sum(-1).mean()
    return -pi.log_prob(actions).sum(-1).mean()


_actor_value_and_grad = jax.value_and_grad(actor_loss_fn)


def _replace_command_with_realised(
    obs: jax.Array, rtg: jax.Array, horizon: jax.Array
) -> jax.Array:
    """Overwrite the trailing 2 dims of obs with the per-step realised
    (RTG, horizon)."""
    base = obs[..., :-2]
    rtg = rtg.reshape(rtg.shape[:-1])  # drop trailing 1 if present
    horizon = horizon.reshape(horizon.shape[:-1])
    cmd = jnp.stack([rtg, horizon], axis=-1)
    return jnp.concatenate([base, cmd], axis=-1)


def _completed_episode_returns(
    rewards: jax.Array, dones: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Sum rewards into per-env episode returns, emitting at episode end.

    rewards, dones: (T, n_envs, 1). Returns (sum_of_completed_returns,
    n_completed_episodes), both scalars summed over the (T, n_envs) axes.
    """

    def body(carry, x):
        running, sum_done, count = carry
        r, d = x
        new_running = running + r
        sum_done = sum_done + d * new_running
        count = count + d
        new_running = new_running * (1.0 - d)
        return (new_running, sum_done, count), None

    init = (
        jnp.zeros_like(rewards[0]),
        jnp.zeros_like(rewards[0]),
        jnp.zeros_like(rewards[0]),
    )
    (_, sum_done, count), _ = jax.lax.scan(body, init, (rewards, dones))
    return sum_done.sum(), count.sum()


def _topk_episode_stats(
    rewards: jax.Array, dones: jax.Array, k: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Return (mean_topk_return, mean_topk_horizon, n_completed) for the
    top-``k`` completed episodes in the rollout, by return.

    Episodes that did not complete in this segment contribute -inf to the
    return ranking (so they're dropped) and their horizon is masked. If
    fewer than k episodes completed, only the valid ones contribute to the
    means; mean is NaN if zero completed.
    """

    def body(carry, x):
        running_r, running_h = carry
        r, d = x
        running_r = running_r + r
        running_h = running_h + 1.0
        ep_r = running_r
        ep_h = running_h
        running_r = running_r * (1.0 - d)
        running_h = running_h * (1.0 - d)
        return (running_r, running_h), (ep_r, ep_h, d)

    init = (jnp.zeros_like(rewards[0]), jnp.zeros_like(rewards[0]))
    _, (ep_r, ep_h, d) = jax.lax.scan(body, init, (rewards, dones))
    flat_r = ep_r.reshape(-1)
    flat_h = ep_h.reshape(-1)
    flat_d = d.reshape(-1)
    # Mask non-done positions out of the top-k ranking.
    masked_r = jnp.where(flat_d > 0, flat_r, -jnp.inf)
    k_eff = min(k, flat_r.shape[0])
    topk_vals, topk_idx = jax.lax.top_k(masked_r, k_eff)
    topk_h = flat_h[topk_idx]
    valid = topk_vals > -jnp.inf
    valid_count = valid.sum()
    safe_count = jnp.maximum(valid_count, 1)
    sum_r = jnp.where(valid, topk_vals, 0.0).sum()
    sum_h = jnp.where(valid, topk_h, 0.0).sum()
    mean_r = jnp.where(valid_count > 0, sum_r / safe_count, jnp.nan)
    mean_h = jnp.where(valid_count > 0, sum_h / safe_count, jnp.nan)
    n_completed = flat_d.sum()
    return mean_r, mean_h, n_completed


def _shuffle_and_minibatch(
    rng: jax.Array, arrays: Tuple[jax.Array, ...], batch_size: int
):
    """Return a tuple of shape (n_batches, batch_size, ...) per-array.
    The leading axis is partial-batch-padded by repeating earlier samples;
    UDRL's loss is a mean so duplicates do not bias gradients meaningfully
    on small budgets, and this keeps the function shape-static for jit.
    """
    n = arrays[0].shape[0]
    # Effective batch_size never exceeds n (avoids 0-length minibatches when
    # callers pass n_steps * n_envs < batch_size).
    bs = min(batch_size, n)
    n_batches = n // bs
    perm = jax.random.permutation(rng, n)
    take = n_batches * bs
    perm = perm[:take]
    out = tuple(a[perm].reshape((n_batches, bs) + a.shape[1:]) for a in arrays)
    return out


def training_iteration(
    agent_state: UDRLState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    agent_config: UDRLConfig,
    recurrent: bool,
    extension_stack: Optional[ExtensionStack] = None,
    total_timesteps: int = 1,
) -> Tuple[UDRLState, UDRLAuxiliaries]:
    # 1. Collect one rollout segment (Algorithm 4).
    collect_fn = partial(
        _udrl_collect_step,
        env_args=env_args,
        mode=mode,
        agent_config=agent_config,
        recurrent=recurrent,
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

    # 3. Refresh the rollout command target from buffer top-K (Algorithm 5).
    topk_r, topk_std, topk_h, _ = topk_command_stats(
        new_buffer, k=agent_config.command_topk
    )
    proposal_r = topk_r * agent_config.command_return_boost
    proposal_h = topk_h
    proposal_r = jnp.where(
        jnp.isnan(proposal_r), agent_state.command_target_return, proposal_r
    )
    proposal_std = jnp.where(
        jnp.isnan(topk_std), agent_state.command_target_return_std, topk_std
    )
    proposal_h = jnp.where(
        jnp.isnan(proposal_h), agent_state.command_target_horizon, proposal_h
    )
    tau = agent_config.command_target_tau
    new_target_r = (1.0 - tau) * agent_state.command_target_return + tau * proposal_r
    new_target_std = (
        1.0 - tau
    ) * agent_state.command_target_return_std + tau * proposal_std
    new_target_h = (1.0 - tau) * agent_state.command_target_horizon + tau * proposal_h

    # 4. Train: sample (s, a, dr, dh) from the buffer and fit the actor
    #    (Algorithm 3). The paper does a fixed number of gradient updates
    #    per iteration; we replace the previous "n_epochs over current
    #    rollout" loop with that.
    rng, train_rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)

    def update_step(carry, key):
        a_state = carry
        obs_b, act_b, _dr, _dh = sample_training_batch(
            key, a_state.buffer, agent_config.batch_size
        )
        # Scale the (dr, dh) portion of obs to match what the actor sees
        # at rollout time.
        obs_b = _scale_obs_command(
            obs_b, agent_config.command_scale_r, agent_config.command_scale_h
        )

        def _loss(params):
            loss = actor_loss_fn(
                params, a_state.actor_state, obs_b, act_b, agent_config.bc_loss_type
            )
            if extension_stack is not None:
                _al_batch = {
                    "observations": obs_b,
                    "actions": act_b,
                    "actor_params": params,
                    "actor_state": a_state.actor_state,
                }
                loss = loss + extension_stack.fold_actor_loss(
                    a_state,
                    _al_batch,
                    a_state.collector_state.timestep,
                    key,
                    total_timesteps,
                )
            return loss

        loss, grads = jax.value_and_grad(_loss)(a_state.actor_state.params)
        new_actor_state = a_state.actor_state.apply_gradients(grads=grads)
        return a_state.replace(actor_state=new_actor_state), loss

    update_keys = jax.random.split(train_rng, agent_config.n_updates_per_iter)
    agent_state, update_losses = jax.lax.scan(update_step, agent_state, update_keys)

    # Extension post_update — folded after the per-iteration update loop.
    # Empty stack ⇒ identity.
    if extension_stack is not None:
        _pu_rng, _pu_rng2 = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=_pu_rng2)
        agent_state = extension_stack.fold_post_update(
            agent_state,
            agent_state.collector_state.timestep,
            _pu_rng,
            total_timesteps,
        )

    # 5. Compute training-curve metric: mean completed-episode return in this
    #    segment. (Used purely for logging; not for training signal.)
    sum_returns, n_completed = _completed_episode_returns(rollout.reward, dones)
    mean_episode_return = jnp.where(
        n_completed > 0, sum_returns / jnp.maximum(n_completed, 1.0), jnp.nan
    )

    # Realised RTG / horizon over the segment, only used for diagnostics.
    rtg, horizon = compute_returns_to_go_horizons(
        rollout.reward, dones, gamma=agent_config.gamma
    )

    aux = UDRLAuxiliaries(
        actor_loss=update_losses.mean(),
        mean_rtg=rtg.mean(),
        mean_horizon=horizon.mean(),
        episodic_return=mean_episode_return,
        n_completed_episodes=n_completed,
        timestep=agent_state.collector_state.timestep,
    )
    agent_state = agent_state.replace(
        n_updates=agent_state.n_updates + 1,
        command_target_return=new_target_r,
        command_target_return_std=new_target_std,
        command_target_horizon=new_target_h,
    )
    return agent_state, aux


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: UDRLConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[Any] = None,
    cnn_image_shape: Optional[Tuple[int, int, int]] = None,
    extensions: Sequence = (),
    **_unused: Any,
):
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    recurrent = network_args.lstm_hidden_size is not None
    extension_stack = ExtensionStack(extensions) if extensions else None

    @train_jit
    def train(
        key: jax.Array,
        index: Optional[int] = None,
        initial_state: Optional[UDRLState] = None,
        resume_from_state: bool = False,
    ):
        init_key, ext_key = jax.random.split(key, 2)
        if resume_from_state and initial_state is not None:
            agent_state = initial_state
        else:
            agent_state = init_UDRL(
                key=init_key,
                env_args=env_args,
                actor_optimizer_args=actor_optimizer_args,
                critic_optimizer_args=critic_optimizer_args,
                network_args=network_args,
                agent_config=agent_config,
                cnn_image_shape=cnn_image_shape,
            )
            if extension_stack is not None:
                _ext_key, _pre_key = jax.random.split(ext_key)
                agent_state = extension_stack.fold_init_states(agent_state, _ext_key)
                agent_state = extension_stack.fold_pretrain(
                    agent_state, jnp.asarray(0), _pre_key, total_timesteps
                )

        per_iter = env_args.n_envs * agent_config.n_steps
        num_updates = max(total_timesteps // per_iter, 1)

        scan_fn = partial(
            training_iteration,
            env_args=env_args,
            mode=mode,
            agent_config=agent_config,
            recurrent=recurrent,
            extension_stack=extension_stack,
            total_timesteps=total_timesteps,
        )
        agent_state, aux = jax.lax.scan(
            scan_fn, agent_state, xs=None, length=num_updates
        )
        return agent_state, aux

    return train
