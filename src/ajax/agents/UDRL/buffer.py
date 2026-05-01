"""Segment-based episode replay buffer for UDRL (Schmidhuber 2019).

Each buffer slot stores one rollout segment of shape ``(T, n_envs, ...)``.
Episode boundaries inside a segment are tracked via the per-step ``done``
flag. At training time we sample (slot, env, t1, t2) tuples uniformly
under the constraint that no episode boundary lies strictly between t1
and t2; see ``sample_training_batch``. This implements Algorithm 3 of the
UDRL paper (interval-based RTG sampling) without the explicit
per-episode storage that would require variable-length JAX arrays.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class SegmentBuffer:
    obs: jnp.ndarray  # (B, T, n_envs, obs_dim_with_cmd)
    actions: jnp.ndarray  # (B, T, n_envs, action_dim)
    rewards: jnp.ndarray  # (B, T, n_envs, 1)
    dones: jnp.ndarray  # (B, T, n_envs, 1)  float in {0., 1.}
    cum_rewards: jnp.ndarray  # (B, T+1, n_envs, 1)  prefix sum, cum[..., 0]=0
    write_idx: jnp.ndarray  # scalar int32
    fill_count: jnp.ndarray  # scalar int32 (capped at B)


def init_buffer(
    capacity: int,
    segment_length: int,
    n_envs: int,
    obs_dim: int,
    action_dim: int,
    action_dtype=jnp.float32,
) -> SegmentBuffer:
    """Allocate an empty segment buffer.

    obs_dim is the FULL augmented obs dim (env_obs + 2 for the command).
    """
    return SegmentBuffer(
        obs=jnp.zeros((capacity, segment_length, n_envs, obs_dim), dtype=jnp.float32),
        actions=jnp.zeros(
            (capacity, segment_length, n_envs, action_dim), dtype=action_dtype
        ),
        rewards=jnp.zeros((capacity, segment_length, n_envs, 1), dtype=jnp.float32),
        dones=jnp.zeros((capacity, segment_length, n_envs, 1), dtype=jnp.float32),
        cum_rewards=jnp.zeros(
            (capacity, segment_length + 1, n_envs, 1), dtype=jnp.float32
        ),
        write_idx=jnp.asarray(0, dtype=jnp.int32),
        fill_count=jnp.asarray(0, dtype=jnp.int32),
    )


def _ensure_action_trailing_dim(actions: jnp.ndarray) -> jnp.ndarray:
    """Ensure actions have a trailing action_dim axis (=1 for discrete)."""
    if actions.ndim == 2:  # (T, n_envs)
        return actions[..., None]
    return actions


def add_segment(
    buffer: SegmentBuffer,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    dones: jnp.ndarray,
) -> SegmentBuffer:
    """Add one rollout segment at the next slot (FIFO over capacity).

    obs, actions, rewards, dones must already have shapes:
      obs:     (T, n_envs, obs_dim)
      actions: (T, n_envs, action_dim) — the discrete trailing-1 case is added here
      rewards: (T, n_envs, 1)
      dones:   (T, n_envs, 1)  in {0., 1.}
    """
    capacity = buffer.obs.shape[0]
    slot = buffer.write_idx % capacity
    actions = _ensure_action_trailing_dim(actions)
    cum = jnp.concatenate(
        [jnp.zeros_like(rewards[:1]), jnp.cumsum(rewards, axis=0)],
        axis=0,
    )  # (T+1, n_envs, 1)

    return buffer.replace(
        obs=buffer.obs.at[slot].set(obs.astype(buffer.obs.dtype)),
        actions=buffer.actions.at[slot].set(actions.astype(buffer.actions.dtype)),
        rewards=buffer.rewards.at[slot].set(rewards.astype(buffer.rewards.dtype)),
        dones=buffer.dones.at[slot].set(dones.astype(buffer.dones.dtype)),
        cum_rewards=buffer.cum_rewards.at[slot].set(
            cum.astype(buffer.cum_rewards.dtype)
        ),
        write_idx=(buffer.write_idx + 1) % capacity,
        fill_count=jnp.minimum(buffer.fill_count + 1, capacity),
    )


def _first_done_at_or_after(dones_ten: jnp.ndarray, t1: jnp.ndarray) -> jnp.ndarray:
    """Given a 1D dones vector (T,) and t1 scalar, return the first index
    t >= t1 with done=1, or T-1 if none. Pure JAX, jit-friendly.
    """
    T = dones_ten.shape[0]
    mask = jnp.arange(T) >= t1
    masked = dones_ten * mask
    any_done = jnp.any(masked > 0)
    first_done = jnp.argmax(masked)
    return jnp.where(any_done, first_done, T - 1)


def sample_training_batch(
    rng: jax.Array,
    buffer: SegmentBuffer,
    batch_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample a training batch following UDRL Algorithm 3 (interval-based).

    For each sampled (slot, env, t1):
      - end_idx = first done at t >= t1 in that segment/env (or T-1 if none).
      - t2 ~ Uniform[t1+1, end_idx+1].
      - dr = sum(rewards[slot, t1:t2, env]) (computed via prefix sum).
      - dh = t2 - t1.
    The state at t1 has its trailing 2 command dims overwritten with (dr, dh)
    so the actor is supervised on the realised command.

    Returns (obs_with_cmd, action, dr, dh) each with leading axis ``batch_size``.
    """
    T, n_envs = buffer.obs.shape[1], buffer.obs.shape[2]

    rng, k_slot, k_env, k_t1, k_t2 = jax.random.split(rng, 5)
    valid = jnp.maximum(buffer.fill_count, 1)
    slot = jax.random.randint(k_slot, (batch_size,), 0, valid)
    env_idx = jax.random.randint(k_env, (batch_size,), 0, n_envs)
    t1 = jax.random.randint(k_t1, (batch_size,), 0, T - 1)

    def _end(slot_i, env_i, t1_i):
        return _first_done_at_or_after(buffer.dones[slot_i, :, env_i, 0], t1_i)

    end_idx = jax.vmap(_end)(slot, env_idx, t1)
    span = (
        jnp.maximum(end_idx + 1 - (t1 + 1), 0) + 1
    )  # at least 1: t2 ∈ [t1+1, end_idx+1]
    t2 = t1 + 1 + jax.random.randint(k_t2, (batch_size,), 0, span)
    t2 = jnp.minimum(t2, T)  # clamp for safety

    cum = buffer.cum_rewards  # (B, T+1, n_envs, 1)
    dr = cum[slot, t2, env_idx, 0] - cum[slot, t1, env_idx, 0]
    dh = (t2 - t1).astype(jnp.float32)

    state = buffer.obs[slot, t1, env_idx]  # (batch, obs_dim)
    action = buffer.actions[slot, t1, env_idx]  # (batch, action_dim)

    state_no_cmd = state[..., :-2]
    new_cmd = jnp.stack([dr, dh], axis=-1)
    state = jnp.concatenate([state_no_cmd, new_cmd], axis=-1)
    return state, action, dr, dh


def topk_command_stats(
    buffer: SegmentBuffer, k: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute (mean_R, std_R, mean_H, n_completed) over the top-K episodes
    in the buffer (by realised return). Used to set rollout commands per
    UDRL Algorithm 5.

    A "position" is one (slot, t, env) triple where the dones flag is 1 —
    that signals an episode terminating at step t. We accumulate per-(env,
    slot) running rewards/horizons and emit them at done positions.
    """
    B = buffer.obs.shape[0]
    rewards = buffer.rewards  # (B, T, n_envs, 1)
    dones = buffer.dones  # (B, T, n_envs, 1)

    # Mask out unfilled slots so they don't contribute episodes.
    slot_valid = (jnp.arange(B) < buffer.fill_count).astype(jnp.float32)
    slot_valid_b = slot_valid[:, None, None, None]
    dones_masked = dones * slot_valid_b

    # Per-(slot, env) running sum/horizon along time, emitting at dones.
    def body(carry, x):
        running_r, running_h = carry  # each (B, n_envs, 1)
        r, d = x  # (B, n_envs, 1) each
        running_r = running_r + r
        running_h = running_h + 1.0
        ep_r = running_r
        ep_h = running_h
        running_r = running_r * (1.0 - d)
        running_h = running_h * (1.0 - d)
        return (running_r, running_h), (ep_r, ep_h, d)

    init = (jnp.zeros_like(rewards[:, 0]), jnp.zeros_like(rewards[:, 0]))
    rewards_tT = jnp.transpose(rewards, (1, 0, 2, 3))  # (T, B, n_envs, 1)
    dones_tT = jnp.transpose(dones_masked, (1, 0, 2, 3))
    _, (ep_r, ep_h, d) = jax.lax.scan(body, init, (rewards_tT, dones_tT))
    # ep_r, ep_h, d: (T, B, n_envs, 1)
    flat_r = ep_r.reshape(-1)
    flat_h = ep_h.reshape(-1)
    flat_d = d.reshape(-1)
    masked_r = jnp.where(flat_d > 0, flat_r, -jnp.inf)
    k_eff = min(k, flat_r.shape[0])
    topk_vals, topk_idx = jax.lax.top_k(masked_r, k_eff)
    topk_h = flat_h[topk_idx]
    valid = topk_vals > -jnp.inf
    valid_count = valid.sum()
    safe_count = jnp.maximum(valid_count, 1)
    sum_r = jnp.where(valid, topk_vals, 0.0).sum()
    sum_r2 = jnp.where(valid, topk_vals * topk_vals, 0.0).sum()
    sum_h = jnp.where(valid, topk_h, 0.0).sum()
    mean_r = jnp.where(valid_count > 0, sum_r / safe_count, jnp.nan)
    var_r = jnp.maximum(sum_r2 / safe_count - mean_r * mean_r, 0.0)
    std_r = jnp.where(valid_count > 1, jnp.sqrt(var_r), 0.0)
    mean_h = jnp.where(valid_count > 0, sum_h / safe_count, jnp.nan)
    n_completed = flat_d.sum()
    return mean_r, std_r, mean_h, n_completed
