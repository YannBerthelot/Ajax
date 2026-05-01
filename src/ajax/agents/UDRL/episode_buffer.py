"""Per-episode replay buffer with top-K-by-return eviction (paper-faithful).

Schmidhuber et al. 2019 Algorithm 1 keeps the highest-return episodes in a
fixed-size buffer (700 in the paper). When the buffer is full and a new
episode arrives, the lowest-return episode is replaced if the new one's
return exceeds it; otherwise the new episode is dropped.

This module stores complete, padded-to-T_max individual episodes. It's
used from the Gymnasium-driven LunarLander training loop (Box2D step is
CPU-native, so we maintain in-progress episodes in Python and call the
JIT'd ``insert_episode`` once per episode end).

The original ``buffer.py`` (segment-based, FIFO over n_steps-long rollout
segments) is kept for the jax-pure on-policy UDRL path (CartPole etc.)
where per-episode bookkeeping inside JIT would be awkward.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EpisodeBuffer:
    obs: jnp.ndarray  # (B, T_max, obs_dim)
    actions: jnp.ndarray  # (B, T_max, action_dim)
    rewards: jnp.ndarray  # (B, T_max, 1)
    cum_rewards: jnp.ndarray  # (B, T_max + 1, 1) prefix sum, cum[..., 0]=0
    lengths: jnp.ndarray  # (B,) int32  - actual episode length (<= T_max)
    returns: jnp.ndarray  # (B,) float32 - episode return for sort/eviction
    fill_count: jnp.ndarray  # int32


def init_episode_buffer(
    capacity: int,
    T_max: int,
    obs_dim: int,
    action_dim: int,
    action_dtype=jnp.float32,
) -> EpisodeBuffer:
    return EpisodeBuffer(
        obs=jnp.zeros((capacity, T_max, obs_dim), dtype=jnp.float32),
        actions=jnp.zeros((capacity, T_max, action_dim), dtype=action_dtype),
        rewards=jnp.zeros((capacity, T_max, 1), dtype=jnp.float32),
        cum_rewards=jnp.zeros((capacity, T_max + 1, 1), dtype=jnp.float32),
        lengths=jnp.zeros((capacity,), dtype=jnp.int32),
        # Initialise empty-slot returns to -inf so any real episode is
        # preferable when the buffer is filling up; eviction picks argmin.
        returns=jnp.full((capacity,), -jnp.inf, dtype=jnp.float32),
        fill_count=jnp.asarray(0, dtype=jnp.int32),
    )


def insert_episode(
    buffer: EpisodeBuffer,
    ep_obs: jnp.ndarray,  # (T_max, obs_dim) padded
    ep_actions: jnp.ndarray,  # (T_max, action_dim) padded
    ep_rewards: jnp.ndarray,  # (T_max, 1) padded with zeros
    ep_length: jnp.ndarray,  # scalar int32 (<= T_max)
    ep_return: jnp.ndarray,  # scalar float32
) -> EpisodeBuffer:
    """Insert one episode using the paper's top-K-by-return eviction policy.

    - If the buffer is not full: write to the next free slot.
    - Else: replace the lowest-return slot if the new return is higher;
      otherwise drop the new episode.

    The cum_rewards prefix sum is computed inline so callers don't have to.
    """
    capacity = buffer.obs.shape[0]
    is_full = buffer.fill_count >= capacity
    lowest_idx = jnp.argmin(buffer.returns)
    lowest_return = buffer.returns[lowest_idx]
    insert_idx = jnp.where(is_full, lowest_idx, buffer.fill_count)
    should_insert = jnp.logical_or(jnp.logical_not(is_full), ep_return > lowest_return)

    # Prefix-sum the reward stream up-front (zeros past length, so cum stays
    # flat there — fine because sampling restricts t1, t2 <= ep_length).
    cum = jnp.concatenate(
        [jnp.zeros_like(ep_rewards[:1]), jnp.cumsum(ep_rewards, axis=0)], axis=0
    )

    # Conditional update: only the chosen slot is touched. Compute the new
    # per-slot value (either the incoming episode or the existing slot
    # contents) and write it back; this avoids an O(B*T_max) jnp.where.
    def slot_value(new_val, old_slot):
        return jnp.where(should_insert, new_val, old_slot)

    obs = buffer.obs.at[insert_idx].set(
        slot_value(ep_obs.astype(buffer.obs.dtype), buffer.obs[insert_idx])
    )
    actions = buffer.actions.at[insert_idx].set(
        slot_value(ep_actions.astype(buffer.actions.dtype), buffer.actions[insert_idx])
    )
    rewards = buffer.rewards.at[insert_idx].set(
        slot_value(ep_rewards.astype(buffer.rewards.dtype), buffer.rewards[insert_idx])
    )
    cum_rewards = buffer.cum_rewards.at[insert_idx].set(
        slot_value(cum.astype(buffer.cum_rewards.dtype), buffer.cum_rewards[insert_idx])
    )
    lengths = buffer.lengths.at[insert_idx].set(
        slot_value(ep_length.astype(buffer.lengths.dtype), buffer.lengths[insert_idx])
    )
    returns = buffer.returns.at[insert_idx].set(
        slot_value(ep_return.astype(buffer.returns.dtype), buffer.returns[insert_idx])
    )
    fill_count = jnp.where(
        jnp.logical_and(should_insert, jnp.logical_not(is_full)),
        buffer.fill_count + 1,
        buffer.fill_count,
    )
    return buffer.replace(
        obs=obs,
        actions=actions,
        rewards=rewards,
        cum_rewards=cum_rewards,
        lengths=lengths,
        returns=returns,
        fill_count=fill_count,
    )


def sample_training_batch(
    rng: jax.Array, buffer: EpisodeBuffer, batch_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample (state, action, dr, dh) for a training batch (Algorithm 3).

    For each batch element:
      - episode ~ Uniform([0, fill_count))
      - t1 ~ Uniform([0, length-1))
      - t2 ~ Uniform((t1, length])
      - dr = sum(rewards[t1:t2]),  dh = t2 - t1
      - state[..., -2:] is overwritten with (dr, dh)
    """
    rng, k_ep, k_t1, k_t2 = jax.random.split(rng, 4)
    valid = jnp.maximum(buffer.fill_count, 1)
    ep_idx = jax.random.randint(k_ep, (batch_size,), 0, valid)
    lengths = buffer.lengths[ep_idx]  # (batch,)

    t1_max = jnp.maximum(lengths - 1, 1)
    t1 = jax.random.randint(k_t1, (batch_size,), 0, t1_max)
    span = jnp.maximum(lengths - 1 - t1, 0) + 1
    t2 = t1 + 1 + jax.random.randint(k_t2, (batch_size,), 0, span)

    cum = buffer.cum_rewards[ep_idx]  # (batch, T_max+1, 1)
    batch_arange = jnp.arange(batch_size)
    dr = cum[batch_arange, t2, 0] - cum[batch_arange, t1, 0]
    dh = (t2 - t1).astype(jnp.float32)

    state = buffer.obs[ep_idx, t1]  # (batch, obs_dim)
    action = buffer.actions[ep_idx, t1]  # (batch, action_dim)

    state_no_cmd = state[..., :-2]
    new_cmd = jnp.stack([dr, dh], axis=-1)
    state = jnp.concatenate([state_no_cmd, new_cmd], axis=-1)
    return state, action, dr, dh


def topk_command_stats(
    buffer: EpisodeBuffer, k: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (mean_R, std_R, mean_H, n_valid) over the buffer's top-K episodes.

    Since insert_episode already enforces top-K-by-return semantics, this is
    just argpartition + mean/std on returns and lengths.
    """
    capacity = buffer.returns.shape[0]
    valid = jnp.arange(capacity) < buffer.fill_count
    masked_returns = jnp.where(valid, buffer.returns, -jnp.inf)
    k_eff = min(k, capacity)
    topk_vals, topk_idx = jax.lax.top_k(masked_returns, k_eff)
    valid_mask = topk_vals > -jnp.inf
    valid_count = valid_mask.sum()
    safe_count = jnp.maximum(valid_count, 1)

    topk_h = buffer.lengths[topk_idx].astype(jnp.float32)
    sum_r = jnp.where(valid_mask, topk_vals, 0.0).sum()
    sum_r2 = jnp.where(valid_mask, topk_vals * topk_vals, 0.0).sum()
    sum_h = jnp.where(valid_mask, topk_h, 0.0).sum()

    mean_r = jnp.where(valid_count > 0, sum_r / safe_count, jnp.nan)
    var_r = jnp.maximum(sum_r2 / safe_count - mean_r * mean_r, 0.0)
    std_r = jnp.where(valid_count > 1, jnp.sqrt(var_r), 0.0)
    mean_h = jnp.where(valid_count > 0, sum_h / safe_count, jnp.nan)
    return mean_r, std_r, mean_h, valid_count
