"""Segment-based episode replay buffer for UDRL (Schmidhuber 2019).

Each buffer slot stores one rollout segment of shape ``(T, n_envs, ...)``.
Consecutive slots continue each env's stream (:class:`EpisodeStream`), so an
episode may span several; episode boundaries are the per-step ``done``
flags. At training time we sample (env, t1, t2) on the stream uniformly
under the constraint that no episode boundary lies strictly between t1
and t2; see ``sample_training_batch``. This implements Algorithm 3 of the
UDRL paper (interval-based RTG sampling) without the explicit
per-episode storage that would require variable-length JAX arrays.
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ajax.types import Replaceable


@struct.dataclass
class SegmentBuffer(Replaceable):
    obs: jnp.ndarray  # (B, T, n_envs, obs_dim_with_cmd)
    actions: jnp.ndarray  # (B, T, n_envs, action_dim)
    rewards: jnp.ndarray  # (B, T, n_envs, 1)
    dones: jnp.ndarray  # (B, T, n_envs, 1)  float in {0., 1.}
    # (B, T+1, n_envs, 1) per-slot prefix sum, cum[..., 0]=0. Deliberately
    # kept though nothing reads it since the sampler reads episode_stream:
    # checkpoints saved with it still restore (expand-then-contract, rule
    # 10). Drop it with a restore shim that discards the saved field.
    cum_rewards: jnp.ndarray
    # (B, n_envs, 1) 1. where the slot's first step starts an episode: what
    # the top-K statistics need of an evicted slot (``topk_command_stats``).
    episode_start: jnp.ndarray
    write_idx: jnp.ndarray  # scalar int32
    fill_count: jnp.ndarray  # scalar int32 (capped at B)
    # (B, T, n_envs, env_obs_dim) the env's own observations, which an
    # expert reads, kept only for a run with one (holds_expert): ``obs``
    # holds them normalised under normalize_observations. None otherwise.
    raw_obs: Optional[jnp.ndarray] = None


def init_buffer(
    capacity: int,
    segment_length: int,
    n_envs: int,
    obs_dim: int,
    action_dim: int,
    action_dtype=jnp.float32,
    keep_raw_obs: bool = False,
) -> SegmentBuffer:
    """Allocate an empty segment buffer.

    obs_dim is the FULL augmented obs dim (env_obs + 2 for the command).
    keep_raw_obs also keeps the env's own observations (``raw_obs``).
    """
    raw_shape = (capacity, segment_length, n_envs, obs_dim - 2)
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
        episode_start=jnp.zeros((capacity, n_envs, 1), dtype=jnp.float32),
        write_idx=jnp.asarray(0, dtype=jnp.int32),
        fill_count=jnp.asarray(0, dtype=jnp.int32),
        raw_obs=jnp.zeros(raw_shape, dtype=jnp.float32) if keep_raw_obs else None,
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
    raw_obs: Optional[jnp.ndarray] = None,
) -> SegmentBuffer:
    """Add one rollout segment at the next slot (FIFO over capacity).

    obs, actions, rewards, dones must already have shapes:
      obs:     (T, n_envs, obs_dim)
      actions: (T, n_envs, action_dim) — the discrete trailing-1 case is added here
      rewards: (T, n_envs, 1)
      dones:   (T, n_envs, 1)  in {0., 1.}
      raw_obs: (T, n_envs, obs_dim - 2), exactly when the buffer keeps them
    """
    if (raw_obs is None) != (buffer.raw_obs is None):
        raise ValueError(
            "add_segment takes the raw observations exactly when the buffer"
            " keeps them (init_buffer's keep_raw_obs)."
        )
    capacity = buffer.obs.shape[0]
    slot = buffer.write_idx  # stored modulo capacity
    actions = _ensure_action_trailing_dim(actions)
    cum = jnp.concatenate(
        [jnp.zeros_like(rewards[:1]), jnp.cumsum(rewards, axis=0)],
        axis=0,
    )  # (T+1, n_envs, 1)
    # The segment starts an episode where the previous one ended on a done,
    # or when it is the first.
    previous_end = buffer.dones[(slot - 1) % capacity, -1]
    starts = jnp.where(buffer.fill_count > 0, previous_end, 1.0)
    kept_raw = buffer.raw_obs
    if kept_raw is not None and raw_obs is not None:
        kept_raw = kept_raw.at[slot].set(raw_obs.astype(kept_raw.dtype))

    return buffer.replace(
        obs=buffer.obs.at[slot].set(obs.astype(buffer.obs.dtype)),
        actions=buffer.actions.at[slot].set(actions.astype(buffer.actions.dtype)),
        rewards=buffer.rewards.at[slot].set(rewards.astype(buffer.rewards.dtype)),
        dones=buffer.dones.at[slot].set(dones.astype(buffer.dones.dtype)),
        cum_rewards=buffer.cum_rewards.at[slot].set(
            cum.astype(buffer.cum_rewards.dtype)
        ),
        episode_start=buffer.episode_start.at[slot].set(starts),
        write_idx=(buffer.write_idx + 1) % capacity,
        fill_count=jnp.minimum(buffer.fill_count + 1, capacity),
        raw_obs=kept_raw,
    )


def sample_training_batch(
    rng: jax.Array,
    buffer: SegmentBuffer,
    stream: EpisodeStream,
    batch_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a training batch following UDRL Algorithm 3 (interval-based).

    For each (env, t1) drawn uniformly among the stream's filled steps
    (``stream`` is ``episode_stream(buffer)``):
      - t2 ~ Uniform[t1+1, last+1], ``last`` the last step of t1's episode
        in the stream (its done, or the stream's end), across slots;
      - dr = sum(rewards[t1:t2]) (from the episode's running return);
      - dh = t2 - t1.
    The state at t1 has its trailing 2 command dims overwritten with (dr, dh)
    so the actor is supervised on the realised command.

    Returns (obs_with_cmd, action), each with leading axis ``batch_size``.
    """
    slot, t, env, t1, t2 = _draw_rows(rng, buffer, stream, batch_size)
    reward = buffer.rewards[slot, t, env, 0]
    dr = stream.returns[t2 - 1, env] - stream.returns[t1, env] + reward
    dh = (t2 - t1).astype(jnp.float32)

    state = buffer.obs[slot, t, env]  # (batch, obs_dim)
    action = buffer.actions[slot, t, env]  # (batch, action_dim)

    state_no_cmd = state[..., :-2]
    new_cmd = jnp.stack([dr, dh], axis=-1)
    state = jnp.concatenate([state_no_cmd, new_cmd], axis=-1)
    return state, action


def sample_raw_observations(
    rng: jax.Array,
    buffer: SegmentBuffer,
    stream: EpisodeStream,
    batch_size: int,
) -> jnp.ndarray:
    """The env's own observations (``buffer.raw_obs``) at the rows
    :func:`sample_training_batch` reads given the same arguments."""
    if buffer.raw_obs is None:
        raise ValueError(
            "This segment buffer keeps no raw observations: a run keeps them"
            " only with an expert (init_buffer's keep_raw_obs)."
        )
    slot, t, env, _, _ = _draw_rows(rng, buffer, stream, batch_size)
    return buffer.raw_obs[slot, t, env]


def _draw_rows(
    rng: jax.Array, buffer: SegmentBuffer, stream: EpisodeStream, batch_size: int
) -> Tuple[jnp.ndarray, ...]:
    """The (slot, t, env) rows of :func:`sample_training_batch` and their
    interval's (t1, t2) on the stream."""
    T, n_envs = buffer.obs.shape[1], buffer.obs.shape[2]
    k_env, k_t1, k_t2 = jax.random.split(rng, 3)
    env = jax.random.randint(k_env, (batch_size,), 0, n_envs)
    t1 = jax.random.randint(k_t1, (batch_size,), 0, stream.length)
    span = stream.last[t1, env] - t1 + 1  # >= 1
    t2 = t1 + 1 + jax.random.randint(k_t2, (batch_size,), 0, span)
    return stream.order[t1 // T], t1 % T, env, t1, t2


@struct.dataclass
class EpisodeStream:
    """The buffer's filled slots oldest first, one stream of steps per env,
    each step placed in its episode: what the top-K statistics and the
    sampler read, computed once per iteration (:func:`episode_stream`). The
    collector runs on across iterations, so consecutive slots continue each
    env's stream and an episode may span several.
    """

    order: jnp.ndarray  # (B,) the slots, oldest first
    returns: jnp.ndarray  # (B * T, n_envs) the episode's return through the step
    horizons: jnp.ndarray  # (B * T, n_envs) the episode's steps through the step
    # (B * T, n_envs) 1. where the step ends an episode whose start the
    # stream holds (not the oldest slot's leading fragment once its start
    # was evicted).
    ends_whole: jnp.ndarray
    # (B * T, n_envs) the last step of the step's episode in the stream: its
    # done, or the stream's last filled step while the episode runs.
    last: jnp.ndarray
    length: jnp.ndarray  # scalar: the stream's filled steps, fill_count * T


def episode_stream(buffer: SegmentBuffer) -> EpisodeStream:
    """Read the buffer as :class:`EpisodeStream`: a running return and
    horizon per env, restarted after each done."""
    B, T, n_envs = buffer.rewards.shape[:3]
    # Oldest first: the FIFO's oldest slot is write_idx once it is full.
    order = (buffer.write_idx - buffer.fill_count + jnp.arange(B)) % B
    length = buffer.fill_count * T
    filled = (jnp.arange(B) < buffer.fill_count).astype(jnp.float32)
    rewards = buffer.rewards[order].reshape(B * T, n_envs, 1)
    dones = buffer.dones[order] * filled[:, None, None, None]
    dones = dones.reshape(B * T, n_envs, 1)

    def body(carry, x):
        running_r, running_h, whole = carry  # each (n_envs, 1)
        r, d = x  # (n_envs, 1) each
        running_r, running_h = running_r + r, running_h + 1.0
        out, keep = (running_r, running_h, d * whole), 1.0 - d
        # The episode after a done starts inside the stream.
        return (running_r * keep, running_h * keep, jnp.maximum(whole, d)), out

    zeros = jnp.zeros_like(rewards[0])
    init = (zeros, zeros, buffer.episode_start[order[0]])
    _, (ep_r, ep_h, ends_whole) = jax.lax.scan(body, init, (rewards, dones))
    steps = jnp.arange(B * T)[:, None]
    last = jnp.where(dones[..., 0] > 0, steps, length - 1)
    last = jax.lax.cummin(last, axis=0, reverse=True)
    return EpisodeStream(
        order, ep_r[..., 0], ep_h[..., 0], ends_whole[..., 0], last, length
    )


def topk_command_stats(
    stream: EpisodeStream, k: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute (mean_R, std_R, mean_H) over the top-K whole episodes of the
    stream (by realised return). Used to set rollout commands per UDRL
    Algorithm 5.
    """
    flat_r = stream.returns.reshape(-1)
    flat_h = stream.horizons.reshape(-1)
    flat_d = stream.ends_whole.reshape(-1)
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
    return mean_r, std_r, mean_h
