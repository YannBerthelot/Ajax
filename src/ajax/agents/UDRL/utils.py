"""JAX-pure helpers for Upside-Down RL: command updates and per-step
return-to-go / horizon-to-go computation."""

from typing import Tuple

import jax
import jax.numpy as jnp


def update_command(
    prev_d_r: jax.Array,
    prev_d_h: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    return_init: float,
    horizon_init: float,
) -> Tuple[jax.Array, jax.Array]:
    """Per-step update of the (desired_return, desired_horizon) command vector.

    Within an episode, d_r is decremented by the realised reward and d_h by 1.
    On episode end (done=1), both are reset to their init values so the next
    episode starts with a fresh command.
    The horizon is floored at 1.0 to avoid 0/negative conditioning if the
    actual episode runs longer than the desired horizon.
    """
    decayed_d_r = prev_d_r - reward
    decayed_d_h = jnp.maximum(prev_d_h - 1.0, 1.0)
    new_d_r = jnp.where(done > 0.0, return_init, decayed_d_r)
    new_d_h = jnp.where(done > 0.0, horizon_init, decayed_d_h)
    return new_d_r, new_d_h


def compute_returns_to_go_horizons(
    rewards: jax.Array,
    dones: jax.Array,
    gamma: float = 1.0,
) -> Tuple[jax.Array, jax.Array]:
    """Compute per-step return-to-go and steps-until-done over a rollout.

    rewards, dones: leading axis is time (shape (T, ...)). dones[t] is 1 when
    the episode terminated/truncated at step t. The reverse scan resets the
    accumulator on dones, so within each segment-of-an-episode RTG covers
    exactly the rewards from t up to (and including) the terminating step.
    """

    def step(carry, x):
        G_prev, h_prev = carry
        r, d = x
        not_done = 1.0 - d
        G = r + gamma * G_prev * not_done
        h = 1.0 + h_prev * not_done
        return (G, h), (G, h)

    init = (jnp.zeros_like(rewards[0]), jnp.zeros_like(rewards[0]))
    _, (G_seq, h_seq) = jax.lax.scan(
        step, init, (rewards[::-1], dones[::-1])
    )
    return G_seq[::-1], h_seq[::-1]
