from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial


@partial(jax.jit, static_argnames=["gamma", "gae_lambda"])
def _compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminateds: jax.Array,
    truncateds: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute gae advantages

    Args:
        traj_batch (Transition): The transition buffer
        last_val (jax.Array): The value of the last state encoutered.
        gamma (float): The discount factor to consider.
        gae_lambda (float): The gae lambda parameter to consider.

    Returns:
        tuple[jax.Array, jax.Array]: Gae and value (carry-over values to feed for next\
              iteration)\
            and the gae for actual return
    """

    def _get_advantages(
        gae: jax.Array,
        transition,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Iteratively compute advantages in gae style using previous gae, next value\
              and transition buffer

        Args:
            gae_and_next_value (tuple[jax.Array, jax.Array]): Previous gae and next_value
            transition (Transition): The transitions to consider

        Returns:
            tuple[tuple[jax.Array, jax.Array], jax.Array]: The updated gaes + \
                the transition's values
        """
        # current estimation of gae + value at t+1 because we have working in reverse
        reward, value, next_value, terminated, truncated = transition

        next_state_is_non_terminal = 1.0 - terminated
        done = 1 - jnp.logical_or(terminated, truncated)

        delta = reward + gamma * next_value * next_state_is_non_terminal - value
        gae = delta + gamma * gae_lambda * done * gae

        # tuple is carry-over state for scan, gae after the comma is the actual return at the end of the scan
        return gae, gae

    _, advantages = jax.lax.scan(
        f=_get_advantages,
        init=jnp.zeros_like(values[-1]),
        xs=(rewards, values, next_values, terminateds, truncateds),
        reverse=True,
        unroll=1,
    )
    returns = advantages + values
    return advantages, returns


def split_fragments(
    batch: Any, perm: jax.Array, num_minibatches: int, length: int
) -> Any:
    """Split a ``(T, n_envs, ...)`` rollout into ``num_minibatches``
    minibatches of whole fragments of ``length`` steps. Fragment ``c *
    n_envs + e`` (chunk ``c`` of env ``e``) goes in the order of ``perm``;
    per leaf ``(num_minibatches, length, fragments_per_minibatch, ...)``,
    so a scan over the minibatches sees time-major fragments."""
    T, n_envs = jax.tree.leaves(batch)[0].shape[:2]
    n_chunks = T // length

    def split(x: jax.Array) -> jax.Array:
        # (T, n_envs, ...) -> (n_chunks * n_envs, length, ...)
        x = x.reshape(n_chunks, length, n_envs, *x.shape[2:])
        x = jnp.swapaxes(x, 1, 2).reshape(n_chunks * n_envs, length, *x.shape[3:])
        x = jnp.take(x, perm, axis=0)
        x = x.reshape(num_minibatches, -1, length, *x.shape[2:])
        return jnp.swapaxes(x, 1, 2)

    return jax.tree.map(split, batch)


@partial(jax.jit, static_argnames=["num_minibatches", "unroll_length"])
def get_minibatches_preserving_time(
    batch: Any,
    rng: jax.Array,
    num_minibatches: int,
    unroll_length: Optional[int] = None,
) -> Any:
    """Split a ``(T, n_envs, ...)`` rollout into minibatches that keep the
    time axis (brax PPO): fragments of ``unroll_length`` steps (the whole
    rollout by default), shuffled and split evenly
    (:func:`split_fragments`). GAE in the loss then scans each fragment and
    bootstraps at its end, brax's per-minibatch ``(T=unroll_length,
    B=batch_size)``."""
    T, n_envs = jax.tree.leaves(batch)[0].shape[:2]
    length = unroll_length or T
    assert T % length == 0, f"n_steps={T} must be a multiple of {length=}."
    n_fragments = T // length * n_envs
    assert n_fragments % num_minibatches == 0, (
        f"{n_fragments} fragments (n_steps={T} / {length=} * n_envs={n_envs})"
        f" do not split into {num_minibatches=}."
    )
    perm = jax.random.permutation(rng, n_fragments)
    return split_fragments(batch, perm, num_minibatches, length)


def get_minibatches_from_batch(batch: Any, rng: jax.Array, num_minibatches: int):
    """Split a ``(T, n_envs, ...)`` rollout batch into ``num_minibatches``
    minibatches of shape ``(num_minibatches, T*n_envs/num_minibatches, ...)``.

    The flatten step joins the leading ``(T, n_envs)`` dims into a single
    sample axis regardless of how many trailing feature dims a leaf has,
    so leaves with a feature dim (e.g. observations ``(T, n_envs, obs_dim)``)
    and leaves without one (e.g. ``terminated``, ``truncated`` of shape
    ``(T, n_envs)``) are reshuffled with the same permutation and stay
    aligned across leaves.
    """
    T, n_envs = jax.tree.leaves(batch)[0].shape[:2]
    total = T * n_envs
    assert total % num_minibatches == 0, (
        "T * n_envs should be a multiple of num_minibatches, got "
        f"T={T}, n_envs={n_envs}, total={total}, {num_minibatches=}"
    )

    # Flatten the leading (T, n_envs) dims into a single sample axis for
    # every leaf, regardless of trailing feature dims.
    new_batch = jax.tree_util.tree_map(
        lambda x: x.reshape((total,) + x.shape[2:]), batch
    )
    # Same rng + same leading dim => identical permutation across leaves
    # => sample alignment preserved.
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jax.random.permutation(rng, x, axis=0), new_batch
    )
    # Split the flat sample axis into num_minibatches chunks.
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (num_minibatches, -1) + x.shape[1:]),
        shuffled_batch,
    )
    return minibatches
