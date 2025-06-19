from typing import Optional

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial


@partial(jax.jit, static_argnames=["gamma"])
def _compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float,
    gae_lambda: float,
    last_value: jax.Array,
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
        gae_next_value: tuple[jax.Array, jax.Array, Optional[jax.Array]],
        transition,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
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
        reward, value, done = transition
        gae, next_value = gae_next_value

        next_state_is_non_terminal = 1.0 - done

        delta = reward + gamma * next_value * next_state_is_non_terminal - value
        gae = delta + gamma * gae_lambda * next_state_is_non_terminal * gae

        # tuple is carry-over state for scan, gae after the comma is the actual return at the end of the scan
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        f=_get_advantages,
        init=(jnp.zeros_like(last_value), last_value),
        xs=(rewards, values, dones),
        reverse=True,
        unroll=1,
    )
    returns = advantages + values
    return advantages, returns


@partial(jax.jit, static_argnames=["num_minibatches"])
def get_minibatches_from_batch(
    batch: tuple[jax.Array, ...], rng: jax.Array, num_minibatches: int
):
    assert batch[0].shape[0] % num_minibatches == 0, (
        "batch size should be a multiple of num_minibatches, got"
        f" batch_size={batch[0].shape[0]}, {num_minibatches=}"
    )
    new_batch = jax.tree_util.tree_map(lambda x: x.reshape((-1, x.shape[-1])), batch)
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jax.random.permutation(rng, x, axis=0), new_batch
    )

    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [num_minibatches, -1, x.shape[-1]]),
        shuffled_batch,
    )
    return minibatches
