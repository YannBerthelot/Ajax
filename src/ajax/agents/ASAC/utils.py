import jax
import jax.numpy as jnp


def compute_episode_termination_penalty(
    episode_termination_penalty: jax.Array,
    rewards: jax.Array,
    terminated: jax.Array,
    p_0: float,
    tau: float,
) -> jax.Array:
    """Move the penalty by ``tau`` towards ``p_0`` times the largest reward
    ``gather(rewards, 0, terminated)`` reads, as the authors' code
    (BobMak/asac@4b68862, ``darer/ASAC.py:187``, with 20 for ``p_0``): a
    row reads the reward of the flattened batch's row 0 when it did not
    terminate, of row 1 when it did. Their comment means the mean
    non-terminal reward."""
    rows = terminated.reshape(-1).astype(jnp.int32)
    p_bar = p_0 * jnp.max(rewards.reshape(-1)[rows])
    episode_termination_penalty = episode_termination_penalty * (1 - tau) + tau * p_bar
    return episode_termination_penalty


def get_episode_termination_penalized_rewards(
    episode_termination_penalty: jax.Array, rewards: jax.Array, terminated: jax.Array
) -> jax.Array:
    return rewards - episode_termination_penalty * terminated
