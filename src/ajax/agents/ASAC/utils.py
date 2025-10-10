import jax
import jax.numpy as jnp


def compute_episode_termination_penalty(
    episode_termination_penalty: jax.Array,
    rewards: jax.Array,
    terminated: jax.Array,
    p_0: float,
    tau: float,
) -> jax.Array:
    p_bar = p_0 * jnp.mean(rewards * (1 - terminated))
    episode_termination_penalty = episode_termination_penalty * (1 - tau) + tau * p_bar
    return episode_termination_penalty


def get_episode_termination_penalized_rewards(
    episode_termination_penalty: jax.Array, rewards: jax.Array, terminated: jax.Array
) -> jax.Array:
    return rewards - episode_termination_penalty * terminated
