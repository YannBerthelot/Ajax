"""Discounted returns and advantages over an on-policy rollout, shared by
the on-policy agents (PPO, APO): one source for how an episode
end cuts a return and where a value bootstraps it."""

import jax
import jax.numpy as jnp


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    next_values: jax.Array,
    terminateds: jax.Array,
    truncateds: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """GAE (Schulman et al., 2016) over a time-major rollout, in reverse:
    delta_t = r_t + gamma V(s_{t+1}) (1 - terminated_t) - V(s_t) and
    A_t = delta_t + gamma lambda (1 - done_t) A_{t+1}: a truncation cuts
    the trace but still bootstraps. Returns (A, A + V)."""

    def _get_advantages(
        gae: jax.Array,
        transition,
    ) -> tuple[jax.Array, jax.Array]:
        reward, value, next_value, terminated, truncated = transition

        next_state_is_non_terminal = 1.0 - terminated
        episode_continues = 1 - jnp.logical_or(terminated, truncated)

        delta = reward + gamma * next_value * next_state_is_non_terminal - value
        gae = delta + gamma * gae_lambda * episode_continues * gae

        return gae, gae

    # Unrolled: the same operations in fewer loop iterations. On a GPU each
    # iteration of this small body costs a few kernel launches: a rollout of
    # 2048 steps took about 22 ms rolled and 2 ms unrolled (GTX 1080 Ti, 10
    # seeds); on a CPU either takes under 0.5 ms, with the same advantages to
    # the bit.
    _, advantages = jax.lax.scan(
        f=_get_advantages,
        init=jnp.zeros_like(values[-1]),
        xs=(rewards, values, next_values, terminateds, truncateds),
        reverse=True,
        unroll=16,
    )
    returns = advantages + values
    return advantages, returns
