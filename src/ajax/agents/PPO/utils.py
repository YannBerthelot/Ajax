import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.deprecation import moved_names


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
    """GAE (Schulman et al., 2016) over a time-major rollout, in reverse:
    ``delta_t = r_t + gamma V(s_{t+1}) (1 - terminated_t) - V(s_t)`` and
    ``A_t = delta_t + gamma lambda (1 - done_t) A_{t+1}``: a truncation cuts
    the trace but still bootstraps. Returns ``(A, A + V)``."""

    def _get_advantages(
        gae: jax.Array,
        transition,
    ) -> tuple[jax.Array, jax.Array]:
        reward, value, next_value, terminated, truncated = transition

        next_state_is_non_terminal = 1.0 - terminated
        done = 1 - jnp.logical_or(terminated, truncated)

        delta = reward + gamma * next_value * next_state_is_non_terminal - value
        gae = delta + gamma * gae_lambda * done * gae

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


__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
        (
            "split_fragments",
            "get_minibatches_preserving_time",
            "get_minibatches_from_batch",
        ),
        "ajax.agents.minibatch",
    ),
)
