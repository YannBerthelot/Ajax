"""Q(lambda) target computation for PQN."""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial


@partial(jax.jit, static_argnames=["gamma", "q_lambda"])
def compute_q_lambda_targets(
    rewards: jax.Array,
    next_q_max: jax.Array,
    terminateds: jax.Array,
    truncateds: jax.Array,
    gamma: float,
    q_lambda: float,
) -> jax.Array:
    """Backward Q(lambda) returns over a rollout.

    With the bootstrap value ``V(s) := max_a Q(s, a)``, for each step t:

        G_t = r_t + gamma * ( V(s_{t+1})                          if done
                              (1-lambda) V(s_{t+1}) + lambda G_{t+1}  else )

    ``V(s_{t+1})`` is zeroed on natural termination; a truncated step
    bootstraps on ``V(s_{t+1})`` but does not propagate the lambda trace
    past the cut. The reverse-scan carry is seeded with the final step's
    bootstrap so the last transition gets a clean 1-step target.

    All inputs are shaped ``(n_steps, n_envs, 1)``; returns the same.
    """

    def body(next_return, transition):
        reward, q_next, terminated, truncated = transition
        non_terminal = 1.0 - terminated
        not_done = 1.0 - jnp.logical_or(
            terminated.astype(bool), truncated.astype(bool)
        ).astype(reward.dtype)
        v_next = non_terminal * q_next
        g = reward + gamma * (
            (1.0 - q_lambda) * v_next
            + q_lambda * (not_done * next_return + (1.0 - not_done) * v_next)
        )
        return g, g

    _, returns = jax.lax.scan(
        body,
        next_q_max[-1],
        (rewards, next_q_max, terminateds, truncateds),
        reverse=True,
    )
    return returns
