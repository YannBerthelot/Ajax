"""Q(lambda) target computation for PQN."""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from ajax.utils import reverse_linear_scan


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
    past the cut. The recursion is seeded with the final step's bootstrap
    so the last transition gets a clean 1-step target.

    All inputs are shaped ``(n_steps, n_envs, 1)``; returns the same.
    """
    non_terminal = 1.0 - terminateds
    not_done = 1.0 - jnp.logical_or(
        terminateds.astype(bool), truncateds.astype(bool)
    ).astype(rewards.dtype)
    v_next = non_terminal * next_q_max
    # G_t = a_t + b_t G_{t+1}, b_t = gamma lambda (1 - done_t): a done cuts
    # the trace and bootstraps on V(s_{t+1}) in full.
    a = rewards + gamma * (
        (1.0 - q_lambda) * v_next + q_lambda * ((1.0 - not_done) * v_next)
    )
    return reverse_linear_scan(a, gamma * q_lambda * not_done, next_q_max[-1])
