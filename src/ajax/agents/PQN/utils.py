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
    """Backward Q(lambda) returns over a rollout, as purejaxql's PQN
    computes them (pqn_gymnax.py:216-254 at 47af6d7).

    With the bootstrap ``V_t := max_a Q(s_{t+1}, a)``, for each step t:

        G_t = r_t                                         at an end
        G_t = r_t + gamma ((1 - lambda) V_t + lambda G_{t+1})   else

    and the last step's ``G = r + gamma V``. An end is a termination or a
    time limit: gymnax's done flag, which purejaxql reads, does not tell
    them apart. purejaxql seeds its backward pass with the last step's
    bootstrap, ``V(s_T)``, as the next value of step ``T - 2``, whose own is
    ``V(s_{T-1})``; that target, and through lambda the ones before it in
    its episode, are not Q(lambda)'s (purejaxql issue #12). Ajax does the
    same.

    All inputs are shaped ``(n_steps, n_envs, 1)``; returns the same.
    """
    done = jnp.logical_or(terminateds.astype(bool), truncateds.astype(bool))
    done = done.astype(rewards.dtype)
    v = (1.0 - done) * next_q_max
    last = rewards[-1] + gamma * v[-1]
    if rewards.shape[0] == 1:
        return last[None]
    v = v.at[-2].set(v[-1])  # purejaxql's off-by-one

    def body(next_return, transition):
        reward, v_next, d = transition
        g = reward + gamma * ((1.0 - q_lambda) * v_next + q_lambda * next_return)
        g = (1.0 - d) * g + d * reward
        return g, g

    _, returns = jax.lax.scan(
        body, last, (rewards[:-1], v[:-1], done[:-1]), reverse=True
    )
    return jnp.concatenate([returns, last[None]])
