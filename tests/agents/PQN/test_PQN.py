import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DQN.train_DQN import make_huber_td_loss
from ajax.agents.PQN.PQN import PQN
from ajax.agents.PQN.utils import compute_q_lambda_targets


def test_PQN_rejects_indivisible_minibatches():
    """The rollout's n_steps * n_envs rows must split into num_minibatches."""
    with pytest.raises(ValueError, match="must be divisible"):
        PQN(env_id="CartPole-v1", n_envs=1, n_steps=7, num_minibatches=2)
    PQN(env_id="CartPole-v1", n_envs=2, n_steps=7, num_minibatches=2)


def test_pqn_huber_loss_smoke():
    """The Huber td_loss_fn hook composes and runs end-to-end."""
    agent = PQN(
        env_id="CartPole-v1",
        n_envs=2,
        architecture=("16", "relu"),
        n_steps=8,
        n_epochs=2,
        num_minibatches=2,
        td_loss_fn=make_huber_td_loss(1.0),
    )
    agent.train(seed=42, n_timesteps=256)


def _serial_q_lambda(rewards, next_q_max, terminateds, truncateds, gamma, lam):
    """compute_q_lambda_targets' serial reverse scan from before its
    associative one."""

    def step(next_return, x):
        reward, q_next, terminated, truncated = x
        done = jnp.logical_or(terminated.astype(bool), truncated.astype(bool))
        not_done = 1.0 - done.astype(reward.dtype)
        v_next = (1.0 - terminated) * q_next
        g = reward + gamma * (
            (1.0 - lam) * v_next
            + lam * (not_done * next_return + (1.0 - not_done) * v_next)
        )
        return g, g

    xs = (rewards, next_q_max, terminateds, truncateds)
    return jax.lax.scan(step, next_q_max[-1], xs, reverse=True)[1]


@pytest.mark.parametrize("T, n_envs", [(1, 2), (37, 3), (128, 8)])
def test_q_lambda_targets_match_the_serial_recursion(T, n_envs):
    """The associative scan reorders the serial recursion's roundings only:
    within 8 float32 eps of the sum of the absolute terms each target adds
    up (1.4 eps at worst over 480 random rollouts). Truncations and
    terminations, some on one step; the last step seeds the recursion."""
    k = jax.random.split(jax.random.PRNGKey(T), 4)
    r, q = (10 * jax.random.normal(k[i], (T, n_envs, 1)) for i in range(2))
    term, trunc = (
        (jax.random.uniform(k[i], (T, n_envs, 1)) < 0.03).astype(jnp.float32)
        for i in (2, 3)
    )
    got = compute_q_lambda_targets(r, q, term, trunc, 0.99, 0.65)
    expected = _serial_q_lambda(r, q, term, trunc, 0.99, 0.65)
    zeros = jnp.zeros_like(q)
    scale = _serial_q_lambda(abs(r) + 0.99 * abs(q), zeros, term, trunc, 0.99, 0.65)
    eps = np.finfo(np.float32).eps
    assert np.all(np.abs(got - expected) <= 8 * eps * scale)
