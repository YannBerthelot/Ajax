import jax.numpy as jnp
import pytest

from ajax.agents.APO.utils import _compute_gae


@pytest.mark.parametrize(
    (
        "rewards, values, dones, gae_lambda, last_value, average_reward,"
        " expected_advantages, expected_returns"
    ),
    [
        (
            jnp.array([1.0, 1.0, 1.0]),  # rewards
            jnp.array([0.5, 0.5, 0.5]),  # values
            jnp.array([0.0, 0.0, 1.0]),  # dones (not used in APO variant)
            0.95,  # gae_lambda
            jnp.array(0.0),  # last_value
            0.0,  # average_reward
            # Manual calculation:
            # step 2: δ2 = 1 - 0 + 0 - 0.5 = 0.5, GAE2 = 0.5
            # step 1: δ1 = 1 - 0 + 0.5 - 0.5 = 1.0, GAE1 = 1.0 + 0.95*0.5 = 1.475
            # step 0: δ0 = 1 - 0 + 0.5 - 0.5 = 1.0, GAE0 = 1.0 + 0.95*1.475 = 2.40125
            jnp.array([2.40125, 1.475, 0.5]),  # expected_advantages
            jnp.array([2.90125, 1.975, 1.0]),  # expected_returns = adv + values
        ),
        (
            jnp.array([0.0, 0.0, 1.0]),  # rewards
            jnp.array([0.0, 0.0, 0.0]),  # values
            jnp.array([0.0, 0.0, 1.0]),  # dones (ignored)
            0.95,  # gae_lambda
            jnp.array(0.0),  # last_value
            0.0,  # average_reward
            # step 2: δ2 = 1, GAE2 = 1
            # step 1: δ1 = 0, GAE1 = 0 + 0.95*1 = 0.95
            # step 0: δ0 = 0, GAE0 = 0.95*0.95 = 0.9025
            jnp.array([0.9025, 0.95, 1.0]),  # expected_advantages
            jnp.array([0.9025, 0.95, 1.0]),  # returns = adv + values
        ),
        (
            jnp.array([1.0, 2.0, 3.0]),  # rewards
            jnp.array([1.0, 1.0, 1.0]),  # values
            jnp.array([0.0, 0.0, 0.0]),
            0.5,  # gae_lambda
            jnp.array(1.0),  # last_value
            2.0,  # average_reward
            # δ2 = 3 - 2 + 1 - 1 = 1, GAE2 = 1
            # δ1 = 2 - 2 + 1 - 1 = 0, GAE1 = 0 + 0.5*1 = 0.5
            # δ0 = 1 - 2 + 1 - 1 = -1, GAE0 = -1 + 0.5*0.5 = -0.75
            jnp.array([-0.75, 0.5, 1.0]),
            jnp.array([0.25, 1.5, 2.0]),  # adv + values
        ),
    ],
)
def test_compute_gae_apo(
    rewards,
    values,
    dones,
    gae_lambda,
    last_value,
    average_reward,
    expected_advantages,
    expected_returns,
):
    advantages, returns = _compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gae_lambda=gae_lambda,
        last_value=last_value,
        average_reward=average_reward,
    )

    assert jnp.allclose(
        advantages, expected_advantages, atol=1e-4
    ), f"Advantages mismatch: {advantages} != {expected_advantages}"
    assert jnp.allclose(
        returns, expected_returns, atol=1e-4
    ), f"Returns mismatch: {returns} != {expected_returns}"


def test_compute_gae_with_zeros_apo():
    rewards = jnp.zeros(5)
    values = jnp.zeros(5)
    dones = jnp.zeros(5)
    gae_lambda = 0.95
    last_value = jnp.array(0.0)
    average_reward = 0.0

    advantages, returns = _compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gae_lambda=gae_lambda,
        last_value=last_value,
        average_reward=average_reward,
    )

    assert jnp.allclose(advantages, jnp.zeros(5)), "Advantages should be all zeros."
    assert jnp.allclose(returns, jnp.zeros(5)), "Returns should be all zeros."
