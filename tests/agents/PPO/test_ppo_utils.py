import jax
import jax.numpy as jnp
import pytest

from ajax.agents.PPO.utils import _compute_gae, get_minibatches_from_batch


@pytest.mark.parametrize(
    (
        "rewards, values, dones, gamma, gae_lambda, last_value, expected_advantages,"
        " expected_returns"
    ),
    [
        (
            jnp.array([1.0, 1.0, 1.0]),  # rewards
            jnp.array([0.5, 0.5, 0.5]),  # values
            jnp.array([0.0, 0.0, 1.0]),  # dones
            0.99,  # gamma
            0.95,  # gae_lambda
            jnp.array(0.0),  # last_value
            jnp.array(
                [
                    (1.0 + 0.99 * 0.5 - 0.5)
                    + 0.99 * 0.95 * ((1 + 0.99 * 0.5 - 0.5) + 0.99 * 0.95 * (1 - 0.5)),
                    (1 + 0.99 * 0.5 - 0.5) + 0.99 * 0.95 * (1 - 0.5),
                    0.5,
                ]
            ),  # expected_advantages = (reward + gamma * next_val - value) +  gamma * gae * gae
            jnp.array(
                [
                    (1.0 + 0.99 * 0.5 - 0.5)
                    + 0.99 * 0.95 * ((1 + 0.99 * 0.5 - 0.5) + 0.99 * 0.95 * (1 - 0.5))
                    + 0.5,
                    (1 + 0.99 * 0.5 - 0.5) + 0.99 * 0.95 * (1 - 0.5) + 0.5,
                    1.0,
                ]
            ),  # expected_returns = expected_advantages + values
        ),
        (
            jnp.array([0.0, 0.0, 1.0]),  # rewards
            jnp.array([0.0, 0.0, 0.0]),  # values
            jnp.array([0.0, 0.0, 1.0]),  # dones
            0.99,  # gamma
            0.95,  # gae_lambda
            jnp.array(0.0),  # last_value
            jnp.array([(0.99 * 0.95) ** 2, 0.99 * 0.95, 1.0]),  # expected_advantages
            jnp.array([(0.99 * 0.95) ** 2, 0.99 * 0.95, 1.0]),  # expected_returns
        ),
    ],
)
def test_compute_gae(
    rewards,
    values,
    dones,
    gamma,
    gae_lambda,
    last_value,
    expected_advantages,
    expected_returns,
):
    advantages, returns = _compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
        last_value=last_value,
    )

    assert jnp.allclose(
        advantages, expected_advantages, atol=1e-4
    ), f"Advantages mismatch: {advantages} != {expected_advantages}"
    assert jnp.allclose(
        returns, expected_returns, atol=1e-4
    ), f"Returns mismatch: {returns} != {expected_returns}"


def test_compute_gae_with_zeros():
    rewards = jnp.zeros(5)
    values = jnp.zeros(5)
    dones = jnp.zeros(5)
    gamma = 0.99
    gae_lambda = 0.95
    last_value = jnp.array(0.0)

    advantages, returns = _compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
        last_value=last_value,
    )

    assert jnp.allclose(advantages, jnp.zeros(5)), "Advantages should be all zeros."
    assert jnp.allclose(returns, jnp.zeros(5)), "Returns should be all zeros."


def test_compute_gae_with_terminal_state():
    rewards = jnp.array([1.0, 1.0, 1.0])
    values = jnp.array([0.5, 0.5, 0.5])
    dones = jnp.array([0.0, 1.0, 0.0])  # Terminal state in the middle
    gamma = 0.99
    gae_lambda = 0.95
    last_value = jnp.array(0.0)

    advantages, returns = _compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
        last_value=last_value,
    )

    assert advantages.shape == rewards.shape, "Advantages shape mismatch."
    assert returns.shape == rewards.shape, "Returns shape mismatch."
    assert jnp.isfinite(advantages).all(), "Advantages contain invalid values."
    assert jnp.isfinite(returns).all(), "Returns contain invalid values."


@pytest.mark.parametrize(
    "batch_size, n_envs, num_minibatches, feature_dim",
    [
        (16, 2, 4, 8),  # Batch size 16, 2 envs, 4 minibatches, feature dimension 8
        (32, 2, 8, 4),  # Batch size 32, 2 envs, 8 minibatches, feature dimension 4
    ],
)
def test_get_minibatches_from_batch(batch_size, n_envs, num_minibatches, feature_dim):
    rng = jax.random.PRNGKey(0)
    batch = (
        jnp.arange(batch_size * n_envs * feature_dim).reshape(
            batch_size, n_envs, feature_dim
        ),
        jnp.arange(batch_size * n_envs).reshape(batch_size, n_envs, 1),
    )  # Example batch with two arrays

    minibatches = get_minibatches_from_batch(
        batch=batch, rng=rng, num_minibatches=num_minibatches
    )

    # Validate the number of minibatches
    assert len(minibatches) == len(
        batch
    ), "Minibatches should have the same structure as the input batch."
    for minibatch in minibatches:
        assert (
            minibatch.shape[0] == num_minibatches
        ), "Number of minibatches is incorrect."
        assert (
            minibatch.shape[1] == batch_size * n_envs // num_minibatches
        ), "Minibatch size is incorrect."

    # Validate that all elements are present in the minibatches
    for original, shuffled in zip(batch, minibatches):
        flattened_original = original.flatten()
        flattened_shuffled = shuffled.flatten()
        assert jnp.all(
            jnp.sort(flattened_original) == jnp.sort(flattened_shuffled)
        ), "Minibatches do not contain all elements from the original batch."
