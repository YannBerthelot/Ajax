import jax.numpy as jnp

from ajax.agents.AVG.utils import (
    NormalizationInfo,
    compute_td_error_scaling,
)


def test_compute_td_error_scaling_initial():
    reward_value = jnp.array([[1.0]])
    reward = NormalizationInfo(
        value=reward_value,
        count=jnp.zeros_like(reward_value),
        mean=jnp.zeros_like(reward_value),
        mean_2=jnp.zeros_like(reward_value),
    )
    gamma_value = jnp.array([[0.99]])
    gamma = NormalizationInfo(
        value=gamma_value,
        count=jnp.zeros_like(gamma_value),
        mean=jnp.zeros_like(gamma_value),
        mean_2=jnp.zeros_like(gamma_value),
    )
    G_return_value = jnp.array([[10.0]])
    G_return = NormalizationInfo(
        value=G_return_value,
        count=jnp.zeros_like(G_return_value),
        mean=jnp.zeros_like(G_return_value),
        mean_2=jnp.zeros_like(G_return_value),
    )

    td_error_scaling, updated_reward, updated_gamma, updated_G_return = (
        compute_td_error_scaling(reward, gamma, G_return)
    )

    assert jnp.allclose(td_error_scaling, jnp.ones_like(td_error_scaling))
    assert jnp.all(updated_reward.count == 1)
    assert jnp.all(updated_gamma.count == 1)
    assert jnp.all(updated_G_return.count == 1)
    assert jnp.allclose(updated_G_return.mean, jnp.array([[100.0]]))  # Square of 10.0


def test_compute_td_error_scaling_update():
    reward = NormalizationInfo(
        value=jnp.array([2.0]),
        count=jnp.array([1]),
        mean=jnp.array([1.0]),
        mean_2=jnp.array([0.0]),
    )
    gamma = NormalizationInfo(
        value=jnp.array([0.98]),
        count=jnp.array([1]),
        mean=jnp.array([0.99]),
        mean_2=jnp.array([0.0]),
    )
    G_return = NormalizationInfo(
        value=jnp.array([20.0]),
        count=jnp.array([1]),
        mean=jnp.array([10.0]),
        mean_2=jnp.array([0.0]),
    )

    td_error_scaling, updated_reward, updated_gamma, updated_G_return = (
        compute_td_error_scaling(reward, gamma, G_return)
    )

    assert jnp.all(updated_reward.count == 2)
    assert jnp.all(updated_gamma.count == 2)
    assert jnp.all(updated_G_return.count == 2)
    assert jnp.allclose(updated_reward.mean, jnp.array([[1.5]]))
    assert jnp.allclose(updated_gamma.mean, jnp.array([[0.985]]))
    assert jnp.allclose(updated_G_return.mean, jnp.array([[205.0]]))  # Mean of squares


def test_compute_td_error_scaling_initial_batched():
    reward_value = jnp.array([[1.0], [2.0]])
    reward = NormalizationInfo(
        value=reward_value,
        count=jnp.zeros_like(reward_value),
        mean=jnp.zeros_like(reward_value),
        mean_2=jnp.zeros_like(reward_value),
    )
    gamma_value = jnp.array([[0.99], [0.95]])
    gamma = NormalizationInfo(
        value=gamma_value,
        count=jnp.zeros_like(gamma_value),
        mean=jnp.zeros_like(gamma_value),
        mean_2=jnp.zeros_like(gamma_value),
    )
    G_return_value = jnp.array([[10.0], [20.0]])
    G_return = NormalizationInfo(
        value=G_return_value,
        count=jnp.zeros_like(G_return_value),
        mean=jnp.zeros_like(G_return_value),
        mean_2=jnp.zeros_like(G_return_value),
    )

    td_error_scaling, updated_reward, updated_gamma, updated_G_return = (
        compute_td_error_scaling(reward, gamma, G_return)
    )

    assert not jnp.allclose(td_error_scaling, jnp.ones_like(td_error_scaling))
    assert jnp.all(updated_reward.count == 2)
    assert jnp.all(updated_gamma.count == 2)
    assert jnp.all(updated_G_return.count == 2)
    assert jnp.allclose(updated_G_return.mean, jnp.array([[250.0], [250.0]]))  # Squares


def test_compute_td_error_scaling_update_batched():
    reward = NormalizationInfo(
        value=jnp.array([[2.0], [4.0]]),
        count=jnp.array([[1], [1]]),
        mean=jnp.array([[1.0], [1.0]]),
        mean_2=jnp.array([[0.0], [0.0]]),
    )
    gamma = NormalizationInfo(
        value=jnp.array([[0.98], [0.96]]),
        count=jnp.array([[1], [1]]),
        mean=jnp.array([[0.99], [0.99]]),
        mean_2=jnp.array([[0.0], [0.0]]),
    )
    G_return = NormalizationInfo(
        value=jnp.array([[20.0], [30.0]]),
        count=jnp.array([[1], [1]]),
        mean=jnp.array([[10.0], [10.0]]),
        mean_2=jnp.array([[0.0], [0.0]]),
    )

    td_error_scaling, updated_reward, updated_gamma, updated_G_return = (
        compute_td_error_scaling(reward, gamma, G_return)
    )

    assert jnp.all(updated_reward.count == 3)
    assert jnp.all(updated_gamma.count == 3)
    assert jnp.all(updated_G_return.count == 3)
    assert jnp.allclose(updated_reward.mean, jnp.array([[7 / 3], [7 / 3]]))
    assert jnp.allclose(
        updated_gamma.mean,
        jnp.array([[(0.99 + 0.98 + 0.96) / 3], [(0.99 + 0.98 + 0.96) / 3]]),
    )
    assert jnp.allclose(
        updated_G_return.mean,
        jnp.array([[(10 + 20**2 + 30**2) / 3], [(10 + 20**2 + 30**2) / 3]]),
    )  # Squares
