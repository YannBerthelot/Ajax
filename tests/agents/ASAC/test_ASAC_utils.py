import jax.numpy as jnp

from ajax.agents.ASAC.utils import (
    compute_episode_termination_penalty,
    get_episode_termination_penalized_rewards,
)


def test_compute_episode_termination_penalty_basic():
    episode_termination_penalty = jnp.array(0.5)
    rewards = jnp.array([1.0, 2.0, 3.0])
    terminated = jnp.array([0.0, 0.0, 0.0])
    p_0 = 0.1
    tau = 0.5

    minibatch_size = rewards.shape[0]
    p_bar = p_0 * (1 / minibatch_size) * jnp.sum(rewards)
    expected = episode_termination_penalty * (1 - tau) + tau * p_bar

    result = compute_episode_termination_penalty(
        episode_termination_penalty, rewards, terminated, p_0, tau
    )

    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"


def test_compute_episode_termination_penalty_tau_zero():
    # tau = 0 should return the same penalty
    penalty = jnp.array(0.5)
    rewards = jnp.array([1.0, 2.0, 3.0])
    terminated = jnp.array([0.0, 0.0, 0.0])
    result = compute_episode_termination_penalty(
        penalty, rewards, terminated, p_0=0.1, tau=0.0
    )
    assert jnp.allclose(result, penalty)


def test_compute_episode_termination_penalty_tau_one():
    # tau = 1 should fully update to p_bar
    penalty = jnp.array(0.5)
    rewards = jnp.array([1.0, 2.0, 3.0])
    terminated = jnp.array([0.0, 0.0, 0.0])
    p_0 = 0.2
    expected_pbar = p_0 * (1 / rewards.shape[0]) * jnp.sum(rewards)

    result = compute_episode_termination_penalty(
        penalty, rewards, terminated, p_0, tau=1.0
    )
    assert jnp.allclose(result, expected_pbar)


def test_compute_episode_termination_penalty_batched():
    # Works with batched minibatch rewards (e.g., shape [batch, n])
    penalty = jnp.array(0.5)
    rewards = jnp.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    terminated = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    p_0 = 0.1
    tau = 0.3

    result = compute_episode_termination_penalty(penalty, rewards, terminated, p_0, tau)
    assert result.shape == (), "Should return scalar even with vector rewards"


def test_get_episode_termination_penalized_rewards_no_termination():
    rewards = jnp.array([1.0, 2.0, 3.0])
    terminated = jnp.array([0.0, 0.0, 0.0])
    penalty = jnp.array(0.5)

    result = get_episode_termination_penalized_rewards(penalty, rewards, terminated)
    assert jnp.allclose(result, rewards), "No termination → rewards unchanged"


def test_get_episode_termination_penalized_rewards_with_termination():
    rewards = jnp.array([1.0, 2.0, 3.0])
    terminated = jnp.array([1.0, 0.0, 1.0])
    penalty = jnp.array(0.5)

    expected = rewards - penalty * terminated
    result = get_episode_termination_penalized_rewards(penalty, rewards, terminated)

    assert jnp.allclose(result, expected)


def test_get_episode_termination_penalized_rewards_shape_consistency():
    rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    terminated = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    penalty = jnp.array(0.2)

    result = get_episode_termination_penalized_rewards(penalty, rewards, terminated)
    assert result.shape == rewards.shape, "Output shape must match rewards shape"
    assert result.dtype == rewards.dtype, "Output dtype must match rewards dtype"
