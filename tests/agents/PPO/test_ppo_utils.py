import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.PPO.core import policy_entropy
from ajax.agents.PPO.utils import (
    _compute_gae,
    get_minibatches_from_batch,
    get_minibatches_preserving_time,
)
from ajax.agents.SAC.utils import SquashedNormal


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
        next_values=jnp.concat(
            (values[1:], jnp.expand_dims(last_value, axis=0)), axis=0
        ),
        terminateds=dones,
        truncateds=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
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

    advantages, returns = _compute_gae(
        rewards=rewards,
        values=values,
        next_values=values,
        terminateds=dones,
        truncateds=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
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
        next_values=jnp.concat(
            (values[1:], jnp.expand_dims(last_value, axis=0)), axis=0
        ),
        terminateds=dones,
        truncateds=dones,
        gamma=gamma,
        gae_lambda=gae_lambda,
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


@pytest.mark.parametrize("unroll_length", [4, None])
def test_time_minibatches_hold_whole_fragments_in_time_order(unroll_length):
    """(T=8, 4 envs) into 2 minibatches: each holds whole fragments of
    ``unroll_length`` steps (the rollout by default), time-major, and every
    fragment lands once."""
    T, n_envs, length = 8, 4, unroll_length or 8
    step, env = jnp.meshgrid(jnp.arange(T), jnp.arange(n_envs), indexing="ij")
    mbs = get_minibatches_preserving_time(
        {"step": step, "env": env}, jax.random.PRNGKey(0), 2, unroll_length
    )
    per_mb = T // length * n_envs // 2
    assert mbs["step"].shape == mbs["env"].shape == (2, length, per_mb)
    np.testing.assert_array_equal(np.diff(mbs["step"], axis=1), 1)
    assert (mbs["env"] == mbs["env"][:, :1]).all()
    starts = zip(np.ravel(mbs["step"][:, 0]), np.ravel(mbs["env"][:, 0]))
    expected = [(c * length, e) for c in range(T // length) for e in range(n_envs)]
    assert sorted(starts) == expected


def test_policy_entropy_is_the_joint_actions():
    """Summed over the action dimensions (CleanRL, baselines); a squashed
    policy's is the executed action's, sampled with a key it requires."""
    loc, scale = jnp.zeros((3, 2)), jnp.full((3, 2), 2.0)
    normal = distrax.Normal(loc, scale)
    joint = normal.entropy().sum(-1)
    np.testing.assert_allclose(policy_entropy(normal), joint, rtol=1e-6)
    uniform = distrax.Categorical(logits=jnp.zeros((3, 4)))
    np.testing.assert_allclose(policy_entropy(uniform), jnp.full(3, jnp.log(4.0)))
    squashed = SquashedNormal(loc, scale)
    with pytest.raises(ValueError):
        policy_entropy(squashed)
    executed = policy_entropy(squashed, jax.random.PRNGKey(0))
    assert executed.shape == (3,) and bool(jnp.all(executed < joint))  # tanh shrinks
