import distrax
import jax
import jax.numpy as jnp

from ajax.agents.SAC.utils import (  # correct_log_probs,; sample_actions_and_log_prob,
    SquashedNormal,
)

# def test_correct_log_probs():
#     log_prob = jnp.array([1.0, 2.0])
#     raw_action = jnp.array([[0.5, -0.5], [1.0, -1.0]])

#     corrected = correct_log_probs(log_prob, raw_action)

#     assert corrected.shape == log_prob.shape, "Shape mismatch in corrected log probs."
#     assert jnp.all(
#         jnp.isfinite(corrected)
#     ), "Corrected log probs contain invalid values."


def test_sample_actions_and_log_prob():
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros(2)
    std = jnp.ones(2)
    pi = distrax.Normal(loc=mean, scale=std)

    squashed_action, corrected_log_prob = pi.sample_and_log_prob(seed=key)

    assert squashed_action.shape == mean.shape, "Shape mismatch in squashed actions."
    assert (
        corrected_log_prob.shape == squashed_action.shape
    ), "Shape mismatch in corrected log probs."
    assert jnp.all(
        jnp.isfinite(corrected_log_prob)
    ), "Corrected log probs contain invalid values."


def test_squashed_normal_sample():
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros(2)
    std = jnp.ones(2)
    pi = SquashedNormal(loc=mean, scale=std)

    squashed_action = pi.sample(seed=key)

    assert squashed_action.shape == mean.shape, "Shape mismatch in squashed actions."
    assert jnp.all(
        jnp.abs(squashed_action) <= 1.0
    ), "Squashed actions exceed bounds [-1, 1]."


def test_squashed_normal_log_prob():
    mean = jnp.zeros(2)
    std = jnp.ones(2)
    pi = SquashedNormal(loc=mean, scale=std)

    value = jnp.array([0.5, -0.5])
    log_prob = pi.log_prob(value)

    assert log_prob.shape == value.shape, "Shape mismatch in log probs."
    assert jnp.isfinite(log_prob.sum(-1)), "Log probs contain invalid values."


def test_squashed_normal_sample_and_log_prob():
    key = jax.random.PRNGKey(0)
    mean = jnp.zeros(2)
    std = jnp.ones(2)
    pi = SquashedNormal(loc=mean, scale=std)

    squashed_action, corrected_log_prob = pi.sample_and_log_prob(seed=key)

    assert squashed_action.shape == mean.shape, "Shape mismatch in squashed actions."
    assert (
        corrected_log_prob.shape == squashed_action.shape
    ), "Shape mismatch in corrected log probs."
    assert jnp.all(
        jnp.abs(squashed_action) <= 1.0
    ), "Squashed actions exceed bounds [-1, 1]."
    assert jnp.isfinite(
        corrected_log_prob.sum(-1)
    ), "Corrected log probs contain invalid values."
