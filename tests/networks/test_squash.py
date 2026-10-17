"""Squash, tanh of any policy distribution: the SAC family's, PPO and APO's
continuous actor. Its standard deviation of 1.5 puts many samples in the
saturated region (|action| > 0.95), where recomputing a log-probability by
inverting tanh is unstable: PPO once stored post-tanh actions and
recomputed through ``arctanh`` (the May 2026 m4 audit); it now stores the
pre-tanh ``raw_action`` and calls ``log_prob_from_raw``."""

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.TD3.networks import DeterministicActor
from ajax.networks.distributions import (
    Deterministic,
    Squash,
    entropy,
    latent,
    log_prob_from_raw,
    sample_with_raw,
)
from ajax.networks.networks import Actor, Squashed

NORMAL = distrax.Normal(jnp.zeros(7), jnp.full(7, 1.5))
PI = Squash(NORMAL)
KEY = jax.random.PRNGKey(0)


def test_samples_are_bounded_with_finite_log_probs():
    actions, log_probs = PI.sample_and_log_prob(seed=KEY)
    assert actions.shape == log_probs.shape == (7,)
    assert jnp.all(jnp.abs(actions) <= 1.0)
    assert jnp.all(jnp.isfinite(log_probs))
    np.testing.assert_array_equal(PI.sample(seed=KEY), actions)
    np.testing.assert_array_equal(PI.mean(), jnp.tanh(NORMAL.mean()))


def test_the_log_prob_of_a_sample_is_the_one_sampling_gave():
    """``log_prob(action)`` goes through ``arctanh``: finite and within 1e-3
    of the sampling's even at saturation."""
    actions, log_probs = PI.sample_and_log_prob(seed=KEY)
    recomputed = PI.log_prob(actions)
    assert recomputed.shape == actions.shape
    assert jnp.all(jnp.isfinite(recomputed.sum(-1)))
    np.testing.assert_allclose(
        recomputed.sum(-1), log_probs.sum(-1), rtol=1e-3, atol=1e-3
    )


def test_the_raw_sample_gives_the_sampling_log_prob():
    """The path behind PPO and APO's ratio: from the pre-tanh draw, the
    log-probability sampling gave, without inverting tanh."""
    actions, log_probs, raw = PI.sample_with_raw(KEY)
    np.testing.assert_array_equal(jnp.tanh(raw), actions)
    np.testing.assert_allclose(log_probs, PI.sample_and_log_prob(seed=KEY)[1], 1e-5)
    np.testing.assert_array_equal(PI.log_prob_from_raw(raw), log_probs)


def test_a_multivariate_latent_is_squashed_per_event():
    """Any distrax distribution: a diagonal multivariate Normal's squash is
    the Normal's, summed over the event."""
    mvn = distrax.MultivariateNormalDiag(jnp.zeros(7), jnp.full(7, 1.5))
    actions, log_prob, raw = Squash(mvn).sample_with_raw(KEY)
    assert log_prob.shape == ()
    np.testing.assert_allclose(log_prob, PI.log_prob_from_raw(raw).sum(), 1e-5)
    np.testing.assert_allclose(Squash(mvn).log_prob(actions), log_prob, 1e-3)


def test_the_sampled_entropy_is_the_squashed_actions():
    """No closed form: it needs a seed; averaged over seeds it matches a
    Monte Carlo estimate of -E[log pi(a)], below the latent's (tanh
    shrinks)."""
    with pytest.raises(ValueError):
        PI.entropy()
    keys = jax.random.split(KEY, 20_000)
    sampled = jax.vmap(PI.entropy)(keys).mean(0)
    monte_carlo = -jax.vmap(lambda k: PI.sample_and_log_prob(seed=k)[1])(keys).mean(0)
    np.testing.assert_allclose(sampled, monte_carlo, atol=0.03)
    assert jnp.all(sampled < NORMAL.entropy())


def test_the_helpers_read_any_policy():
    """Unsquashed, the latent is the policy and its raw sample the action."""
    assert latent(PI) is NORMAL and latent(NORMAL) is NORMAL
    actions, log_probs, raw = sample_with_raw(NORMAL, KEY)
    assert raw is actions
    np.testing.assert_allclose(log_prob_from_raw(NORMAL, raw), log_probs, 1e-6)
    np.testing.assert_array_equal(entropy(NORMAL), NORMAL.entropy())
    np.testing.assert_array_equal(entropy(PI, KEY), PI.entropy(KEY))
    np.testing.assert_array_equal(
        sample_with_raw(PI, KEY)[2], PI.sample_with_raw(KEY)[2]
    )


@pytest.mark.parametrize(
    "actor",
    [Actor(("8", "relu"), 2, continuous=True), DeterministicActor(("8", "relu"), 2)],
)
def test_squashed_bounds_any_actor_with_its_parameters(actor):
    """One shared scope: the wrapped actor's parameters are the actor's
    own, and its policy is the actor's through tanh (a point mass's action
    through tanh)."""
    obs = jnp.ones((3, 4))
    params = actor.init(KEY, obs)
    wrapped = Squashed(actor).init(KEY, obs)
    assert jax.tree.structure(wrapped) == jax.tree.structure(params)
    jax.tree.map(np.testing.assert_array_equal, wrapped, params)
    pi, inner = Squashed(actor).apply(params, obs), actor.apply(params, obs)
    np.testing.assert_array_equal(pi.mean(), jnp.tanh(inner.mean()))
    assert isinstance(pi, Deterministic if isinstance(inner, Deterministic) else Squash)
