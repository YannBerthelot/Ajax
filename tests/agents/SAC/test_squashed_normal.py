"""SquashedNormal, the tanh-squashed Gaussian policy of the SAC family and of
PPO and APO's continuous actor. Its standard deviation of 1.5 puts many
samples in the saturated region (|action| > 0.95), where recomputing a
log-probability by inverting tanh is unstable: PPO once stored post-tanh
actions and recomputed through ``arctanh`` (the May 2026 m4 audit); it now
stores the pre-tanh ``raw_action`` and calls ``log_prob_from_raw``."""

import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.SAC.utils import SquashedNormal

PI = SquashedNormal(jnp.zeros(7), jnp.full(7, 1.5))
KEY = jax.random.PRNGKey(0)


def test_samples_are_bounded_with_finite_log_probs():
    actions, log_probs = PI.sample_and_log_prob(seed=KEY)
    assert actions.shape == log_probs.shape == (7,)
    assert jnp.all(jnp.abs(actions) <= 1.0)
    assert jnp.all(jnp.isfinite(log_probs))
    np.testing.assert_array_equal(PI.sample(seed=KEY), actions)


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


def test_log_prob_from_raw_is_the_one_sampling_gave():
    """The live method behind PPO and APO's ratio: from the pre-tanh draw
    (``tanh`` of which is the action), the summed log-probability sampling
    gave, without inverting tanh."""
    raw = PI.distribution.sample(seed=KEY)
    actions, log_probs = PI.sample_and_log_prob(seed=KEY)
    np.testing.assert_allclose(jnp.tanh(raw), actions, rtol=1e-5)
    from_raw = PI.log_prob_from_raw(raw)
    assert from_raw.shape == (1,)
    np.testing.assert_allclose(from_raw, log_probs.sum(-1, keepdims=True), rtol=1e-5)
