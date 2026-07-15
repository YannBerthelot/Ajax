"""PPO RL-tier tests: algorithm correctness on synthetic data.

These verify the mathematical primitives PPO depends on (loss math,
GAE conventions, log_prob recomputation, advantage normalization).
They run on hand-constructed inputs with known closed-form expected
outputs -- no full training loop, no env, no randomness in the
assertion targets.

The contrast with the wire/code tier (``test_PPO_code.py``) is:
  * Code tier: "is the configured value actually reaching the loss?"
  * RL tier:   "is the loss math itself correct?"

The contrast with the smoke / probing tier (``test_PPO.py``,
``test_probing.py``) is:
  * Smoke / probing: "does the full agent learn a task?"
  * RL tier:         "is each individual numerical primitive correct?"

A failing RL test points at a specific math bug; a failing smoke test
just says "something is wrong somewhere".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ajax.agents.SAC.utils import SquashedNormal


def test_squashed_normal_log_prob_round_trip_is_stable():
    """``pi.sample_and_log_prob(seed)`` and ``pi.sample(seed)`` then
    ``pi.log_prob(action)`` MUST agree to numerical precision. If they
    don't, the second path is going through the unstable
    ``arctanh(post_tanh_action)`` inverse and any PPO ratio computation
    built on it is corrupted.

    This is the diagnostic for the May 2026 m4 audit finding: PPO was
    storing post-tanh ``action`` and recomputing log_prob via
    ``pi.log_prob(action)``, which distrax explicitly warns is
    unstable near saturation. The current fix stores the pre-tanh
    ``raw_action`` and recomputes via ``base.log_prob(raw) -
    forward_log_det_jacobian(raw)`` -- never inverting tanh.

    This unit test catches the math primitive in isolation; the
    boundary-action probing env catches the end-to-end agent
    behaviour. Belt-and-suspenders.
    """
    # std=1.5 ⇒ many samples land in the saturated region |action| > 0.95
    pi = SquashedNormal(jnp.zeros(7), jnp.ones(7) * 1.5)
    key = jax.random.PRNGKey(0)
    actions, log_probs_safe = pi.sample_and_log_prob(seed=key)
    # The "safe" log_prob comes from sample_and_log_prob -- distrax
    # computes it on the pre-tanh sample as part of sampling.
    # The "unsafe" path recomputes log_prob from the post-tanh action,
    # going through arctanh internally.
    log_probs_recompute = pi.log_prob(actions)

    # Sum over action dims to compare scalars per sample.
    safe_sum = log_probs_safe.sum(-1) if log_probs_safe.ndim > 0 else log_probs_safe
    recompute_sum = (
        log_probs_recompute.sum(-1)
        if log_probs_recompute.ndim > 0
        else log_probs_recompute
    )

    # First catch: NaN/Inf in the recompute path.
    assert jnp.isfinite(recompute_sum).all(), (
        f"pi.log_prob(post_tanh_action) produced non-finite values: "
        f"{recompute_sum}. Some sampled action is at or extremely "
        "close to ±1 saturation, where arctanh is undefined. "
        "Fix the PPO/APO log_prob recompute path to use pre-tanh "
        "raw_action via base.log_prob(raw) - forward_log_det_jacobian(raw)."
    )

    # Second catch: silent numerical drift even when both paths return
    # finite numbers. With std=1.5 and at moderate saturation, the
    # arctanh path can return finite-but-biased values that the safe
    # path doesn't have.
    assert jnp.allclose(safe_sum, recompute_sum, rtol=1e-3, atol=1e-3), (
        f"sample_and_log_prob and sample-then-log_prob disagree at "
        f"rtol=1e-3: safe={safe_sum}, recompute={recompute_sum}. "
        "The recompute path is going through distrax.Tanh.inverse_and_log_det "
        "(arctanh) which is unstable near saturation. PPO's ratio "
        "computation on stored vs recomputed log_prob will diverge."
    )


def test_squashed_normal_log_prob_via_raw_action_matches_safe_path():
    """The brax-style fix: compute log_prob from the pre-tanh sample
    via ``base.log_prob(raw) - forward_log_det_jacobian(raw)``. This
    MUST agree with the ``sample_and_log_prob`` output. Verifies the
    formula PPO uses post-fix is correct.
    """
    pi = SquashedNormal(jnp.zeros(7), jnp.ones(7) * 1.5)
    key = jax.random.PRNGKey(0)

    # Sample the underlying Normal directly (the pre-tanh sample) and
    # apply tanh manually. This mirrors what PPO's collector does
    # post-m4-fix.
    base = pi.distribution
    raw_action = base.sample(seed=key)
    action = jnp.tanh(raw_action)

    # The two log_prob computations that should agree:
    # 1. sample_and_log_prob on a deterministic-from-the-same-key
    #    sample (we re-sample with the same key to get the same draw)
    _, log_probs_safe = pi.sample_and_log_prob(seed=key)
    # 2. Manual base + forward Jacobian on the raw sample.
    log_probs_via_raw = base.log_prob(
        raw_action
    ) - pi.bijector.forward_log_det_jacobian(raw_action)

    assert jnp.allclose(log_probs_safe, log_probs_via_raw, rtol=1e-5), (
        f"Manual base + forward-Jacobian log_prob diverges from "
        f"sample_and_log_prob: via_raw={log_probs_via_raw}, "
        f"safe={log_probs_safe}. Either distrax's sample path uses a "
        "different formula than (base.log_prob - forward_jacobian), "
        "or the bijector's forward_log_det_jacobian is buggy."
    )
    # Sanity check the actions match (deterministic sample from same key)
    actions_safe, _ = pi.sample_and_log_prob(seed=key)
    assert jnp.allclose(action, actions_safe, rtol=1e-5)
