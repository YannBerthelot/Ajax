"""Smoke + key-set tests for ``ajax.extensions.instrumentation``.

End-to-end checks that each of the four EVarEst measurement /
instrumentation extensions:

* :class:`ConditioningMetrics`
* :class:`BiasVoreDecomposition`
* :class:`DiagnosticSnapshots`  — host-side I/O via ``io_callback``
* :class:`BiasVorePenalty`      — additive critic-loss penalty

wires into the Extension surface on PPO (on-policy, through
``agent_state.last_rollout``). BiasVorePenalty's coefficient also has
unit checks. Numerical exactness is intentionally NOT covered here --
the math is ported verbatim from EVAREST (see the per-extension
docstring); these tests only pin the *integration contract*.
"""

from __future__ import annotations

import os

import pytest

from ajax.agents.PPO.PPO import PPO
from ajax.extensions.instrumentation import (
    COND_METRIC_KEYS,
    EVAREST_DECOMP_KEYS,
    BiasVoreDecomposition,
    BiasVorePenalty,
    ConditioningMetrics,
    DiagnosticSnapshots,
    evarest_coeff,
)
from ajax.logging.wandb_logging import LoggingConfig

# Evaluate (and return the evaluations) without a logging backend.
EVALUATE = LoggingConfig(config={}, log_frequency=64, use_wandb=False)

# --------------------------------------------------------------------------
# BiasVorePenalty: standalone unit checks (no agent in the loop)
# --------------------------------------------------------------------------


def test_bias_vore_penalty_alpha_half_is_zero():
    """alpha=0.5 ⇒ coeff = 0 ⇒ the penalty short-circuits to ``0.0``."""
    ext = BiasVorePenalty(alpha=0.5)
    assert ext._coeff() == 0.0


def test_bias_vore_penalty_coeff_matches_paper_formula():
    """coeff = (1 - 2 alpha) / alpha for alpha != 0.5."""
    for alpha in (0.1, 0.3, 0.7, 0.9):
        ext = BiasVorePenalty(alpha=alpha)
        assert ext._coeff() == pytest.approx(evarest_coeff(alpha))


def test_bias_vore_penalty_coeff_override_wins():
    """An explicit ``coeff_override`` takes precedence over alpha."""
    ext = BiasVorePenalty(alpha=0.1, coeff_override=0.0)
    assert ext._coeff() == 0.0
    ext = BiasVorePenalty(alpha=0.5, coeff_override=3.7)
    assert ext._coeff() == pytest.approx(3.7)


# --------------------------------------------------------------------------
# ConditioningMetrics on PPO: eval_metrics fold emits the four Cond/* keys.
# --------------------------------------------------------------------------


def _train_ppo_with_extension(ext, n_steps=32):
    agent = PPO(
        env_id="CartPole-v1",
        n_envs=2,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=1,
        expose_recent_rollout=True,
        extensions=[ext],
    )
    return agent.train(seed=42, n_timesteps=128, logging_config=EVALUATE)


def test_conditioning_metrics_on_ppo_completes():
    """ConditioningMetrics folds into PPO's eval_metrics phase end-to-end:
    its keys are among the evaluations ``train`` returns."""
    ext = ConditioningMetrics(on_policy_batch=16)
    out = _train_ppo_with_extension(ext)
    # ``train`` returns ``(state, metrics)``.
    assert isinstance(out, tuple) and len(out) == 2
    _, metrics = out
    for key in COND_METRIC_KEYS:
        assert key in metrics, f"missing eval-metric key {key!r}; got {list(metrics)}"


def test_bias_vore_decomposition_on_ppo_completes():
    """BiasVoreDecomposition folds into PPO's eval_metrics phase."""
    ext = BiasVoreDecomposition(on_policy_batch=16, gamma=0.99)
    out = _train_ppo_with_extension(ext)
    _, metrics = out
    for key in EVAREST_DECOMP_KEYS:
        assert key in metrics, f"missing eval-metric key {key!r}; got {list(metrics)}"


# --------------------------------------------------------------------------
# DiagnosticSnapshots: smoke test (uses a temp dir, post_update fires).
# --------------------------------------------------------------------------


def test_diagnostic_snapshots_writes_files(tmp_path):
    """DiagnosticSnapshots writes a chunk file every ``every_n_iters``."""
    ext = DiagnosticSnapshots(
        directory=str(tmp_path),
        every_n_iters=1,
        n_sub=8,
        sample_batch=4,
    )
    out = _train_ppo_with_extension(ext)
    _, _ = out
    # post_update is gated on (counter % every_n_iters == 0), so every
    # iteration writes a file. n_timesteps=128 / (n_envs=2 * n_steps=32)
    # plus 1 from the num_updates+1 formula ⇒ >= 1 snapshot expected.
    files = [f for f in os.listdir(tmp_path) if f.startswith("snapshot_")]
    assert (
        len(files) >= 1
    ), f"DiagnosticSnapshots wrote no files to {tmp_path}; got: {files}"


# --------------------------------------------------------------------------
# BiasVorePenalty on PPO: the additive critic-loss term lands in the loss
# fold without breaking training (numerical equivalence at alpha=0.5).
# --------------------------------------------------------------------------


def test_bias_vore_penalty_on_ppo_completes():
    """BiasVorePenalty (alpha=0.5 ⇒ no-op) keeps PPO training stable."""
    out = _train_ppo_with_extension(BiasVorePenalty(alpha=0.5))
    _, metrics = out
    # No new keys; just ensure training survived.
    assert metrics is not None


def test_bias_vore_penalty_alpha_nonhalf_runs():
    """alpha=0.3 ⇒ non-zero penalty; PPO still trains end-to-end."""
    out = _train_ppo_with_extension(BiasVorePenalty(alpha=0.3, loss_scale=0.5))
    _, metrics = out
    assert metrics is not None


# --------------------------------------------------------------------------
# Compose two instrumentation extensions: both contribute eval-metric keys.
# --------------------------------------------------------------------------


def test_conditioning_and_decomposition_compose_on_ppo():
    """Both Cond/* and EVarEst/* keys appear when the two are stacked."""
    agent = PPO(
        env_id="CartPole-v1",
        n_envs=2,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        expose_recent_rollout=True,
        extensions=[
            ConditioningMetrics(on_policy_batch=16),
            BiasVoreDecomposition(on_policy_batch=16),
        ],
    )
    out = agent.train(seed=42, n_timesteps=128, logging_config=EVALUATE)
    _, metrics = out
    for key in COND_METRIC_KEYS + EVAREST_DECOMP_KEYS:
        assert key in metrics, f"missing {key!r}; got {list(metrics)}"
