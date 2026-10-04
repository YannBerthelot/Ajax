"""The judged statistic and the registry of ``benchmarks/learning_checks.py``.

The script is not a package module; it is loaded from its path. Training is
not run here (the checks take an hour or more on CPU).
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "benchmarks",
    "learning_checks.py",
)
_SPEC = importlib.util.spec_from_file_location("learning_checks", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
learning_checks = importlib.util.module_from_spec(_SPEC)
# Registered before execution: dataclasses resolves its classes' module there.
sys.modules["learning_checks"] = learning_checks
_SPEC.loader.exec_module(learning_checks)

CURVE = np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 400.0]])
TIMESTEPS = np.array([100, 200, 300, 400])


def test_without_a_window_the_last_log_is_judged():
    np.testing.assert_array_equal(
        learning_checks.statistic(CURVE, TIMESTEPS, None), [40.0, 400.0]
    )


def test_a_window_averages_its_logs_bounds_included():
    np.testing.assert_array_equal(
        learning_checks.statistic(CURVE, TIMESTEPS, (200, 300)), [25.0, 2.5]
    )
    np.testing.assert_array_equal(
        learning_checks.statistic(CURVE, TIMESTEPS, (300, 400)), [35.0, 201.5]
    )


def test_a_window_without_logs_raises():
    with pytest.raises(ValueError, match="no log in the window"):
        learning_checks.statistic(CURVE, TIMESTEPS, (210, 290))


@pytest.mark.parametrize("registry", ["CHECKS", "SMOKE"])
def test_every_window_lies_in_its_budget_on_logged_ticks(registry):
    for name, check in getattr(learning_checks, registry).items():
        assert check.n_timesteps % check.log_frequency == 0, name
        if check.window is None:
            continue
        lo, hi = check.window
        assert 0 <= lo <= hi <= check.n_timesteps, name
        logs = np.arange(1, check.n_timesteps // check.log_frequency + 1)
        inside = (logs * check.log_frequency >= lo) & (logs * check.log_frequency <= hi)
        assert inside.any(), name


def test_dreamerv3_cartpole_follows_the_reference_comparison_protocol():
    """The protocol of the reference comparison's round 3 (PERFORMANCE_REPORT.md):
    24 000 rows, evaluated every 400 rows with 10 episodes, judged on the mean
    over [12 000, 24 000] (31 logs) against half the reference's mean (312.7)."""
    check = learning_checks.CHECKS["dreamerv3-cartpole"]
    assert (check.n_timesteps, check.log_frequency, check.num_episode_test) == (
        24_000,
        400,
        10,
    )
    assert check.window == (12_000, 24_000)
    assert check.bar == 156.0
    assert check.kwargs == {"model_size": "1m"}
