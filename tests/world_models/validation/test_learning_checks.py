"""The judged statistic and the registry of
``benchmarks/world_models/learning_checks.py``. Training is not run here
(the checks take an hour or more on CPU).
"""

import learning_checks
import numpy as np
import paper_protocol
import pytest

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


def test_the_smoke_runs_use_the_paper_protocols_tiny_models():
    """One tiny model per agent: paper_protocol's, each check keeping its
    task's own settings."""

    def model(kwargs):
        task = ("action_repeat", "episode_length", "seed_steps")
        return {key: value for key, value in kwargs.items() if key not in task}

    smoke = learning_checks.SMOKE
    for name in ("dreamerv3-cartpole", "dreamerv3-pendulum"):
        assert smoke[name].kwargs == model(paper_protocol.DREAMERV3_TINY), name
    tdmpc2 = model(paper_protocol.TDMPC2_TINY)
    assert smoke["tdmpc2-pendulum"].kwargs == {**tdmpc2, "seed_steps": 200}
    assert smoke["tdmpc2-cartpole-balance"].kwargs == paper_protocol.TDMPC2_TINY
