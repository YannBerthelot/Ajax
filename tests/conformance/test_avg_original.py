"""AVG against its original, gauthamvasan/avg (vendored at 6b6822c), one update at a time.

AVG's own avg.py learns on two gymnax tasks by its main() loop under its gymnasium wrappers and records
two updates each, with the action's and the next action's noise (benchmarks/reference/conformance_avg.py,
originals.recorded). From the same state, on the same transition, Ajax's AVG folds its TD-error
statistics and makes the same update (avg_original.py), and every leaf of the actor, the critic, both
Adams and the TD-error scaler's statistics must change as AVG's do within the float32-rounding
tolerance. Two registered differences (refcmp_specs.AVG_DEVIATIONS) are asserted, not hidden: AVG's
log_std head is a log-variance, and AVG hands its tanh action to the env unscaled. A planted fault
(gamma 0.991 instead of 0.99) must fail the check.
"""

import numpy as np
import pytest
from refcmp_specs import AVG_DEVIATIONS

from ajax.agents.AVG.AVG import AVG
from ajax.environments.utils import critic_action_units
from tests.conformance import avg_original, originals

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")
HEAD = ("actor.log_std.weight", "actor.log_std.bias")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("avg")
    return originals.recorded("gauthamvasan_avg", "conformance_avg.py", ENVS, scratch)


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: avg_original.run(records, p) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_one_update_matches_avgs(records, ours, env_id):
    """Every leaf but the log-std head's (test_avgs_log_std_head_is_a_log_variance)."""
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not [c for c in report.failures if c.leaf not in HEAD], str(report)


def test_avgs_log_std_head_is_a_log_variance(records, ours):
    """Mapped forward-equivalently (Ajax's head = AVG's / 2), only the head's leaves differ, and Adam
    steps the head twice as far in AVG's units: its change is twice AVG's (up to Adam's eps, which
    weighs differently at the two scales, so the smallest changes are left out)."""
    for env_id in ENVS:
        for report in originals.reports(records, ours, env_id, ours["f32"]):
            assert sorted(c.leaf for c in report.failures) == sorted(HEAD), str(report)
            start = np.load(records / f"{report.context}.npz")
            for k in HEAD:
                theirs = start[f"f32.{k}"] - start[f"in.{k}"]
                mine = ours["f32"][report.context][k] - start[f"in.{k}"]
                keep = np.abs(theirs) > 1e-3 * np.abs(theirs).max()
                assert abs(np.median(mine[keep] / theirs[keep]) - 2.0) < 1e-3, k


def test_avg_hands_its_tanh_action_to_the_env_unscaled(records):
    """AVG's executed action is its tanh action, in [-1, 1] even on Pendulum's [-2, 2] torque, which
    ClipAction leaves as it is; Ajax maps [-1, 1] onto the box."""
    for path in records.glob("Pendulum-v1_*.npz"):
        a = np.load(path)
        np.testing.assert_allclose(a["action"], np.tanh(a["action_pre"]), rtol=1e-6)
    pendulum = AVG("Pendulum-v1").env_args
    assert critic_action_units(pendulum.env, pendulum.env_params) == ((0.0,), (2.0,))


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = avg_original.run(records, "f32", "gamma")
    reports = originals.reports(records, ours, ENVS[0], faulty)
    assert any(c.leaf not in HEAD for r in reports for c in r.failures)


def test_every_avg_deviation_is_pinned_here():
    for d in AVG_DEVIATIONS:
        path, name = d.check.split("::")
        assert path.endswith(__file__.split("/tests/")[-1]) and name in globals(), (
            d.name
        )
