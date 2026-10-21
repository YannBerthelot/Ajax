"""AVG against its original, gauthamvasan/avg (vendored at 6b6822c), one update at a time.

AVG's own avg.py learns on two gymnax tasks by its main() loop under its gymnasium wrappers and records
two updates each, with the action's and the next action's noise (benchmarks/reference/conformance_avg.py,
originals.recorded). From the same state, on the same transition, Ajax's AVG folds its TD-error
statistics and makes the same update (avg_original.py), and every leaf of the actor, the critic, both
Adams and the TD-error scaler's statistics must change as AVG's do within the float32-rounding
tolerance. Ajax follows AVG where it reads like a slip (its log_std head is a log-variance); one
registered difference (refcmp_specs.AVG_DEVIATIONS) is asserted, not hidden: AVG hands its tanh action
to the env unscaled, where Ajax maps it onto the box. A planted fault (gamma 0.991 instead of 0.99) must
fail the check.
"""

import numpy as np
import pytest
from refcmp_specs import AVG_DEVIATIONS

from ajax.agents.AVG.AVG import AVG
from ajax.environments.utils import critic_action_units
from tests.conformance import avg_original, originals

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("avg")
    return originals.recorded("gauthamvasan_avg", "conformance_avg.py", ENVS, scratch)


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: avg_original.run(records, p) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_one_update_matches_avgs(records, ours, env_id):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not report.failures, str(report)


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
    assert any(r.failures for r in reports)


def test_every_avg_deviation_is_pinned_here():
    for d in AVG_DEVIATIONS:
        path, name = d.check.split("::")
        assert path.endswith(__file__.split("/tests/")[-1]) and name in globals(), (
            d.name
        )
