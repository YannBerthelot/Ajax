"""TD3 against its original, sfujim/TD3 (vendored at 34770cc), one update at a time.

sfujim's own TD3.py trains on two gymnax tasks (main.py's loop) in its pinned PyTorch env and records
four states each, with the update's batch and target noise and the state after it in float32 and
float64 (benchmarks/reference/conformance_sfujim_td3.py, originals.recorded). Ajax makes the same update
from the same state, batch and noise (td3_sfujim.py, in this process), and every leaf's change must
agree within 10 times the two sides' float32 rounding errors (originals.reports). MountainCarContinuous's
box is [-1, 1]; Pendulum's torque is [-2, 2], where Ajax's critics must read the env's units. A planted
fault (target noise 0.22 instead of 0.2) must fail the check.
"""

import pytest

from tests.conformance import originals, td3_sfujim

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("sfujim")
    return originals.recorded("sfujim_td3", "conformance_sfujim_td3.py", ENVS, scratch)


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: td3_sfujim.run(records, p) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_one_update_matches_sfujims(records, ours, env_id):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not report.failures, str(report)


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = td3_sfujim.run(records, "f32", "target_noise")
    assert any(f.failures for f in originals.reports(records, ours, ENVS[0], faulty))
