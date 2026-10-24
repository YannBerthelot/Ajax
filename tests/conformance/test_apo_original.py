"""APO against its original, xtma/apo (vendored at bfcd997, on the authors' rlpyt fork e28bb20), one
update at a time.

rlpyt's MujocoFfAgent collects 10 x 200 steps of two gymnax tasks and APPO (mujoco_appo.py's config,
the authors' run grid's discount 1.0) optimises on them; its 1st and 6th updates are recorded with
their minibatch permutations (benchmarks/reference/conformance_apo.py, originals.recorded). From the
same state, on the same rollout, Ajax's APO makes the same update (apo_original.py), and every leaf of
both networks, the log-std, Adam's moments and step, the reward rate and the value bias must change as
APPO's do within the float32-rounding tolerance, Pendulum's time limits included: both end the
episode there and leave the step out of the loss. A planted fault (gae_lambda 0.96 instead of 0.95)
must fail the check.
"""

import numpy as np
import pytest

from tests.conformance import apo_original, originals

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("apo")
    return originals.recorded("xtma_apo", "conformance_apo.py", ENVS, scratch)


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: apo_original.run(records, p) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_one_update_matches_appos(records, ours, env_id):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not report.failures, str(report)


def test_the_records_hold_time_limits_and_none(records):
    """Pendulum's batches hold a time limit per env, MountainCar's none: the check covers both."""
    for env_id, limits in (("Pendulum-v1", True), ("MountainCarContinuous-v0", False)):
        for path in records.glob(f"{env_id}_*.npz"):
            timeout = np.load(path)["samples.env.env_info.timeout"]
            assert bool(timeout.any()) == limits, path.stem


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = apo_original.run(records, "f32", "gae_lambda")
    assert all(r.failures for r in originals.reports(records, ours, ENVS[0], faulty))
