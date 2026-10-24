"""ASAC against its original, BobMak/asac (vendored at 4b68862), one update at a time.

ASAC's own darer/ASAC.py learns by its learn() loop on two gymnasium tasks and on PendulumEnds
(Pendulum-v1 driven by the mean of a two-dimensional torque, terminated every 20th step) and records
two updates each, with the actor's and the next actions' noise (benchmarks/reference/conformance_asac.py,
in its own env: stable-baselines3 2.4, gymnasium 1.0). From the same state, on the same batch, Ajax's
ASAC makes the same update (asac_original.py), and every leaf of the actor, both critics and their
targets, the three Adams, theta and the termination penalty must change as ASAC's do within the
float32-rounding tolerance. Ajax follows the original where it reads like a slip (the penalty's two
rows, ``log(1/nA)`` for a continuous policy), logged for upstream, so nothing is set aside. A planted
fault (tau 0.0055 instead of 0.005) must fail the check.
"""

import numpy as np
import pytest

from tests.conformance import asac_original, originals

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1", "PendulumEnds")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("asac")
    recorder, project = "conformance_asac.py", originals.ASAC_ENV
    return originals.recorded("bobmak_asac", recorder, ENVS, scratch, project)


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: asac_original.run(records, p) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_one_update_matches_asacs(records, ours, env_id):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not report.failures, str(report)


def test_a_recorded_batch_ends_episodes(records):
    """PendulumEnds' later batch holds terminations and other rows, so the check covers the done
    mask, the penalty charged and its rule, with two action dimensions (log(1/nA))."""
    done = np.load(records / "PendulumEnds_1000.npz")["batch.done"]
    assert done.any() and not done.all()


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = asac_original.run(records, "f32", "tau")
    for env_id in ENVS:
        assert all(r.failures for r in originals.reports(records, ours, env_id, faulty))
