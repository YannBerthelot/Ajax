"""TQC against its original, bayesgroup/tqc_pytorch (vendored from Software Heritage's archive), one
update at a time.

The authors' Actor, Critic and Trainer train on two gymnax tasks by main.py's loop and record their 1st
and 25th updates with every draw (benchmarks/reference/conformance_tqc.py, originals.recorded). Ajax's
TQC takes the same update from the same state with the same batch and noises (tqc_original.py), and
every leaf of the 5 critics, their targets, the actor, the temperature and their Adams must change as
the authors' do within the float32-rounding tolerance, on MountainCarContinuous's [-1, 1] actions and
Pendulum's [-2, 2] torque. The recorder makes one change to their update, recorded: their actor's
backward pass runs after the critics' step there, before it here, where Ajax reads the critics too
(the upstream bug the recorder's docstring describes). A planted fault (gamma 0.991 instead of 0.99)
must fail the check.
"""

import pytest

from tests.conformance import originals, tqc_original

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("tqc")
    return originals.recorded(
        "bayesgroup_tqc_pytorch", "conformance_tqc.py", ENVS, scratch
    )


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: tqc_original.run(records, p) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_one_update_matches_the_authors(records, ours, env_id):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not report.failures, str(report)


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = tqc_original.run(records, "f32", "gamma")
    assert any(f.failures for f in originals.reports(records, ours, ENVS[0], faulty))
