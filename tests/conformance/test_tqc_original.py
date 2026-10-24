"""TQC against its original, bayesgroup/tqc_pytorch (vendored from Software Heritage's archive), one
update at a time.

The authors' Actor, Critic and Trainer train on two gymnasium tasks by main.py's loop and record their
1st and 25th updates with every draw (benchmarks/reference/conformance_tqc.py, run on torch 1.4, its
records committed: originals.committed). Their Trainer.train runs unchanged: its actor's backward pass
runs after the critics' optimizer step and, on that torch, reads the stepped critics' weights. Ajax's
TQC takes the same update from the same state with the same batch and noises (tqc_original.py) and
still differentiates the actor's loss at the critics before their step: every leaf of the 5 critics,
their targets, the temperature and their Adams changes as the authors' do within the float32-rounding
tolerance, and the actor's Adam moments differ. Ajax's TQC is to read the stepped critics too (owner,
2026-10-24) once its file is converted; the test then asks every leaf to conform. A planted fault
(gamma 0.991 instead of 0.99) must fail the check.
"""

import pytest

from tests.conformance import originals, tqc_original

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")


def actors(leaf: str) -> bool:
    return leaf.startswith(("actor.", "opt.actor."))


@pytest.fixture(scope="module")
def records():
    return originals.committed("bayesgroup_tqc_pytorch")


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: tqc_original.run(records, p) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_all_but_the_actor_matches_the_authors(records, ours, env_id):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not [f for f in report.failures if not actors(f.leaf)], str(report)


@pytest.mark.parametrize("env_id", ENVS)
def test_the_actor_differs_while_ajax_reads_the_critics_before_their_step(
    records, ours, env_id
):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert any(f.leaf.startswith("opt.actor.") for f in report.failures), str(
            report
        )


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = tqc_original.run(records, "f32", "gamma")
    reports = originals.reports(records, ours, ENVS[0], faulty)
    assert any(f for r in reports for f in r.failures if not actors(f.leaf))
