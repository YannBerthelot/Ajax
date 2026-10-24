"""ReBRAC against its original, tinkoff-ai/ReBRAC (vendored at 694385b), one update at a time.

ReBRAC learns offline by main()'s loop from rows its qlearning_dataset builds out of uniform
random-action episodes on two tasks with the action box [-1, 1] (MountainCarEnds terminating every
20th step), and records updates 0 and 1000 (an actor step) and 1 and 1001 (the critics alone)
(benchmarks/reference/conformance_rebrac.py, originals.recorded). Ajax is the docs' recipe, TD3 with
TD3+BC's term and TargetBC (rebrac_original.py). The critics' steps match. On the actor steps Ajax
still differs from the original in the actor alone, until ReBRAC becomes an agent of its own (after
TD3's file is converted): TD3's actor reads its first critic and scales by that critic's mean |Q|
where ReBRAC takes the minimum over both, and ReBRAC moves the actor's target towards the actor's
parameters from before the step. A planted fault (TargetBC's beta 1.1) must fail the check.
"""

import pytest

from tests.conformance import originals, rebrac_original

ENVS = ("MountainCarContinuous-v0", "MountainCarEnds")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("rebrac")
    return originals.recorded(
        "tinkoff_rebrac",
        "conformance_rebrac.py",
        ENVS,
        scratch,
        project=originals.JAX_OFFLINE_ENV,
    )


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: rebrac_original.run(records, p) for p in ("f32", "f64")}


def by_update(records, ours, env_id, update, updates) -> dict:
    reports = originals.reports(records, ours, env_id, update)
    return {
        r.context.rsplit("_", 1)[1]: r
        for r in reports
        if r.context.rsplit("_", 1)[1] in updates
    }


@pytest.mark.parametrize("env_id", ENVS)
def test_the_critics_steps_match_rebracs(records, ours, env_id):
    for report in by_update(records, ours, env_id, ours["f32"], ("1", "1001")).values():
        assert not report.failures, str(report)


@pytest.mark.parametrize("env_id", ENVS)
def test_the_actor_steps_differ_in_the_actor_alone(records, ours, env_id):
    for report in by_update(records, ours, env_id, ours["f32"], ("0", "1000")).values():
        failed = {f.leaf.split(".")[0] for f in report.failures}
        assert failed == {"actor"}, str(report)
        assert any(f.leaf.startswith("actor.target.") for f in report.failures), str(
            report
        )


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = rebrac_original.run(records, "f32", beta=1.1)
    critic_steps = by_update(records, ours, ENVS[1], faulty, ("1", "1001"))
    assert critic_steps and all(r.failures for r in critic_steps.values())
