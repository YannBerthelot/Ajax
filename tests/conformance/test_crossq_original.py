"""CrossQ against its original, adityab/CrossQ (vendored at a318a26), one update at a time.

The authors' SAC, built as their train.py builds CrossQ, trains on two gymnasium tasks by
Stable-Baselines3's loop and records three updates with every draw
(benchmarks/reference/conformance_crossq.py, originals.recorded):
- the first, a critic step only;
- the 24th, on which the policy delay steps the actor and the temperature too;
- the 24th again, every Batch Renorm layer past its 100,000-pass warm-up.

Ajax's CrossQ takes the same update from the same state, with the same batch and normals
(crossq_original.py). Every leaf must change as the authors' do within the float32-rounding
tolerance, on MountainCarContinuous's [-1, 1] actions and Pendulum's [-2, 2] torque:
- the critics' and the actor's parameters, running statistics and pass counts;
- the temperature;
- their Adams.

Ajax follows the authors where it reads like a slip: their temperature loss has alpha where SAC
has log alpha, and their Batch Renorm's gradient is NaN at a zero batch variance (UPSTREAM_BUGS).
A planted fault (gamma 0.991 instead of 0.99) must fail the check.
"""

import numpy as np
import pytest

from tests.conformance import crossq_original, originals

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("crossq")
    return originals.recorded(
        "adityab_crossq", "conformance_crossq.py", ENVS, scratch, originals.CROSSQ_ENV
    )


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: crossq_original.run(records, p) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_one_update_matches_the_authors(records, ours, env_id):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not report.failures, str(report)


def test_the_records_cover_the_policy_delay_and_the_renorm(records):
    """The first update leaves the actor as it was, the 24th steps it; the renormed record's layers
    are past their warm-up."""
    for env_id in ENVS:
        first, delayed, renorm = (
            dict(np.load(records / f"{env_id}_{n}.npz"))
            for n in ("1", "24", "24renorm")
        )
        kernel = "actor.params.Dense_0.kernel"
        assert (first[f"f32.{kernel}"] == first[f"in.{kernel}"]).all()
        assert (delayed[f"f32.{kernel}"] != delayed[f"in.{kernel}"]).any()
        assert (
            renorm["in.qf.batch_stats.VmapCritic_0.BatchRenorm_1.steps"] >= 100_000
        ).all()


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = crossq_original.run(records, "f32", "gamma")
    assert any(f.failures for f in originals.reports(records, ours, ENVS[0], faulty))
