"""REDQ against its original, watchernyu/REDQ (vendored at 7b5d1bf), one training step at a time.

REDQ's own agent trains on two gymnax tasks by train_redq_sac.py's loop and records its 1st and 25th
training steps, each 20 critic updates then the actor's and the temperature's, with every draw
(benchmarks/reference/conformance_redq.py, originals.recorded). Ajax's REDQ takes the same step from the
same state with the same batches, subsets and noises (redq_original.py), and every leaf of the 10
critics, their targets, the actor, the temperature and their Adams must change as REDQ's do within the
float32-rounding tolerance, on MountainCarContinuous's [-1, 1] actions and Pendulum's [-2, 2] torque.
The two registered differences (refcmp_specs.REDQ_DEVIATIONS) are accounted for and asserted: REDQ's
first step reads the temperature its __init__ overwrote, and its tanh log-det adds 1e-6. A planted
fault (gamma 0.991 instead of 0.99) must fail the check.
"""

import numpy as np
import pytest
from refcmp_specs import REDQ_DEVIATIONS

from tests.conformance import originals, redq_original

ENVS = ("MountainCarContinuous-v0", "Pendulum-v1")


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("redq")
    return originals.recorded("watchernyu_redq", "conformance_redq.py", ENVS, scratch)


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {p: redq_original.run(records, p) for p in ("f32", "f64")}


@pytest.mark.parametrize("env_id", ENVS)
def test_one_step_matches_redqs(records, ours, env_id):
    for report in originals.reports(records, ours, env_id, ours["f32"]):
        assert not report.failures, str(report)


def test_redqs_first_step_reads_its_overwritten_temperature(records, ours):
    """REDQ's first step reads 0.2 while its log_alpha is 0, its later ones exp(log_alpha): started
    from exp(log_alpha) instead, Ajax's first steps differ and its later ones do not."""
    plain = redq_original.run(records, "f32", effective_alpha=False)
    for env_id in ENVS:
        paths = sorted(records.glob(f"{env_id}_*.npz"))
        for path, report in zip(
            paths, originals.reports(records, ours, env_id, plain), strict=True
        ):
            a = np.load(path)
            used, own = float(a["in.alpha_used"]), float(np.exp(a["in.log_alpha"])[0])
            first = path.stem.endswith("_1")
            assert (used, own) == (0.2, 1.0) if first else np.isclose(used, own)
            assert bool(report.failures) == first, report.context


def test_a_planted_fault_fails_the_check(records, ours):
    faulty = redq_original.run(records, "f32", "gamma")
    assert any(f.failures for f in originals.reports(records, ours, ENVS[0], faulty))


def test_every_redq_deviation_is_pinned_here():
    for d in REDQ_DEVIATIONS:
        path, name = d.check.split("::")
        assert path.endswith(__file__.split("/tests/")[-1]) and name in globals(), (
            d.name
        )
