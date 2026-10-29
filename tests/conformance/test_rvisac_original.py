"""RVI-SAC against its authors' code (yhisaki/average-reward-drl b5b0d31), one update at a time.

Their RVI_SAC, built with their rvi_sac.yaml at small widths, made the four calls of its update on
fixed batches a quarter of whose rows terminate, from its initialisation (updates 1 and 2: the reset
cost's step clamped at 0) and after 400 warm-up updates (401 and 402: warm Adams, a reset cost above
0 and rising), in float32 and in float64 with the same draws (benchmarks/reference/
conformance_rvisac.py; the repository has no licence, so only its records are committed). From the
same state, on the same batch, Ajax's RVI-SAC makes the same update (rvisac_original.py): every leaf
of the actor, both critics and their targets, the reset critic and its target, the temperature, the
reset cost, the offsets fq and fq_reset and the five Adams must change as theirs do, within 10 times
the two sides' float32 rounding (submaps). Their evaluation action, the mean of tanh at mu +- sigma,
must match Ajax's on fixed states. Two planted faults (kappa 0.0055 for 0.005, eps 0.0011 for 0.001)
must fail. Pre-registered (ajax-scratch/records/rvi-sac/PLAN.md, section 8) before its first run.
"""

import numpy as np
import pytest

from tests.conformance import originals, rvisac_original

RECORDS = originals.committed("yhisaki_average_reward_drl")


@pytest.fixture(scope="module")
def ours() -> dict:
    return {p: rvisac_original.run(RECORDS, p) for p in ("f32", "f64")}


def test_each_recorded_update_matches_theirs(ours):
    for report in originals.reports(RECORDS, ours, "rvisac", ours["f32"]):
        assert not report.failures, str(report)


def test_the_records_cover_the_clamp_and_a_rising_cost():
    """At the start the termination rate is below eps, so the cost's step is clamped at 0; after the
    warm-up it is above 0 and rising; every batch holds terminations and other rows."""
    start, warm = (dict(np.load(RECORDS / f"rvisac_{n}.npz")) for n in (1, 401))
    assert start["f32.fq_reset"] < 1e-3 and start["f32.reset_cost.value"] == 0.0
    assert 0.0 < warm["in.reset_cost.value"] < warm["f32.reset_cost.value"]
    for record in (start, warm):
        assert 0.0 < record["batch.terminated"].mean() < 1.0


@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_the_evaluation_action_matches_theirs(precision):
    """The mean of tanh at mu +- sigma (their modules.py:115-116). Tolerances: float64 on both sides
    differs only by operation order (1e-12); float32 by a few roundings of values below 1 (1e-6)."""
    actions = rvisac_original.deterministic_actions(RECORDS, precision)
    assert set(actions) == {"rvisac_1", "rvisac_402"}
    for name, ours_action in actions.items():
        theirs = np.load(RECORDS / f"{name}.npz")[f"{precision}.act"]
        tolerance = 1e-12 if precision == "f64" else 1e-6
        np.testing.assert_allclose(ours_action, theirs, rtol=0, atol=tolerance)


@pytest.mark.parametrize("fault", ["kappa", "eps"])
def test_a_planted_fault_fails_the_check(ours, fault):
    faulty = rvisac_original.run(RECORDS, "f32", fault)
    assert all(r.failures for r in originals.reports(RECORDS, ours, "rvisac", faulty))
