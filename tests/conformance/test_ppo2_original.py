"""PPO against its reference code, OpenAI Baselines' ppo2 (vendored at ea25b9e), one update at a
time, for each kind of network ppo2 builds.

ppo2.learn trains with its own defaults (benchmarks/reference/conformance_ppo2.py) and its updates
are recorded: separate policy and value MLPs on Pendulum (mujoco(), value_network='copy'), and one
nature_cnn both heads read on Breakout's frames (atari()). From the same state, on the same batch,
Ajax's PPO makes the same update (ppo2_original.py), and every leaf of both networks, Adam's moments
and step must change as ppo2's do within ten times the two sides' float32 rounding. A difference of
network topology (a shared encoder kept apart, or the reverse) moves the leaves the other side's
losses train. Planted faults must fail each case: gae_lambda 0.96 instead of 0.95, and the case's
topology flipped (the MLP's encoder shared, the CNN's kept apart).

The records are committed (tests/conformance/records/openai_baselines): ppo2 runs on TensorFlow 1.15,
which CI cannot install (conformance_ppo2.py says how to remake them).
"""

import numpy as np
import pytest
from conformance_start import SAMPLED

from tests.conformance import originals, ppo2_original, submaps

CASES = ("mlp", "cnn")


@pytest.fixture(scope="module")
def records():
    return originals.committed("openai_baselines")


@pytest.fixture(scope="module")
def ours(records) -> dict:
    return {
        (case, p): ppo2_original.run(records, case, p)
        for case in CASES
        for p in ("f32", "f64")
    }


def reports(records, ours: dict, case: str, update: dict) -> list:
    """Per record of ``case``, ``update`` against ppo2's float32 update, on the change from the
    record's state, within ten times the two sides' float32 rounding."""
    out = []
    for path in sorted(records.glob(f"{case}_*.npz")):
        arrays = dict(np.load(path))
        old, r32, r64 = (
            ppo2_original.reference_state(arrays, prefix)
            for prefix in ("in.", "f32.", "f64.")
        )
        a32, a64 = (ours[case, p][path.stem] for p in ("f32", "f64"))
        nulls = submaps.rounding_null((r32, r64), (a32, a64), old=old)
        # A ppo2 leaf Ajax has no counterpart of (its network built otherwise) fails as NaN.
        new = {
            k: update[path.stem].get(k, np.full_like(v, np.nan)) for k, v in r32.items()
        }
        out.append(submaps.compare(new, r32, nulls, old=old, context=path.stem))
    assert out, f"no record of {case}"
    return out


@pytest.mark.parametrize("case", CASES)
def test_one_update_matches_ppo2s(records, ours, case):
    for report in reports(records, ours, case, ours[case, "f32"]):
        assert not report.failures, str(report)


def test_the_records_hold_episode_ends_and_large_leaves(records):
    """Both cases' batches end episodes (the bootstrap's cut is checked), and the CNN's dense layer
    is kept at sampled positions."""
    for path in records.glob("*.npz"):
        arrays = np.load(path)
        assert arrays["env.dones"].any(), path.stem
    cnn = np.load(next(records.glob("cnn_*.npz")))
    assert cnn["at.ppo2_model/pi/fc1/w"].shape == (SAMPLED,)


@pytest.mark.parametrize(
    "case, fault",
    [("mlp", "gae_lambda"), ("mlp", "shared"), ("cnn", "gae_lambda"), ("cnn", "apart")],
)
def test_a_planted_fault_fails_the_check(records, ours, case, fault):
    faulty = ppo2_original.run(records, case, "f32", fault)
    assert all(r.failures for r in reports(records, ours, case, faulty))
