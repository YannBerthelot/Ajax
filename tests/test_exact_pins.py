"""The exact before/after pins (tests/exact_pins.py): a leaf off by one bit
is reported, and a run recorded then compared on one machine is bitwise,
its program's text the same."""

from pathlib import Path

import numpy as np
import pytest

from ajax import TD3
from tests import exact_pins


def test_every_leaf_not_bitwise_the_same_is_reported() -> None:
    one = np.float32(1.0)
    ref = {"a": np.array([one, np.nan]), "b": np.zeros(2), "c": np.zeros(1)}
    got = {"a": np.array([np.nextafter(one, 2 * one), np.nan]), "b": np.zeros(3)}
    assert exact_pins.differences(ref, got) == [
        "c: only in one record",
        "a: largest relative gap 1.19e-07",
        "b: shape (2,) -> (3,)",
    ]
    assert exact_pins.differences(ref, dict(ref)) == []


def test_a_flat_replay_compares_on_its_written_rows() -> None:
    """Rows a replay has not written, past current_index, differ in number
    and value between rings of other lengths: they are not compared, the
    written ones are, all of a full ring's."""
    ring = "state.collector_state.buffer_state."

    def replay(rows: list[float], written: int, full: bool = False) -> dict:
        values = np.array(rows, np.float32).reshape(1, 1, len(rows), 1)
        return {
            ring + "experience['obs']": values,  # (seed, env, row, feature)
            ring + "current_index": np.array([written]),
            ring + "is_full": np.array([full]),
        }

    ref = replay([1.0, 2.0, 3.0, 0.0, 0.0], 3)
    assert exact_pins.differences(ref, replay([1.0, 2.0, 3.0, 9.0], 3)) == []
    assert exact_pins.differences(ref, replay([1.0, 5.0, 3.0, 0.0], 3)) == [
        ring + "experience['obs']: largest relative gap 1.50e+00"
    ]
    full = replay([1.0, 2.0, 3.0], 0, full=True)
    assert exact_pins.differences(full, replay([1.0, 2.0, 4.0], 0, full=True)) == [
        ring + "experience['obs']: largest relative gap 3.33e-01"
    ]


def test_a_recorded_run_compares_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    case = (TD3, {**exact_pins.S, **exact_pins.R}, 40)  # 10 updates a seed
    monkeypatch.setitem(exact_pins.CASES, "tiny", case)
    exact_pins.main(["record", str(tmp_path), "tiny"])
    exact_pins.main(["compare", str(tmp_path), "tiny"])
    assert capsys.readouterr().out.splitlines()[-1] == "tiny: BITWISE; jaxpr same"
