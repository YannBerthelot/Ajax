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


def test_a_recorded_run_compares_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    case = (TD3, {**exact_pins.S, **exact_pins.R}, 40)  # 10 updates a seed
    monkeypatch.setitem(exact_pins.CASES, "tiny", case)
    exact_pins.main(["record", str(tmp_path), "tiny"])
    exact_pins.main(["compare", str(tmp_path), "tiny"])
    assert capsys.readouterr().out.splitlines()[-1] == "tiny: BITWISE; jaxpr same"
