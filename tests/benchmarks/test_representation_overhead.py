"""The overhead script runs, on a tiny run.

Contract (its docstring): one row per pair, holding the CPU seconds of both
programs and their ratio.
"""

from __future__ import annotations

import json
import sys

import representation_overhead


def test_a_pair_writes_a_row_with_both_programs(tmp_path, monkeypatch):
    """Catches the script falling behind the train function it builds by hand
    (an argument added to make_train), which only a measurement would show,
    weeks later."""
    out = tmp_path / "rows.jsonl"
    arguments = "--timesteps 16 --probes 2 --pairs 1 --out".split()
    monkeypatch.setattr(sys, "argv", ["overhead", *arguments, str(out)])

    representation_overhead.main()

    row = json.loads(out.read_text())
    assert row["off_cpu_s"] > 0
    assert row["on_cpu_s"] > 0
