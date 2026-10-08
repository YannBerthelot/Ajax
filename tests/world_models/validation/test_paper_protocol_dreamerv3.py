"""``paper_protocol.py --smoke`` for DreamerV3, resumed from disk, and the
report on it (M9; tiny CPU run on playground CartpoleBalance: plumbing
only, no learning claim)."""

from __future__ import annotations

import os

import numpy as np
import paper_report
import pytest

from .smoke_helpers import run_twice

pytestmark = pytest.mark.slow

NAME = "dreamerv3-cartpole_balance"


def test_smoke_run_resumes_from_disk_and_reports(tmp_path):
    whole, split = run_twice(NAME, tmp_path)
    # Two chunks of 42 ticks of 4 envs, 20-step episodes (21 rows): 2
    # training episodes per env per chunk, on the reference's clock.
    assert [r["progress"] for r in whole] == [168, 336]
    assert [r["env_frames"] for r in whole] == [336, 672]
    for ours, theirs in zip(split, whole):
        assert ours["episodes"] == theirs["episodes"] == [8, 8]
        assert ours["n_updates"] == theirs["n_updates"]
        assert ours["metric"] == "train_episode_return"
        np.testing.assert_allclose(ours["value"], theirs["value"], rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(theirs["return_sum"]) / 8, theirs["value"], rtol=1e-6
        )
        assert {"git_sha", "jax", "device"} <= set(ours["provenance"])
    # The report: smoke runs reach no criteria window.
    out = tmp_path / "report.md"
    code = paper_report.main(
        [
            "--runs",
            str(tmp_path / "whole"),
            "--out",
            str(out),
            "--plot-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    text = out.read_text()
    assert "## DreamerV3: INCOMPLETE (smoke runs: plumbing only)" in text
    assert "| dmc_cartpole_balance | CartpoleBalance | 2 |" in text
    # The run.json the CLI wrote is its registry entry's: on protocol.
    assert f"| {NAME} | ok |" in text and "off protocol**" not in text
    assert "mujoco_playground" in text
    assert os.path.isfile(tmp_path / "DreamerV3_curves.png")
