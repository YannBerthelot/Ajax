"""``paper_protocol.py --smoke`` for DreamerV3, resumed from disk, and the
report on it (M9; tiny CPU run on playground CartpoleBalance: plumbing
only, no learning claim).

That a resumed DreamerV3 run, restored from a checkpoint into a new agent's
skeleton, computes what the uninterrupted run computes is pinned by
``tests/agents/DreamerV3/test_dreamerv3_agent.py`` (on gymnax CartPole: no
test compares the two on playground); here, the script's records.
"""

from __future__ import annotations

import os

import numpy as np
import paper_protocol
import paper_report
import pytest
from wm_runs import build_agent

from .smoke_helpers import run_resumed

pytestmark = pytest.mark.slow

NAME = "dreamerv3-cartpole_balance"


def test_smoke_run_resumes_from_disk_and_reports(tmp_path):
    records = run_resumed(NAME, tmp_path)
    # Two chunks of 42 ticks of 4 envs, 20-step episodes (21 rows): 2
    # training episodes per env per chunk, on the reference's clock.
    assert [r["progress"] for r in records] == [168, 336]
    assert [r["env_frames"] for r in records] == [336, 672]
    schedule = build_agent(paper_protocol.smoke_runs()[NAME]).schedule
    for ticks, record in zip((42, 84), records, strict=True):
        assert record["episodes"] == [8, 8]
        assert record["n_updates"] == [schedule.total_updates(ticks)] * 2
        assert record["metric"] == "train_episode_return"
        np.testing.assert_allclose(
            np.asarray(record["return_sum"]) / 8, record["value"], rtol=1e-6
        )
        assert {"git_sha", "jax", "device"} <= set(record["provenance"])
    # The report: smoke runs reach no criteria window.
    out = tmp_path / "report.md"
    code = paper_report.main(
        [
            "--runs",
            str(tmp_path / "split"),
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
