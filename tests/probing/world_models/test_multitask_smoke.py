"""``multitask_validation.py --smoke`` end to end (M9; tiny CPU runs on two
playground tasks of different dims: plumbing only, no learning claim).

Source runs (full history exported, no eviction), the pooled dataset saved
and loaded back, offline multi-task training and the report; a second
invocation finds every stage done. It trains, so it is in the slow tier; the
pipeline's checks without training are
``tests/world_models/validation/test_multitask_validation.py``.
"""

from __future__ import annotations

import json

import multitask_validation as mt
import numpy as np
import pytest

from ajax.agents.TDMPC2.dataset import load_dataset


@pytest.mark.slow
def test_smoke_pipeline_end_to_end(tmp_path, capsys):
    out = str(tmp_path)
    assert mt.main(["--smoke", "--out", out]) == 0
    protocol = mt.SMOKE
    seeds = len(protocol.source_seeds)
    episodes = protocol.source_budget // 20  # T = 20 agent steps

    # Every source episode of every seed, in the source runs' order.
    dataset = load_dataset(str(tmp_path / "dataset.npz"))
    assert dataset.names == protocol.tasks
    assert dataset.obs_dims == (6, 3) and dataset.action_dims == (2, 1)
    assert dataset.episode_lengths == (20, 20) and dataset.rows == 21
    np.testing.assert_array_equal(dataset.episode_counts(), [seeds * episodes] * 2)
    for i, task in enumerate(protocol.tasks):
        source = load_dataset(mt.episodes_path(out, task))
        np.testing.assert_array_equal(dataset.task_episodes(i).obs, source.obs)
        assert not (tmp_path / "sources" / task / "state.pkl").exists()
    summary = json.loads((tmp_path / "dataset.json").read_text())
    assert summary["episodes_per_task"] == [seeds * episodes] * 2

    # Offline training evaluated every task after each chunk.
    lines = (tmp_path / "offline" / "curve.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["progress"] for r in records] == [10, 20]
    assert set(records[-1]["returns"]) == set(protocol.tasks)
    report = (tmp_path / "multitask_report.md").read_text()
    assert "smoke run: plumbing only" in report and "mujoco_playground" in report
    assert (tmp_path / "multitask.png").exists()

    # Everything is done: a second run only reports.
    capsys.readouterr()
    assert mt.main(["--smoke", "--out", out]) == 0
    printed = capsys.readouterr().out
    assert "exported already" in printed and "built already" in printed
    assert "[train]" not in printed
