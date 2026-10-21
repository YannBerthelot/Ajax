"""``multitask_validation.py`` without training: the report judges the final
records of runs on protocol, and a dataset whose sources changed is refused.
The ``--smoke`` pipeline end to end trains, so it is in the slow tier
(``tests/probing/world_models/test_multitask_smoke.py``).
"""

from __future__ import annotations

import dataclasses
import json
import os

import multitask_validation as mt
import numpy as np
import pytest
from wm_runs import RUN_FILE

from ajax.agents.TDMPC2.dataset import (
    TaskEpisodes,
    pool_tasks,
    save_dataset,
)

from .smoke_helpers import write_run_dir


def _episodes(task: str, seed: int) -> TaskEpisodes:
    rng = np.random.default_rng(seed)
    return TaskEpisodes(
        obs=rng.normal(size=(2, 21, 3)).astype(np.float32),
        action=np.zeros((2, 21, 1), np.float32),
        reward=np.zeros((2, 21), np.float32),
        episode_length=20,
        name=task,
    )


def _pipeline(tmp_path, protocol=mt.SMOKE) -> str:
    """A finished pipeline directory without training: source exports, the
    dataset, and source and offline curves of two records each."""
    out = str(tmp_path)
    for i, task in enumerate(protocol.tasks):
        save_dataset(pool_tasks([_episodes(task, i)]), mt.episodes_path(out, task))
    mt.build_dataset(protocol, out)
    with open(os.path.join(out, "dataset.json")) as f:
        dataset = json.load(f)
    finals = {"reacher-easy": [300.0, 500.0], "pendulum-swingup": [200.0, 200.0]}
    for task in protocol.tasks:
        write_run_dir(
            os.path.join(out, "sources", task),
            {"run": dataclasses.asdict(mt.source_spec(protocol, task))},
            [
                {"progress": protocol.source_budget // 2, "value": [10.0, 20.0]},
                {"progress": protocol.source_budget, "value": finals[task]},
            ],
        )
    write_run_dir(
        os.path.join(out, "offline"),
        {"protocol": dataclasses.asdict(protocol), "dataset": dataset},
        [
            {"progress": protocol.offline_updates // 2, "normalized_score": [1.0],
             "returns": {"reacher-easy": [1.0], "pendulum-swingup": [1.0]}},
            {"progress": protocol.offline_updates, "normalized_score": [2.0],
             "returns": {"reacher-easy": [200.0], "pendulum-swingup": [99.0]}},
        ],
    )  # fmt: skip
    return out


def test_the_report_judges_the_final_records(tmp_path):
    out = _pipeline(tmp_path)
    source, offline = mt.collect(mt.SMOKE, mt.pipeline_runs(mt.SMOKE, out))
    assert source == {
        "reacher-easy": [300.0, 500.0],
        "pendulum-swingup": [200.0, 200.0],
    }
    assert offline == {"reacher-easy": [200.0], "pendulum-swingup": [99.0]}
    # Both judged, one passes: 50% < 80%. (The first records would leave
    # both sources below the floor: INCOMPLETE.)
    assert mt.write_report(mt.SMOKE, out) == "FAIL"
    report = (tmp_path / "multitask_report.md").read_text()
    assert "| reacher-easy | ReacherEasy | 400 | 200 | 0.50 | pass |" in report
    assert "| pendulum-swingup | PendulumSwingup | 200 | 99 | 0.49 | fail |" in report
    assert "Judged: 2 of 2 tasks." in report
    assert "| sources/reacher-easy | ok |" in report and "| offline | ok |" in report


def test_runs_off_the_protocol_are_not_used(tmp_path):
    out = _pipeline(tmp_path)
    # A source run of another budget, and an offline run on another dataset.
    spec_path = tmp_path / "sources" / "reacher-easy" / RUN_FILE
    spec = json.loads(spec_path.read_text())
    spec["spec"]["run"]["budget"] = 30
    spec_path.write_text(json.dumps(spec))
    meta = json.loads((tmp_path / "dataset.json").read_text())
    meta["rows"] += 1
    (tmp_path / "dataset.json").write_text(json.dumps(meta))
    runs = mt.pipeline_runs(mt.SMOKE, out)
    assert runs["sources/reacher-easy"]["off_protocol"] == ["run.budget"]
    assert runs["offline"]["off_protocol"] == ["dataset.rows"]
    source, offline = mt.collect(mt.SMOKE, runs)
    assert source["reacher-easy"] == [] and offline == {}
    assert mt.write_report(mt.SMOKE, out) == "INCOMPLETE"
    report = (tmp_path / "multitask_report.md").read_text()
    assert "| sources/reacher-easy | **off protocol**: run.budget |" in report


def test_a_dataset_whose_sources_changed_is_refused(tmp_path, capsys):
    out = _pipeline(tmp_path)
    meta = json.loads((tmp_path / "dataset.json").read_text())
    assert set(meta["sources"]) == set(mt.SMOKE.tasks)
    assert meta["dataset"] == mt.file_digest(str(tmp_path / "dataset.npz"))
    assert len(meta["dataset"]["sha256"]) == 64
    mt.build_dataset(mt.SMOKE, out)
    assert "built already" in capsys.readouterr().out
    # A source run redone: its export changed.
    task = mt.SMOKE.tasks[1]
    save_dataset(pool_tasks([_episodes(task, 7)]), mt.episodes_path(out, task))
    with pytest.raises(SystemExit, match="stale"):
        mt.build_dataset(mt.SMOKE, out)
    with pytest.raises(SystemExit, match="stale"):
        mt.check_dataset(mt.SMOKE, out)
