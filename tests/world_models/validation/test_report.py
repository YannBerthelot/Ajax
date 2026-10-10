"""The paper-protocol report (``benchmarks/world_models/paper_report.py``).

No training: synthetic run directories (``run.json`` with the registry's
specification, ``curve.jsonl``) that reach every criteria window. Runs that
copy the reference curves pass; the env-step axis, the reached-window
check, the episode weights, the per-seed reference window means and the
TD-MPC2 budgets are pinned; runs off their protocol (other settings,
chosen seeds, an unknown name) are reported and never judged; provenance
spanning several commits or a tree with changes is flagged; a ``FAIL``
exits 1.
"""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import paper_report
import pytest
import wm_acceptance as acc
from paper_protocol import paper_runs, smoke_runs
from wm_runs import CURVE_FILE, RUN_FILE, jsonable, load_reference

DREAMER = load_reference("dreamerv3_dmc_proprio")
TDMPC2 = load_reference("tdmpc2_dmc")
PROVENANCE = {"git_sha": "a" * 40, "git_dirty": False, "jax": "0.4", "device": "cpu"}


def write_run(root, name, records, spec=None, **changes) -> None:
    """A run directory: the registry's specification of ``name`` (with
    ``changes``) and ``records``."""
    spec = spec or dataclasses.replace(paper_runs()[name], **changes)
    directory = os.path.join(root, name)
    os.makedirs(directory)
    with open(os.path.join(directory, RUN_FILE), "w") as f:
        json.dump({"spec": {"run": jsonable(dataclasses.asdict(spec))}}, f)
    with open(os.path.join(directory, CURVE_FILE), "w") as f:
        for record in records:
            f.write(json.dumps({"provenance": PROVENANCE, **record}) + "\n")


def _value_at(seed: dict, x: int) -> float:
    """A reference seed's value at ``x`` (its last value past its end)."""
    return seed["y"][seed["x"].index(x)] if x in seed["x"] else seed["y"][-1]


def replica(agent: str, task: str, scale: float = 1.0) -> list[dict]:
    """Records that copy the reference seeds' curves (times ``scale``) on
    the run's clock: progress = env steps / 2, up to the protocol budget."""
    reference = DREAMER if agent == "DreamerV3" else TDMPC2
    seeds = reference["tasks"][task]
    budget = paper_report.reference_budget(agent, reference, task)
    xs = sorted({x for s in seeds for x in s["x"] if x > 0} | {budget})
    return [
        {
            "progress": x // 2,
            "env_frames": x,
            "value": [scale * _value_at(s, x) for s in seeds],
            "episodes": [16] * len(seeds),
        }
        for x in xs
    ]


def write_replicas(root, scale: float = 1.0, skip=()) -> None:
    for name, spec in paper_runs().items():
        if name not in skip:
            write_run(root, name, replica(spec.agent, spec.reference_task, scale))


def test_runs_that_copy_the_references_pass(tmp_path):
    write_replicas(tmp_path)
    text, verdicts = paper_report.report(str(tmp_path))
    assert verdicts == [acc.PASS, acc.PASS]
    assert "## DreamerV3: PASS" in text and "## TDMPC2: PASS" in text
    assert "off protocol" not in text and "(several)" not in text
    assert paper_report.main(["--runs", str(tmp_path)]) == 0


def test_a_failing_run_set_exits_1(tmp_path):
    write_replicas(tmp_path, scale=0.0)
    _, verdicts = paper_report.report(str(tmp_path))
    assert verdicts == [acc.FAIL, acc.FAIL]
    assert paper_report.main(["--runs", str(tmp_path)]) == 1


def test_a_window_is_reached_at_its_end_exactly(tmp_path):
    """A DreamerV3 run whose last record is at 300K env steps is judged at
    the 250K window ((200K, 300K]) and not at the 500K one."""
    records = [
        r for r in replica("DreamerV3", "dmc_walker_walk") if r["env_frames"] <= 300_000
    ]
    write_run(tmp_path, "dreamerv3-walker_walk", records)
    runs = paper_report.read_runs(str(tmp_path))
    first, last = paper_report.judge_agent("DreamerV3", DREAMER, runs)
    index = list(DREAMER["playground"]).index("dmc_walker_walk")
    assert first.tasks[index].status == "pass"
    assert last.tasks[index].status == "not reached"
    # One record short of the window's end: not reached.
    assert records[-1]["env_frames"] == 300_000
    write_run(tmp_path / "short", "dreamerv3-walker_walk", records[:-1])
    runs = paper_report.read_runs(str(tmp_path / "short"))
    first, _ = paper_report.judge_agent("DreamerV3", DREAMER, runs)
    assert first.tasks[index].status == "not reached"


def test_window_scores_are_per_seed_on_env_steps_weighted_by_episodes():
    records = [
        # progress (rows) is half the env steps: only env_frames places a point.
        {"progress": 120_000, "env_frames": 240_000, "value": [100.0, 0.0],
         "episodes": [16, 16]},
        {"progress": 140_000, "env_frames": 280_000, "value": [400.0, 30.0],
         "episodes": [32, 16]},
        {"progress": 160_000, "env_frames": 320_000, "value": [999.0, 999.0],
         "episodes": [16, 16]},
    ]  # fmt: skip
    scores = paper_report.ajax_window_scores(records, 200_000, 300_000)
    np.testing.assert_allclose(scores, [(1600 + 12800) / 48, 15.0])


def test_the_reference_side_is_per_seed_window_means_on_the_task_budget():
    runs: dict = {}
    for agent, reference, n_seeds in (("DreamerV3", DREAMER, 5), ("TDMPC2", TDMPC2, 3)):
        for verdict, window in zip(
            paper_report.judge_agent(agent, reference, runs), acc.WINDOWS[agent]
        ):
            for result in verdict.tasks:
                budget = paper_report.reference_budget(agent, reference, result.task)
                lo, hi = window.bounds(budget)
                expected = [
                    acc.window_mean(s["x"], s["y"], lo, hi)
                    for s in reference["tasks"][result.task]
                ]
                assert result.reference.shape == (n_seeds,)
                np.testing.assert_allclose(result.reference, expected)
                assert result.status == "no run"
    final = acc.WINDOWS["TDMPC2"][1]
    for task in TDMPC2["playground"]:
        budget = paper_report.reference_budget("TDMPC2", TDMPC2, task)
        end = 14_000_000 if task.startswith("humanoid") else 4_000_000
        assert final.bounds(budget) == (end - 1_000_000, end)
        # The runner's budget is the report's.
        spec = paper_runs()[f"tdmpc2-{task}"]
        assert spec.env_frames(spec.budget) == budget
    assert paper_report.reference_budget("DreamerV3", DREAMER, "dmc_walker_walk") == (
        500_000
    )


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"kwargs": {**paper_runs()["dreamerv3-walker_walk"].kwargs,
                     "train_ratio": 1024}}, "kwargs.train_ratio"),
        ({"budget": 300_000}, "budget"),
        ({"chunk": 5_000}, "chunk"),
        ({"env_id": "WalkerRun"}, "env_id"),
        ({"seeds": (7, 8, 9)}, "seeds"),
        ({"seeds": (1, 2, 3)}, "seeds"),
    ],
)  # fmt: skip
def test_a_run_off_its_protocol_is_reported_not_judged(tmp_path, changes, reason):
    write_replicas(tmp_path, skip=("dreamerv3-walker_walk",))
    write_run(
        tmp_path,
        "dreamerv3-walker_walk",
        replica("DreamerV3", "dmc_walker_walk"),
        **changes,
    )
    runs = paper_report.read_runs(str(tmp_path))
    assert runs["DreamerV3", "dmc_walker_walk"]["off_protocol"] == [reason]
    verdicts = paper_report.judge_agent("DreamerV3", DREAMER, runs)
    index = list(DREAMER["playground"]).index("dmc_walker_walk")
    assert [v.tasks[index].status for v in verdicts] == ["off protocol"] * 2
    assert [v.status for v in verdicts] == [acc.INCOMPLETE] * 2
    text, _ = paper_report.report(str(tmp_path))
    assert "## DreamerV3: INCOMPLETE" in text
    assert f"| dreamerv3-walker_walk | **off protocol**: {reason} |" in text


def test_a_prefix_of_the_protocol_seeds_is_on_protocol(tmp_path):
    write_run(
        tmp_path,
        "dreamerv3-walker_walk",
        [{**r, "value": r["value"][:3], "episodes": r["episodes"][:3]}
         for r in replica("DreamerV3", "dmc_walker_walk")],
        seeds=(0, 1, 2),
    )  # fmt: skip
    runs = paper_report.read_runs(str(tmp_path))
    assert runs["DreamerV3", "dmc_walker_walk"]["off_protocol"] == []


def test_unknown_and_smoke_runs(tmp_path):
    spec = paper_runs()["dreamerv3-walker_walk"]
    write_run(tmp_path, "my-walker", [], spec=spec)
    smoke = smoke_runs()["dreamerv3-cartpole_balance"]
    write_run(tmp_path, "dreamerv3-cartpole_balance", [], spec=smoke)
    runs = paper_report.read_runs(str(tmp_path))
    assert runs["DreamerV3", "dmc_walker_walk"]["off_protocol"] == [
        "no protocol run named 'my-walker'"
    ]
    # A smoke run is checked against the smoke registry.
    assert runs["DreamerV3", "dmc_cartpole_balance"]["off_protocol"] == []
    write_run(
        tmp_path / "other",
        "dreamerv3-cartpole_balance",
        [],
        spec=dataclasses.replace(smoke, smoke=False),
    )
    runs = paper_report.read_runs(str(tmp_path / "other"))
    assert "kwargs.units" in runs["DreamerV3", "dmc_cartpole_balance"]["off_protocol"]


def test_mixed_provenance_is_flagged(tmp_path):
    records = replica("TDMPC2", "walker-walk")
    half = len(records) // 2
    for i, record in enumerate(records):
        record["provenance"] = {
            **PROVENANCE,
            "git_sha": ("a" if i < half else "b") * 40,
            "git_dirty": i == 3,
            "device": "A100" if i < half else "H100",
        }
    write_run(tmp_path, "tdmpc2-walker-walk", records)
    text, _ = paper_report.report(str(tmp_path))
    assert (
        "| tdmpc2-walker-walk | ok | aaaaaaa, bbbbbbb **(several)** | **yes** | 0.4 |"
        " A100, H100 **(several)** |"
    ) in text


def test_a_curve_cut_short_is_read_without_its_last_line(tmp_path):
    write_run(tmp_path, "tdmpc2-walker-walk", replica("TDMPC2", "walker-walk"))
    with open(tmp_path / "tdmpc2-walker-walk" / CURVE_FILE, "a") as f:
        f.write('{"progress": 6, "valu')
    runs = paper_report.read_runs(str(tmp_path))
    assert len(runs["TDMPC2", "walker-walk"]["records"]) == 40
