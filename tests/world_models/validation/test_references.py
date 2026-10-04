"""The published reference curves (``benchmarks/world_models/references``).

The extraction (``extract_references.py``) on small synthetic inputs of
each source format, the task maps, and the committed files: provenance,
units, task maps against playground's DMC suite, and values pinned by the
specs (dreamerv3_spec 7.7: the bundled curves' task mean / median at 250K
and 490K env steps; tdmpc2_spec 4.27: per-task seed means at 100K-14M env
steps), which also pin the x-axis unit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from extract_references import (
    DREAMERV3_TO_PLAYGROUND,
    TDMPC2_DMC_UNAVAILABLE,
    TDMPC2_TO_PLAYGROUND,
    classify_tdmpc2_task,
    dump,
    mt30_playground,
    parse_dreamerv3_scores,
    parse_tdmpc2_csv,
    read_task_set,
)
from wm_runs import load_reference

#: mujoco_playground's dm_control_suite (mujoco_playground 0.2).
PLAYGROUND_DMC = {
    "AcrobotSwingup", "AcrobotSwingupSparse", "BallInCup", "CartpoleBalance",
    "CartpoleBalanceSparse", "CartpoleSwingup", "CartpoleSwingupSparse",
    "CheetahRun", "FingerSpin", "FingerTurnEasy", "FingerTurnHard", "FishSwim",
    "HopperHop", "HopperStand", "HumanoidRun", "HumanoidStand", "HumanoidWalk",
    "PendulumSwingup", "PointMass", "ReacherEasy", "ReacherHard",
    "SwimmerSwimmer6", "WalkerRun", "WalkerStand", "WalkerWalk",
}  # fmt: skip

MT30 = [
    "walker-stand", "walker-walk", "walker-run", "cheetah-run", "reacher-easy",
    "reacher-hard", "acrobot-swingup", "pendulum-swingup", "cartpole-balance",
    "cartpole-balance-sparse", "cartpole-swingup", "cartpole-swingup-sparse",
    "cup-catch", "finger-spin", "finger-turn-easy", "finger-turn-hard",
    "fish-swim", "hopper-stand", "hopper-hop", "walker-walk-backwards",
    "walker-run-backwards", "cheetah-run-backwards", "cheetah-run-front",
    "cheetah-run-back", "cheetah-jump", "hopper-hop-backwards",
    "reacher-three-easy", "reacher-three-hard", "cup-spin", "pendulum-spin",
]  # fmt: skip


#: The only reference tasks whose playground id is not their name.
RENAMED = {"dmc_ball_in_cup_catch": "BallInCup", "cup-catch": "BallInCup"}


def assert_mapped_by_name(task_map: dict) -> None:
    """Every reference task maps to the playground task of the same name
    (``dmc_walker_walk`` / ``walker-walk`` -> ``WalkerWalk``), except
    :data:`RENAMED`: two swapped entries cannot pass."""
    for task, env in task_map.items():
        if task in RENAMED:
            assert env == RENAMED[task], task
        else:
            name = task.removeprefix("dmc_").replace("_", "").replace("-", "")
            assert env.lower() == name, (task, env)


def test_the_task_maps_follow_the_names():
    assert_mapped_by_name(DREAMERV3_TO_PLAYGROUND)
    assert_mapped_by_name(TDMPC2_TO_PLAYGROUND)
    assert len(set(DREAMERV3_TO_PLAYGROUND.values())) == 18
    assert len(set(TDMPC2_TO_PLAYGROUND.values())) == 22
    dreamer, tdmpc2 = (
        load_reference("dreamerv3_dmc_proprio"),
        load_reference("tdmpc2_dmc"),
    )
    assert_mapped_by_name(dreamer["playground"])
    assert_mapped_by_name(tdmpc2["playground"])
    assert_mapped_by_name(tdmpc2["mt30"]["playground"])


def test_dreamerv3_scores_are_parsed_per_task_and_seed():
    records = [
        {"task": "b", "method": "dreamerv3", "seed": 1, "xs": [1e4, 2e4], "ys": [1.234, 2.0]},
        {"task": "b", "method": "dreamerv3", "seed": 0, "xs": [1e4, 2e4], "ys": [0.5, 0.0]},
        {"task": "a", "method": "dreamerv3", "seed": 0, "xs": [1e4], "ys": [3.0]},
    ]  # fmt: skip
    tasks = parse_dreamerv3_scores(records)
    assert list(tasks) == ["a", "b"]
    assert tasks["b"] == [
        {"seed": 0, "x": [10_000, 20_000], "y": [0.5, 0.0]},
        {"seed": 1, "x": [10_000, 20_000], "y": [1.23, 2.0]},
    ]
    with pytest.raises(ValueError, match="method"):
        parse_dreamerv3_scores([{**records[0], "method": "ppo"}])
    with pytest.raises(ValueError, match="x grid"):
        parse_dreamerv3_scores([records[0], {**records[1], "xs": [1e4, 3e4]}])
    with pytest.raises(ValueError, match="bad curve"):
        parse_dreamerv3_scores([{**records[0], "xs": [1e4]}])


def test_tdmpc2_csvs_are_parsed_per_seed(tmp_path):
    path = tmp_path / "task.csv"
    path.write_text(
        "step,reward,seed\n100000,5.5,2\n0,1.0,2\n0,2.0,1\n100000,7.0,1\n200000,8,1\n"
    )
    assert parse_tdmpc2_csv(str(path)) == [
        {"seed": 1, "x": [0, 100_000, 200_000], "y": [2.0, 7.0, 8.0]},
        {"seed": 2, "x": [0, 100_000], "y": [1.0, 5.5]},
    ]
    path.write_text("step,episode_reward\n0,1\n")
    with pytest.raises(ValueError, match="columns"):
        parse_tdmpc2_csv(str(path))
    path.write_text("step,reward,seed\n0,1,1\n0,2,1\n")
    with pytest.raises(ValueError, match="repeats a step"):
        parse_tdmpc2_csv(str(path))


def test_every_tdmpc2_result_is_classified():
    assert classify_tdmpc2_task("walker-walk") == "playground"
    assert classify_tdmpc2_task("dog-run") == "dmc-unavailable"
    assert classify_tdmpc2_task("mw-assembly") == "non-dmc"
    assert classify_tdmpc2_task("myo-hand-reach") == "non-dmc"
    assert classify_tdmpc2_task("pick-ycb") == "non-dmc"
    with pytest.raises(ValueError, match="unclassified"):
        classify_tdmpc2_task("quadruped-fetch")
    assert set(TDMPC2_TO_PLAYGROUND.values()) <= PLAYGROUND_DMC
    assert set(DREAMERV3_TO_PLAYGROUND.values()) <= PLAYGROUND_DMC
    assert not set(TDMPC2_TO_PLAYGROUND) & set(TDMPC2_DMC_UNAVAILABLE)
    # 22 + 17 = the 39 DMC tasks of the released results (tdmpc2_spec 4.27).
    assert len(TDMPC2_TO_PLAYGROUND) + len(TDMPC2_DMC_UNAVAILABLE) == 39


def test_the_mt30_task_set_is_read_and_mapped(tmp_path):
    module = tmp_path / "common.py"
    module.write_text(
        "import torch\nMODEL_SIZE = {}\n"
        f"TASK_SET = {{'mt30': {MT30!r}, 'mt80': {[*MT30, 'mw-reach']!r}}}\n"
    )
    assert read_task_set(str(module), "mt30") == MT30
    available, unavailable = mt30_playground(MT30)
    assert list(available) == MT30[:19]  # the 19 original DMC tasks
    assert available["cup-catch"] == "BallInCup"
    assert unavailable == MT30[19:]  # the 11 custom tasks
    with pytest.raises(ValueError, match="known DMC"):
        mt30_playground(["mw-reach"])
    (tmp_path / "empty.py").write_text("X = 1\n")
    with pytest.raises(ValueError, match="TASK_SET"):
        read_task_set(str(tmp_path / "empty.py"), "mt30")


def test_dump_writes_one_task_per_line_and_round_trips(tmp_path):
    reference = {
        "name": "r",
        "source": {"license": "MIT"},
        "tasks": {"a": [{"seed": 0, "x": [1, 2], "y": [0.5, 1.0]}], "b": []},
    }
    path = tmp_path / "r.json"
    dump(reference, str(path))
    assert json.loads(path.read_text()) == reference
    lines = path.read_text().splitlines()
    assert lines[-4:] == [
        '  "a": [{"seed":0,"x":[1,2],"y":[0.5,1.0]}],',
        '  "b": []',
        " }",
        "}",
    ]


def _seed_means(reference: dict, task: str, x: int) -> float:
    return float(np.mean([s["y"][s["x"].index(x)] for s in reference["tasks"][task]]))


def test_the_committed_dreamerv3_reference():
    ref = load_reference("dreamerv3_dmc_proprio")
    assert ref["source"]["commit"].startswith("29eb964")
    assert ref["source"]["file_added_in"].startswith("2411f7d")
    assert ref["source"]["license"] == "MIT"
    assert "Danijar Hafner" in ref["source"]["copyright"]
    assert ref["units"]["x"].startswith("env (simulator) steps")
    assert "training episodes of the stochastic" in ref["units"]["y"]
    assert ref["playground"] == DREAMERV3_TO_PLAYGROUND and ref["unavailable"] == []
    protocol = ref["protocol"]
    assert (
        protocol["model_size"],
        protocol["train_ratio"],
        protocol["action_repeat"],
    ) == (
        "12m",
        512,
        2,
    )
    assert (protocol["n_envs"], protocol["env_steps"], protocol["seeds"]) == (
        16,
        500_000,
        5,
    )
    assert set(ref["tasks"]) == set(DREAMERV3_TO_PLAYGROUND)
    for curves in ref["tasks"].values():
        assert [s["seed"] for s in curves] == [0, 1, 2, 3, 4]
        assert all(s["x"] == list(range(10_000, 500_000, 10_000)) for s in curves)
    # dreamerv3_spec 7.7: task mean / median of the bundled curves.
    for x, mean, median in ((250_000, 675.4, 790.4), (490_000, 757.8, 868.5)):
        means = [_seed_means(ref, task, x) for task in ref["tasks"]]
        assert np.mean(means) == pytest.approx(mean, abs=0.05)
        assert np.median(means) == pytest.approx(median, abs=0.05)


def test_the_committed_tdmpc2_reference():
    ref = load_reference("tdmpc2_dmc")
    assert ref["source"]["commit"].startswith("5f6fade")
    assert ref["source"]["file_added_in"].startswith("b67b21c")
    assert ref["source"]["license"] == "MIT"
    assert "Nicklas Hansen" in ref["source"]["copyright"]
    assert ref["units"]["x"].startswith("env (simulator) steps")
    assert "eval_mode" in ref["units"]["y"]
    protocol = ref["protocol"]
    assert (protocol["model_size"], protocol["action_repeat"], protocol["seeds"]) == (
        5,
        2,
        3,
    )
    assert (protocol["eval_episodes"], protocol["eval_freq_agent_steps"]) == (
        10,
        50_000,
    )
    assert ref["playground"] == TDMPC2_TO_PLAYGROUND
    assert ref["unavailable"] == sorted(TDMPC2_DMC_UNAVAILABLE)
    assert ref["non_dmc_results"] == 65  # 50 Meta-World, 10 MyoSuite, 5 ManiSkill2
    assert ref["mt30"]["order"] == MT30
    assert list(ref["mt30"]["playground"]) == MT30[:19]
    assert ref["mt30"]["unavailable"] == MT30[19:]
    assert set(ref["tasks"]) == set(TDMPC2_TO_PLAYGROUND)
    for task, curves in ref["tasks"].items():
        assert [s["seed"] for s in curves] == [1, 2, 3]
        budget = 14_000_000 if task.startswith("humanoid") else 4_000_000
        assert max(max(s["x"]) for s in curves) == budget
        # One evaluation every 50K agent steps = 100K env steps (repeat 2).
        assert all(np.all(np.diff(s["x"]) == 100_000) for s in curves)
    # tdmpc2_spec 4.27: seed means at 100K / 500K / 1M / 2M / 4M env steps.
    expected = {
        "walker-walk": (962, 979, 980, 984, 981),
        "cheetah-run": (519, 758, 844, 860, 896),
        "hopper-hop": (14, 303, 338, 375, 449),
        "humanoid-run": (1, 108, 185, 316, 461),
    }
    for task, values in expected.items():
        for x, value in zip(
            (100_000, 500_000, 1_000_000, 2_000_000, 4_000_000), values
        ):
            assert _seed_means(ref, task, x) == pytest.approx(value, abs=0.5)
    assert _seed_means(ref, "humanoid-run", 14_000_000) == pytest.approx(603, abs=0.5)
