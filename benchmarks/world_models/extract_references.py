"""Extract the published DreamerV3 and TD-MPC2 DMC curves into compact JSON.

The M9 validation (``docs/world_models/VALIDATION.md``) compares Ajax's
world-model agents with the curves their authors published next to the
code. This script reads them from local checkouts of the official
repositories and writes the two committed reference files under
``benchmarks/world_models/references/``:

* ``dreamerv3_dmc_proprio.json``: ``danijar/dreamerv3``'s bundled
  ``scores/dmc_proprio-dreamerv3.json.gz`` (added in ``2411f7d``, the
  paper-era commit; read at ``29eb964``): the 18 DMC-proprio tasks of the
  paper's Table 11, 5 seeds, the returns of the stochastic policy's
  *training* episodes (``29eb964:embodied/run/train.py:36-61`` sums each
  training episode's rewards into ``episode/score``; dreamerv3_spec 7.3,
  7.7) binned every 10K env steps from 10K to 490K. The x-axis counts env
  (simulator) steps: the logger multiplies the agent step by the action
  repeat (``29eb964:dreamerv3/main.py:150``), which is 2 on DMC
  (``dreamerv3/configs.yaml:84``). The file does not document its binning
  (with 16 lockstep envs an episode ends only every ~16K env steps, so its
  10K bins must have been filled or smoothed).
* ``tdmpc2_dmc.json``: ``nicklashansen/tdmpc2``'s ``results/<task>.csv``
  (added in ``b67b21c``; read at ``5f6fade``) for the DMC tasks that
  mujoco_playground implements, 3 seeds, columns ``step, reward, seed``:
  the mean return of 10 evaluation episodes of the planner in
  ``eval_mode`` (``5f6fade:tdmpc2/trainer/online_trainer.py:27-48``,
  ``config.yaml:10-11``; tdmpc2_spec 4.21) every 50K agent steps, with
  ``step`` in env steps = agent steps x action repeat 2 (tdmpc2_spec 4.22;
  ``envs/dmcontrol.py:196``). It also records the mt30 task list
  (``tdmpc2/common/__init__.py:26-37``) and its playground equivalents.

Both repositories are MIT-licensed; the JSON files carry the source
repository, commit, file, license and copyright line.

Every reference task is mapped to a mujoco_playground ``dm_control_suite``
env id explicitly (:data:`DREAMERV3_TO_PLAYGROUND`,
:data:`TDMPC2_TO_PLAYGROUND`) or listed as unavailable; the script raises
on any DMC task it cannot classify. Playground tasks are MJX
re-implementations of dm_control: physics, observations and sometimes
rewards differ, so comparisons with these curves are approximate.

Usage (once; the outputs are committed)::

    python benchmarks/world_models/extract_references.py \\
        --dreamerv3 path/to/dreamerv3@29eb964 --tdmpc2 path/to/tdmpc2@5f6fade
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import os
import subprocess
from collections import defaultdict
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_DIR = os.path.join(HERE, "references")
DREAMERV3_FILE = "dreamerv3_dmc_proprio.json"
TDMPC2_FILE = "tdmpc2_dmc.json"

DREAMERV3_COMMIT = "29eb964e2918a3f4db04086f7f51b60388e97f3d"
DREAMERV3_SCORES_ADDED_IN = "2411f7d136832378c0291c587cdbf2fca6506873"
DREAMERV3_SCORES = "scores/dmc_proprio-dreamerv3.json.gz"
TDMPC2_COMMIT = "5f6fadec0fec78304b4b53e8171d348b58cac486"
TDMPC2_RESULTS_ADDED_IN = "b67b21c5c638b48b1351864572ee834127717e98"
TDMPC2_RESULTS = "results"
TDMPC2_TASK_SET = "tdmpc2/common/__init__.py"

#: The 18 DMC-proprio tasks of DreamerV3's Table 11 (p.34) -> playground ids.
DREAMERV3_TO_PLAYGROUND: dict[str, str] = {
    "dmc_acrobot_swingup": "AcrobotSwingup",
    "dmc_ball_in_cup_catch": "BallInCup",
    "dmc_cartpole_balance": "CartpoleBalance",
    "dmc_cartpole_balance_sparse": "CartpoleBalanceSparse",
    "dmc_cartpole_swingup": "CartpoleSwingup",
    "dmc_cartpole_swingup_sparse": "CartpoleSwingupSparse",
    "dmc_cheetah_run": "CheetahRun",
    "dmc_finger_spin": "FingerSpin",
    "dmc_finger_turn_easy": "FingerTurnEasy",
    "dmc_finger_turn_hard": "FingerTurnHard",
    "dmc_hopper_hop": "HopperHop",
    "dmc_hopper_stand": "HopperStand",
    "dmc_pendulum_swingup": "PendulumSwingup",
    "dmc_reacher_easy": "ReacherEasy",
    "dmc_reacher_hard": "ReacherHard",
    "dmc_walker_run": "WalkerRun",
    "dmc_walker_stand": "WalkerStand",
    "dmc_walker_walk": "WalkerWalk",
}

#: TD-MPC2's DMC tasks with a playground version -> playground ids.
TDMPC2_TO_PLAYGROUND: dict[str, str] = {
    "acrobot-swingup": "AcrobotSwingup",
    "cartpole-balance": "CartpoleBalance",
    "cartpole-balance-sparse": "CartpoleBalanceSparse",
    "cartpole-swingup": "CartpoleSwingup",
    "cartpole-swingup-sparse": "CartpoleSwingupSparse",
    "cheetah-run": "CheetahRun",
    "cup-catch": "BallInCup",
    "finger-spin": "FingerSpin",
    "finger-turn-easy": "FingerTurnEasy",
    "finger-turn-hard": "FingerTurnHard",
    "fish-swim": "FishSwim",
    "hopper-hop": "HopperHop",
    "hopper-stand": "HopperStand",
    "humanoid-run": "HumanoidRun",
    "humanoid-stand": "HumanoidStand",
    "humanoid-walk": "HumanoidWalk",
    "pendulum-swingup": "PendulumSwingup",
    "reacher-easy": "ReacherEasy",
    "reacher-hard": "ReacherHard",
    "walker-run": "WalkerRun",
    "walker-stand": "WalkerStand",
    "walker-walk": "WalkerWalk",
}

#: TD-MPC2's DMC tasks that playground does not implement: Dog and
#: Quadruped, and the 11 custom tasks of mt30 (tdmpc2_spec 4.24).
TDMPC2_DMC_UNAVAILABLE: tuple[str, ...] = (
    "cheetah-jump",
    "cheetah-run-back",
    "cheetah-run-backwards",
    "cheetah-run-front",
    "cup-spin",
    "dog-run",
    "dog-stand",
    "dog-trot",
    "dog-walk",
    "hopper-hop-backwards",
    "pendulum-spin",
    "quadruped-run",
    "quadruped-walk",
    "reacher-three-easy",
    "reacher-three-hard",
    "walker-run-backwards",
    "walker-walk-backwards",
)

#: Result files of other domains (Meta-World, MyoSuite, ManiSkill2): not DMC.
TDMPC2_NON_DMC_PREFIXES: tuple[str, ...] = ("mw-", "myo-")
TDMPC2_NON_DMC_TASKS: tuple[str, ...] = (
    "lift-cube",
    "pick-cube",
    "pick-ycb",
    "stack-cube",
    "turn-faucet",
)


def git_commit(checkout: str) -> str:
    """The commit a checkout is at (``git rev-parse HEAD``)."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()


def license_line(checkout: str) -> str:
    """The copyright line of the checkout's MIT ``LICENSE``."""
    with open(os.path.join(checkout, "LICENSE")) as f:
        text = f.read()
    if "MIT License" not in text and "Permission is hereby granted" not in text:
        raise ValueError(f"{checkout}/LICENSE is not the MIT license")
    return next(line.strip() for line in text.splitlines() if "Copyright" in line)


def parse_dreamerv3_scores(records: list[dict]) -> dict[str, list[dict]]:
    """Per task, the seeds' curves ``{seed, x, y}`` (x: int env steps, y:
    returns rounded to 0.01), seeds in order.

    ``records`` is the bundled file's list of ``{task, method, seed, xs,
    ys}``; every record must be DreamerV3's and every seed of a task must
    share the x grid.
    """
    tasks: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["method"] != "dreamerv3":
            raise ValueError(f"unexpected method {record['method']!r}")
        xs, ys = record["xs"], record["ys"]
        if len(xs) != len(ys) or any(x != int(x) for x in xs):
            raise ValueError(f"{record['task']} seed {record['seed']}: bad curve")
        tasks[record["task"]].append(
            {
                "seed": int(record["seed"]),
                "x": [int(x) for x in xs],
                "y": [round(float(y), 2) for y in ys],
            }
        )
    for task, seeds in tasks.items():
        seeds.sort(key=lambda s: s["seed"])
        if len({tuple(s["x"]) for s in seeds}) != 1:
            raise ValueError(f"{task}: the seeds do not share an x grid")
    return dict(sorted(tasks.items()))


def parse_tdmpc2_csv(path: str) -> list[dict]:
    """One results CSV (``step,reward,seed``) as per-seed curves ``{seed, x,
    y}``, seeds in order, points by step."""
    seeds: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["step", "reward", "seed"]:
            raise ValueError(f"{path}: columns {reader.fieldnames}")
        for row in reader:
            step = float(row["step"])
            if step != int(step):
                raise ValueError(f"{path}: non-integer step {row['step']}")
            seeds[int(row["seed"])].append((int(step), float(row["reward"])))
    out = []
    for seed in sorted(seeds):
        points = sorted(seeds[seed])
        if len({x for x, _ in points}) != len(points):
            raise ValueError(f"{path}: seed {seed} repeats a step")
        out.append(
            {
                "seed": seed,
                "x": [x for x, _ in points],
                "y": [y for _, y in points],
            }
        )
    return out


def classify_tdmpc2_task(task: str) -> str:
    """``"playground"``, ``"dmc-unavailable"`` or ``"non-dmc"``; raises on a
    task none of the lists names."""
    if task in TDMPC2_TO_PLAYGROUND:
        return "playground"
    if task in TDMPC2_DMC_UNAVAILABLE:
        return "dmc-unavailable"
    if task.startswith(TDMPC2_NON_DMC_PREFIXES) or task in TDMPC2_NON_DMC_TASKS:
        return "non-dmc"
    raise ValueError(f"unclassified TD-MPC2 result {task!r}: map it or list it")


def read_task_set(path: str, name: str) -> list[str]:
    """``TASK_SET[name]`` of TD-MPC2's ``common/__init__.py`` (parsed, not
    imported: the module imports torch)."""
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "TASK_SET" for t in node.targets
        ):
            return list(ast.literal_eval(node.value)[name])
    raise ValueError(f"no TASK_SET in {path}")


def mt30_playground(mt30: list[str]) -> tuple[dict[str, str], list[str]]:
    """The mt30 tasks with a playground version (mt30 order) and the others."""
    available = {t: TDMPC2_TO_PLAYGROUND[t] for t in mt30 if t in TDMPC2_TO_PLAYGROUND}
    unavailable = [t for t in mt30 if t not in TDMPC2_TO_PLAYGROUND]
    for task in unavailable:
        if classify_tdmpc2_task(task) != "dmc-unavailable":
            raise ValueError(f"mt30 task {task!r} is not a known DMC task")
    return available, unavailable


def dreamerv3_reference(checkout: str) -> dict[str, Any]:
    with gzip.open(os.path.join(checkout, DREAMERV3_SCORES), "rt") as f:
        tasks = parse_dreamerv3_scores(json.load(f))
    if set(tasks) != set(DREAMERV3_TO_PLAYGROUND):
        raise ValueError(
            f"the bundled tasks {sorted(tasks)} are not the 18 Table 11 tasks"
        )
    return {
        "name": "dreamerv3_dmc_proprio",
        "source": {
            "repository": "https://github.com/danijar/dreamerv3",
            "commit": git_commit(checkout),
            "file": DREAMERV3_SCORES,
            "file_added_in": DREAMERV3_SCORES_ADDED_IN,
            "license": "MIT",
            "copyright": license_line(checkout),
        },
        "units": {
            "x": "env (simulator) steps = agent steps x action repeat 2,"
            " summed over the 16 envs (29eb964:dreamerv3/main.py:150)",
            "y": "undiscounted return of training episodes of the stochastic"
            " policy (episode/score, 29eb964:embodied/run/train.py:36-61),"
            " binned every 10K env steps (binning not documented)",
        },
        "protocol": {
            "model_size": "12m",
            "train_ratio": 512,
            "action_repeat": 2,
            "n_envs": 16,
            "episode_length": 1000,
            "env_steps": 500_000,
            "seeds": 5,
            "citation": "paper Table 2 p.19 and Table 11 p.34;"
            " 29eb964:dreamerv3/configs.yaml:44 (16 envs), :84 (dmc repeat 2),"
            " :232-236 (dmc_proprio: size12m, train_ratio 512);"
            " dreamerv3_spec 7.5-7.7",
        },
        "playground": dict(DREAMERV3_TO_PLAYGROUND),
        "unavailable": [],
        "tasks": tasks,
    }


def tdmpc2_reference(checkout: str) -> dict[str, Any]:
    results = os.path.join(checkout, TDMPC2_RESULTS)
    names = sorted(f[: -len(".csv")] for f in os.listdir(results) if f.endswith(".csv"))
    kinds = {name: classify_tdmpc2_task(name) for name in names}
    missing = sorted(set(TDMPC2_TO_PLAYGROUND) - set(names))
    if missing:
        raise ValueError(f"no results CSV for {missing}")
    tasks = {
        name: parse_tdmpc2_csv(os.path.join(results, f"{name}.csv"))
        for name in names
        if kinds[name] == "playground"
    }
    mt30 = read_task_set(os.path.join(checkout, TDMPC2_TASK_SET), "mt30")
    mt30_available, mt30_unavailable = mt30_playground(mt30)
    return {
        "name": "tdmpc2_dmc",
        "source": {
            "repository": "https://github.com/nicklashansen/tdmpc2",
            "commit": git_commit(checkout),
            "file": "results/<task>.csv",
            "file_added_in": TDMPC2_RESULTS_ADDED_IN,
            "license": "MIT",
            "copyright": license_line(checkout),
        },
        "units": {
            "x": "env (simulator) steps = agent steps x action repeat 2"
            " (tdmpc2_spec 4.22)",
            "y": "mean undiscounted return of 10 evaluation episodes of the"
            " planner in eval_mode, every 50K agent steps"
            " (5f6fade:tdmpc2/trainer/online_trainer.py:27-48,"
            " tdmpc2/config.yaml:10-11)",
        },
        "protocol": {
            "model_size": 5,
            "action_repeat": 2,
            "n_envs": 1,
            "episode_length": 1000,
            "eval_episodes": 10,
            "eval_freq_agent_steps": 50_000,
            "eval_mode": True,
            "buffer_size": 1_000_000,
            "seeds": 3,
            "env_steps": "per task, the last step of its CSV (4M; 14M for"
            " humanoid; 5f6fade README.md:111 trains dog-run with"
            " steps=7000000 agent steps)",
            "citation": "paper App. C Table 6 p.20, Sec. 4 p.6 (5M for"
            " single-task); 5f6fade README.md:102, :115 (model_size=5);"
            " tdmpc2/config.yaml:10-11, :27; envs/dmcontrol.py:196;"
            " tdmpc2_spec 4.11, 4.21, 4.22, 4.30",
        },
        "playground": dict(TDMPC2_TO_PLAYGROUND),
        "unavailable": sorted(n for n in names if kinds[n] == "dmc-unavailable"),
        "non_dmc_results": sum(kind == "non-dmc" for kind in kinds.values()),
        "mt30": {
            "order": mt30,
            "citation": f"5f6fade:{TDMPC2_TASK_SET}:26-37",
            "playground": mt30_available,
            "unavailable": mt30_unavailable,
        },
        "tasks": tasks,
    }


def dump(reference: dict[str, Any], path: str) -> None:
    """Write ``reference`` with the metadata indented and one task per line."""
    tasks = reference["tasks"]
    head = {k: v for k, v in reference.items() if k != "tasks"}
    text = json.dumps(head, indent=1)[: -len("\n}")]
    lines = [
        f"  {json.dumps(task)}: {json.dumps(curves, separators=(',', ':'))}"
        for task, curves in tasks.items()
    ]
    text += ',\n "tasks": {\n' + ",\n".join(lines) + "\n }\n}\n"
    if json.loads(text) != json.loads(json.dumps(reference)):
        raise AssertionError("the formatted JSON does not round-trip")
    with open(path, "w") as f:
        f.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dreamerv3", required=True, help="dreamerv3 checkout")
    parser.add_argument("--tdmpc2", required=True, help="tdmpc2 checkout")
    parser.add_argument("--out-dir", default=REFERENCE_DIR)
    args = parser.parse_args(argv)
    for checkout, commit in (
        (args.dreamerv3, DREAMERV3_COMMIT),
        (args.tdmpc2, TDMPC2_COMMIT),
    ):
        if git_commit(checkout) != commit:
            raise SystemExit(f"{checkout} is not at {commit}")
    os.makedirs(args.out_dir, exist_ok=True)
    dump(
        dreamerv3_reference(args.dreamerv3), os.path.join(args.out_dir, DREAMERV3_FILE)
    )
    dump(tdmpc2_reference(args.tdmpc2), os.path.join(args.out_dir, TDMPC2_FILE))
    print(f"wrote {DREAMERV3_FILE} and {TDMPC2_FILE} to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
