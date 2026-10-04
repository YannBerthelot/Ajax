"""Paper-protocol runs of DreamerV3 and TD-MPC2 on the playground DMC tasks.

The GPU half of the world-model validation (M9,
``docs/world_models/VALIDATION.md``): each run trains one agent on one task
with its paper's protocol, for the reference's seed count (vmapped), and
writes the metric the published curves report after every chunk;
``paper_report.py`` then judges the curves against the references
(``references/*.json``) with the acceptance criteria of
``wm_acceptance.py``. Runs are resumable (``wm_runs.py``): rerunning the
same command continues an interrupted run from its last saved state.

**DreamerV3** (``dreamerv3-<task>``, the 18 DMC-proprio tasks of Table 11):
the paper-era protocol, Table 2 p.19 and ``2411f7d``'s ``dmc_proprio``
preset (dreamerv3_spec 7.5-7.7; ``29eb964:dreamerv3/configs.yaml:44, 84,
232-236``): ``model_size="12m"``, ``train_ratio=512``, 16 envs,
``action_repeat=2``, 1000-step episodes, 500K env steps = 250K rows (the
agent's unit, reset rows included; ``DESIGN.md`` section 2), every other
hyperparameter at its default; 5 seeds. Metric: the returns of the
training episodes of the stochastic policy (``episode/score``), per chunk
of 10K rows = 20K env steps (625 ticks of the 16 envs).

**TD-MPC2** (``tdmpc2-<task>``, the 22 DMC tasks of ``results/*.csv`` that
playground implements): ``model_size=5`` (single-task, ``5f6fade``
README.md:102, :115; tdmpc2_spec 4.30), one env, ``action_repeat=2``,
1000-step episodes (T = 500), every other hyperparameter at its
``config.yaml`` default (buffer 1M, batch 256, ``seed_steps`` 2500); the
budget of the task's CSV (4M env steps = 2M agent steps; humanoid 14M =
7M); 3 seeds. Metric: 10 evaluation episodes of the planner in
``eval_mode`` every 50K agent steps (``config.yaml:10-11``,
``online_trainer.py:27-48``; tdmpc2_spec 4.21), one chunk each.

Playground tasks are MJX re-implementations of dm_control (physics,
observations and sometimes rewards differ): the comparison with the
published dm_control curves is approximate.

Usage::

    python benchmarks/world_models/paper_protocol.py --list
    python benchmarks/world_models/paper_protocol.py --out runs/ \\
        --only dreamerv3-walker_walk tdmpc2-walker-walk
    python benchmarks/world_models/paper_protocol.py --out runs/  # all runs
    JAX_PLATFORMS=cpu python benchmarks/world_models/paper_protocol.py \\
        --smoke --out /tmp/smoke  # tiny CPU runs: plumbing only

Each run writes ``<out>/<run>/{run.json, curve.jsonl}`` (and
``state.pkl`` until it finishes).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from typing import Any, Optional

from wm_runs import SAVE_EVERY_S, RunSpec, load_reference, run_single_task

DREAMERV3_REFERENCE = "dreamerv3_dmc_proprio"
TDMPC2_REFERENCE = "tdmpc2_dmc"

#: DreamerV3's paper protocol (module docstring).
DREAMERV3_KWARGS: dict[str, Any] = {
    "model_size": "12m",
    "n_envs": 16,
    "train_ratio": 512,
    "action_repeat": 2,
    "episode_length": 1000,
}
DREAMERV3_ROWS = 250_000  # 500K env steps at action repeat 2
DREAMERV3_CHUNK_ROWS = 10_000  # 20K env steps: 625 ticks of the 16 envs
DREAMERV3_SEEDS = (0, 1, 2, 3, 4)

#: TD-MPC2's paper protocol (module docstring).
TDMPC2_KWARGS: dict[str, Any] = {
    "model_size": 5,
    "n_envs": 1,
    "action_repeat": 2,
    "episode_length": 1000,
}
TDMPC2_CHUNK_STEPS = 50_000  # eval_freq, in agent steps
TDMPC2_EVAL_EPISODES = 10
TDMPC2_SEEDS = (0, 1, 2)


def dreamerv3_run_name(task: str) -> str:
    return "dreamerv3-" + task.removeprefix("dmc_")


def tdmpc2_run_name(task: str) -> str:
    return f"tdmpc2-{task}"


def tdmpc2_budget_env_frames(curves: list[dict]) -> int:
    """The env steps of a reference task's longest seed curve."""
    return max(max(seed["x"]) for seed in curves)


def paper_runs() -> dict[str, RunSpec]:
    """Every paper-protocol run, from the reference files' task maps."""
    runs: dict[str, RunSpec] = {}
    dreamer = load_reference(DREAMERV3_REFERENCE)
    for task, env_id in dreamer["playground"].items():
        runs[dreamerv3_run_name(task)] = RunSpec(
            agent="DreamerV3",
            env_id=env_id,
            reference_task=task,
            description=f"DreamerV3 12m on playground {env_id}, 500K env steps",
            kwargs=dict(DREAMERV3_KWARGS),
            budget=DREAMERV3_ROWS,
            chunk=DREAMERV3_CHUNK_ROWS,
            seeds=DREAMERV3_SEEDS,
        )
    tdmpc2 = load_reference(TDMPC2_REFERENCE)
    repeat = TDMPC2_KWARGS["action_repeat"]
    for task, env_id in tdmpc2["playground"].items():
        frames = tdmpc2_budget_env_frames(tdmpc2["tasks"][task])
        runs[tdmpc2_run_name(task)] = RunSpec(
            agent="TDMPC2",
            env_id=env_id,
            reference_task=task,
            description=(
                f"TD-MPC2 5M on playground {env_id}, {frames / 1e6:g}M env steps"
            ),
            kwargs=dict(TDMPC2_KWARGS),
            budget=frames // repeat,
            chunk=TDMPC2_CHUNK_STEPS,
            seeds=TDMPC2_SEEDS,
            num_eval_episodes=TDMPC2_EVAL_EPISODES,
        )
    return runs


#: Tiny models for ``--smoke`` (the learning checks' tiny presets).
DREAMERV3_TINY: dict[str, Any] = {
    "model_size": "1m",
    "units": 16,
    "hidden": 16,
    "deter": 32,
    "classes": 4,
    "stoch": 4,
    "blocks": 4,
    "imag_horizon": 3,
    "batch_size": 4,
    "batch_length": 8,
    "train_ratio": 32,
    "n_envs": 4,
    "action_repeat": 2,
    "episode_length": 40,
}
TDMPC2_TINY: dict[str, Any] = {
    "model_size": 1,
    "n_envs": 1,
    "enc_dim": 32,
    "mlp_dim": 32,
    "latent_dim": 16,
    "batch_size": 16,
    "num_samples": 32,
    "num_elites": 4,
    "num_pi_trajs": 4,
    "iterations": 2,
    "action_repeat": 2,
    "episode_length": 40,
    "seed_steps": 20,
}


def smoke_runs() -> dict[str, RunSpec]:
    """The plumbing runs of ``--smoke``: one task per agent, tiny models,
    20-step episodes, two chunks each (so the second one resumes), 2 seeds."""
    runs = paper_runs()
    return {
        "dreamerv3-cartpole_balance": dataclasses.replace(
            runs["dreamerv3-cartpole_balance"],
            description="smoke: tiny DreamerV3 on CartpoleBalance (plumbing only)",
            kwargs=DREAMERV3_TINY,
            budget=336,  # 84 ticks of 4 envs: 4 episodes of 21 rows per env
            chunk=168,
            seeds=(0, 1),
            smoke=True,
        ),
        "tdmpc2-cartpole-balance": dataclasses.replace(
            runs["tdmpc2-cartpole-balance"],
            description="smoke: tiny TD-MPC2 on CartpoleBalance (plumbing only)",
            kwargs=TDMPC2_TINY,
            budget=80,  # 4 episodes of T = 20
            chunk=40,
            seeds=(0, 1),
            num_eval_episodes=2,
            smoke=True,
        ),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", help="output directory (one subdirectory per run)")
    parser.add_argument("--only", nargs="*", default=None, help="runs to do")
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=None, help="override the seeds"
    )
    parser.add_argument("--list", action="store_true", help="list the runs")
    parser.add_argument("--smoke", action="store_true", help="tiny runs (plumbing)")
    args = parser.parse_args(argv)

    runs = smoke_runs() if args.smoke else paper_runs()
    if args.list:
        for name, spec in runs.items():
            print(f"{name:34s} {spec.description}, seeds {list(spec.seeds)}")
        return 0
    if not args.out:
        parser.error("--out is required")
    names = args.only if args.only else list(runs)
    unknown = [name for name in names if name not in runs]
    if unknown:
        raise SystemExit(f"unknown runs {unknown}; see --list")
    for name in names:
        spec = runs[name]
        if args.seeds:
            spec = dataclasses.replace(spec, seeds=tuple(args.seeds))
        print(f"[{name}] {spec.description}, seeds {list(spec.seeds)}", flush=True)
        run_single_task(
            spec,
            os.path.join(args.out, name),
            save_every_s=0.0 if args.smoke else SAVE_EVERY_S,
            log=lambda message: print(message, flush=True),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
