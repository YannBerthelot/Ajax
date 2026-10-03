"""TD-MPC2 learning checks on CPU (a script, not a test).

``docs/world_models/DESIGN.md`` §0 and §10: before an agent PR merges, CPU
learning checks show that the agent learns; their results are recorded in
the PR and in ``PERFORMANCE_REPORT.md``. They are too long for CI.

Each check trains TD-MPC2 at a small but faithful configuration -- the
``model_size=1`` preset (enc 256, mlp 384, latent 128, 2 Q heads) with every
other hyperparameter at the paper default: the paper planner (512 samples
including 24 policy-prior trajectories, 64 elites, 6 MPPI iterations), batch
256, one env, UTD 1, ``seed_steps = max(1000, 5 T)``, ``gamma`` from ``T`` --
and evaluates the planner in ``eval_mode`` (the reference's protocol) every
``eval_every`` env steps through the agent's own logging. A check passes
when every seed's final evaluation return clears the bar:

* ``pendulum``: gymnax Pendulum-v1 (T = 200, the torque bounds [-2, 2]
  mapped from the agent's [-1, 1]), 6,000 env steps, 10 evaluation
  episodes; bar: final return > -400 (a random policy scores about -1200, a
  swing-up policy about -150).
* ``cartpole_balance``: mujoco_playground CartpoleBalance with
  ``episode_length=1000, action_repeat=2`` (T = 500 agent steps, the paper's
  DMC protocol), 10,000 agent steps (20,000 env frames), 5 evaluation
  episodes; bar: final return > 800 (at most 1000; the paper's curve is at
  about 998 from 100k frames on, tdmpc2_spec 4.27).

Cost on one CPU core-pool (Apple M-series, measured per agent step at
``model_size=1``: ~35 ms per update, ~65 ms per planning decision): about
15 minutes for pendulum and 20 minutes for cartpole_balance per seed on a
quiet machine; seeds are vmapped and cost roughly linearly. ``--seeds 0 1 2``
triples it. On a shared machine (load average 25-40), ``--seeds 0 1`` took
45 minutes for pendulum and 88 minutes for cartpole_balance (results in
``PERFORMANCE_REPORT.md``).

Usage::

    JAX_PLATFORMS=cpu python benchmarks/learning_checks.py
    JAX_PLATFORMS=cpu python benchmarks/learning_checks.py \\
        --tasks pendulum --seeds 0 1 --out benchmarks/learning_checks.jsonl
    JAX_PLATFORMS=cpu python benchmarks/learning_checks.py --smoke  # plumbing only

``--smoke`` runs a tiny model for a few episodes to check the script itself;
it says nothing about learning. The exit code is 1 when a (non-smoke) check
fails.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import time
from typing import Any

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclasses.dataclass(frozen=True)
class Check:
    """One learning check: an env, a budget, an evaluation cadence, a bar."""

    env_id: str
    agent_kwargs: dict  # env plumbing (episode length, action repeat)
    n_timesteps: int  # agent (env) steps, the agent's unit
    eval_every: int  # agent steps; a multiple of T, divides n_timesteps
    eval_episodes: int
    bar: float  # every seed's final evaluation return must exceed it
    description: str


CHECKS: dict[str, Check] = {
    "pendulum": Check(
        env_id="Pendulum-v1",
        agent_kwargs={},
        n_timesteps=6_000,
        eval_every=2_000,
        eval_episodes=10,
        bar=-400.0,
        description="gymnax Pendulum-v1 (T=200, bounds [-2, 2] mapped)",
    ),
    "cartpole_balance": Check(
        env_id="CartpoleBalance",
        agent_kwargs={"episode_length": 1000, "action_repeat": 2},
        n_timesteps=10_000,
        eval_every=5_000,
        eval_episodes=5,
        bar=800.0,
        description="playground CartpoleBalance (action repeat 2, T=500)",
    ),
}

# Faithful configuration: the model_size=1 preset, paper defaults elsewhere.
FAITHFUL: dict[str, Any] = {"model_size": 1}

# --smoke: a tiny model and planner and short episodes / budgets.
_SMOKE_MODEL: dict[str, Any] = {
    "model_size": 1,
    "enc_dim": 32,
    "mlp_dim": 32,
    "latent_dim": 16,
    "batch_size": 16,
    "num_samples": 32,
    "num_elites": 4,
    "num_pi_trajs": 4,
    "iterations": 2,
}
_SMOKE_CHECKS: dict[str, Check] = {
    "pendulum": dataclasses.replace(
        CHECKS["pendulum"],
        agent_kwargs={"seed_steps": 200},  # updates within the short budget
        n_timesteps=800,
        eval_every=400,
        eval_episodes=2,
        description="smoke: gymnax Pendulum-v1, tiny model",
    ),
    "cartpole_balance": dataclasses.replace(
        CHECKS["cartpole_balance"],
        agent_kwargs={"episode_length": 40, "action_repeat": 2, "seed_steps": 20},
        n_timesteps=80,
        eval_every=40,
        eval_episodes=2,
        description="smoke: playground CartpoleBalance, T=20, tiny model",
    ),
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_check(name: str, check: Check, seeds: list[int], smoke: bool) -> dict:
    """Train, read the evaluation curve from the logged metrics, judge it."""
    import jax
    import numpy as np

    from ajax import TDMPC2
    from ajax.logging.wandb_logging import LoggingConfig

    model = _SMOKE_MODEL if smoke else FAITHFUL
    agent = TDMPC2(check.env_id, n_envs=1, **check.agent_kwargs, **model)
    T = agent.agent_episode_length
    if check.eval_every % T or check.n_timesteps % check.eval_every:
        raise ValueError(
            f"{name}: eval_every must be a multiple of T = {T} and divide"
            " n_timesteps, so that the last tick evaluates"
        )
    logging_config = LoggingConfig(
        config={"learning_check": name},
        project_name="ajax-learning-checks",
        run_name=f"tdmpc2_{name}",
        log_frequency=check.eval_every,
        use_wandb=False,
        use_tensorboard=False,
    )
    t0 = time.perf_counter()
    state, metrics = agent.train(
        seed=seeds,
        n_timesteps=check.n_timesteps,
        num_episode_test=check.eval_episodes,
        logging_config=logging_config,
    )
    jax.block_until_ready(state)
    wall = time.perf_counter() - t0

    returns = np.asarray(metrics["Eval/episodic mean reward"])  # [seeds, ticks]
    evaluated = np.isfinite(returns[0])
    steps = np.asarray(metrics["timestep"])[0, evaluated].tolist()
    curves = returns[:, evaluated]
    final = curves[:, -1].tolist()
    passed = bool(np.all(curves[:, -1] > check.bar))
    return {
        "check": name,
        "description": check.description,
        "smoke": smoke,
        "seeds": seeds,
        "n_timesteps": check.n_timesteps,
        "env_frames": check.n_timesteps * agent.env_args.action_repeat,
        "eval_steps": steps,
        "eval_returns": curves.tolist(),
        "final_returns": final,
        "bar": check.bar,
        "passed": passed,
        "train_episodic_return": np.asarray(
            state.collector_state.episodic_mean_return
        ).tolist(),
        "n_updates": np.asarray(state.n_updates).tolist(),
        "T": T,
        "gamma": agent.gamma,
        "seed_steps": agent.seed_steps,
        "config": {**model, **check.agent_kwargs},
        "wall_s": round(wall, 1),
        "git_sha": _git_sha(),
        "jax_backend": jax.default_backend(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--tasks", nargs="*", default=list(CHECKS), choices=list(CHECKS)
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--out", default=None, help="append JSON lines here")
    parser.add_argument(
        "--smoke", action="store_true", help="tiny run to check the script only"
    )
    args = parser.parse_args()

    checks = _SMOKE_CHECKS if args.smoke else CHECKS
    results = []
    for name in args.tasks:
        check = checks[name]
        print(
            f"[{name}] {check.description}: {check.n_timesteps} agent steps,"
            f" seeds {args.seeds}",
            flush=True,
        )
        result = run_check(name, check, args.seeds, args.smoke)
        results.append(result)
        if args.smoke:
            verdict = "smoke run, no verdict"
        else:
            verdict = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{name}] eval at steps {result['eval_steps']}:"
            f" {result['eval_returns']}\n"
            f"[{name}] final {result['final_returns']} vs bar > {check.bar}:"
            f" {verdict}  ({result['wall_s']} s)",
            flush=True,
        )
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(result) + "\n")
    if args.smoke:
        return 0
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
