"""Learning checks: does an agent learn its task on CPU, at a small preset?

The world-model agents' learning gates (``docs/world_models/DESIGN.md``
sections 0 and 10): run outside CI, before merging an agent's PR, with the
results recorded in the PR and in ``PERFORMANCE_REPORT.md``. They are too
long for CI.

One registry serves every agent: :data:`CHECKS` maps ``<agent>-<task>`` to a
:class:`Check` (the agent class exported by ``ajax``, its constructor
arguments, a budget in the agent's own unit, a logging and evaluation
cadence, a bar). A check trains the agent once per seed (one vmapped run),
reads the curve of its metric from the logged metrics and passes when every
seed's last logged value exceeds the bar. Agents add their checks as
entries; the CLI is shared.

Usage::

    JAX_PLATFORMS=cpu python benchmarks/learning_checks.py --list
    JAX_PLATFORMS=cpu python benchmarks/learning_checks.py \\
        --only dreamerv3-cartpole --seeds 0 1 --out results.jsonl
    JAX_PLATFORMS=cpu python benchmarks/learning_checks.py --smoke  # plumbing

``--smoke`` runs each check's tiny variant (:data:`SMOKE`) to check the
script itself; it says nothing about learning and gives no verdict. Each
check appends one JSON line (curves, finals, bar, verdict, wall time, git
sha) to ``--out``; the exit code is 1 when a (non-smoke) check fails.

**The metric** is ``Eval/episodic mean reward`` at the last log: the mean
return of ``num_episode_test`` episodes of the final policy from fresh
resets (for DreamerV3, sampled actions from a zero carry: the reference has
no deterministic mode, dreamerv3_spec 7.3). The training-episode rolling
mean (``Train/episodic mean reward``, the reference's score) is reported
next to it but not judged: it averages each env's last 10 episodes, so on
these budgets it still holds the early episodes of the run (16 envs share
20 000 rows: 1 250 rows per env, two to six episodes) and lags the policy.

**DreamerV3** (paper-era recipe, every hyperparameter at its default but
the model size): the ``1m`` preset (``d = 64``: deter 512, 4 classes), 16
envs, train ratio 512, batches of 16 x 64, imagination 15, 20 000 rows (the
agent's unit: one per env per vector step, reset rows included), so about
9 500 updates: about 70 minutes per check and seed on a shared 14-core
Apple CPU at load 25-30 (two checks running side by side).

* ``dreamerv3-cartpole``: gymnax CartPole-v1 (discrete, terminating,
  returns at most 500); bar: > 400.
* ``dreamerv3-pendulum``: gymnax Pendulum-v1 (continuous, torque bounds
  [-2, 2] mapped from [-1, 1], 200-step episodes; a random policy scores
  about -1200, a swung-up, balanced pendulum about -150); bar: > -400.

**TD-MPC2** (paper-era recipe, every hyperparameter at its default but the
model size): the ``model_size=1`` preset (enc 256, mlp 384, latent 128, 2 Q
heads), the paper planner (512 samples including 24 policy-prior
trajectories, 64 elites, 6 MPPI iterations), batch 256, one env, UTD 1,
``seed_steps = max(1000, 5 T)``, ``gamma`` from ``T``; the planner is
evaluated in ``eval_mode`` (the reference's protocol). The budget is in env
(agent) steps; ``log_frequency`` must be a multiple of the episode length
``T`` so that evaluations fall on episode boundaries. Measured per agent
step at ``model_size=1``: ~35 ms per update, ~65 ms per planning decision,
i.e. about 15 (pendulum) and 20 (cartpole-balance) minutes per seed on a
quiet machine; ``--seeds 0 1`` took 45 and 88 minutes at load 25-40.

* ``tdmpc2-pendulum``: gymnax Pendulum-v1 (T = 200, bounds [-2, 2] mapped),
  6 000 env steps, 10 evaluation episodes; bar: > -400.
* ``tdmpc2-cartpole-balance``: mujoco_playground CartpoleBalance with
  ``episode_length=1000, action_repeat=2`` (T = 500 agent steps, the
  paper's DMC protocol), 10 000 agent steps (20 000 frames), 5 evaluation
  episodes; bar: > 800 (at most 1000; the paper's curve is at about 998
  from 100k frames on, tdmpc2_spec 4.27).

These bars are ours, not the papers' (the papers report no point at these
budgets): a check is a sanity check that the agent learns its task, not a
reproduction of a published number.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import time
from typing import Any

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclasses.dataclass(frozen=True)
class Check:
    """One learning check.

    Attributes:
        agent: the agent class's name, exported by ``ajax``.
        env_id: the task.
        description: one line for the logs.
        kwargs: constructor arguments besides ``env_id``.
        n_timesteps: training budget, in the agent's own unit (DreamerV3:
            rows; TD-MPC2: env steps).
        log_frequency: logging and evaluation cadence, in the same unit; it
            divides ``n_timesteps``, so that the last tick evaluates.
        num_episode_test: evaluation episodes per log.
        bar: every seed's last logged ``metric`` must exceed it.
        metric: the logged metric compared with the bar.
    """

    agent: str
    env_id: str
    description: str
    kwargs: dict = dataclasses.field(default_factory=dict)
    n_timesteps: int = 20_000
    log_frequency: int = 2_000
    num_episode_test: int = 10
    bar: float = 0.0
    metric: str = "Eval/episodic mean reward"


_DREAMERV3: dict[str, Any] = {"model_size": "1m"}
_TDMPC2: dict[str, Any] = {"model_size": 1, "n_envs": 1}

CHECKS: dict[str, Check] = {
    "dreamerv3-cartpole": Check(
        agent="DreamerV3",
        env_id="CartPole-v1",
        description="DreamerV3 1m on gymnax CartPole-v1 (discrete, terminating)",
        kwargs=_DREAMERV3,
        n_timesteps=20_000,
        log_frequency=2_000,
        bar=400.0,
    ),
    "dreamerv3-pendulum": Check(
        agent="DreamerV3",
        env_id="Pendulum-v1",
        description="DreamerV3 1m on gymnax Pendulum-v1 (bounds [-2, 2] mapped)",
        kwargs=_DREAMERV3,
        n_timesteps=20_000,
        log_frequency=2_000,
        bar=-400.0,
    ),
    "tdmpc2-pendulum": Check(
        agent="TDMPC2",
        env_id="Pendulum-v1",
        description="TD-MPC2 size 1 on gymnax Pendulum-v1 (T=200, bounds mapped)",
        kwargs=_TDMPC2,
        n_timesteps=6_000,
        log_frequency=2_000,
        num_episode_test=10,
        bar=-400.0,
    ),
    "tdmpc2-cartpole-balance": Check(
        agent="TDMPC2",
        env_id="CartpoleBalance",
        description=(
            "TD-MPC2 size 1 on playground CartpoleBalance (action repeat 2, T=500)"
        ),
        kwargs={**_TDMPC2, "episode_length": 1000, "action_repeat": 2},
        n_timesteps=10_000,
        log_frequency=5_000,
        num_episode_test=5,
        bar=800.0,
    ),
}

#: Tiny variants for ``--smoke``: the same code path in about a minute.


def _smoke_description(name: str) -> str:
    return f"smoke run of {name} (tiny model, short budget): plumbing only"


_DREAMERV3_TINY: dict[str, Any] = {
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
}
_TDMPC2_TINY: dict[str, Any] = {
    **_TDMPC2,
    "enc_dim": 32,
    "mlp_dim": 32,
    "latent_dim": 16,
    "batch_size": 16,
    "num_samples": 32,
    "num_elites": 4,
    "num_pi_trajs": 4,
    "iterations": 2,
}
SMOKE: dict[str, Check] = {
    **{
        name: dataclasses.replace(
            CHECKS[name],
            description=_smoke_description(name),
            kwargs=_DREAMERV3_TINY,
            n_timesteps=400,
            log_frequency=200,
            num_episode_test=2,
        )
        for name in ("dreamerv3-cartpole", "dreamerv3-pendulum")
    },
    "tdmpc2-pendulum": dataclasses.replace(
        CHECKS["tdmpc2-pendulum"],
        description=_smoke_description("tdmpc2-pendulum"),
        kwargs={**_TDMPC2_TINY, "seed_steps": 200},
        n_timesteps=800,
        log_frequency=400,
        num_episode_test=2,
    ),
    "tdmpc2-cartpole-balance": dataclasses.replace(
        CHECKS["tdmpc2-cartpole-balance"],
        description=_smoke_description("tdmpc2-cartpole-balance"),
        kwargs={
            **_TDMPC2_TINY,
            "episode_length": 40,
            "action_repeat": 2,
            "seed_steps": 20,
        },
        n_timesteps=80,
        log_frequency=40,
        num_episode_test=2,
    ),
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _logged(metrics: dict, key: str, ticks: np.ndarray) -> np.ndarray:
    """``[seeds, logs]``: ``key`` at the logging ticks."""
    return np.asarray(metrics[key], np.float64)[:, ticks]


def run_check(name: str, check: Check, seeds: list[int], smoke: bool) -> dict:
    """Train ``check`` for every seed (one vmapped run) and judge it."""
    import jax

    import ajax
    from ajax.logging.wandb_logging import LoggingConfig

    agent = getattr(ajax, check.agent)(env_id=check.env_id, **check.kwargs)
    n_envs = agent.env_args.n_envs
    if check.log_frequency % n_envs or check.n_timesteps % check.log_frequency:
        raise ValueError(
            f"{name}: log_frequency must be a multiple of n_envs = {n_envs} and"
            " divide n_timesteps, so that the last tick evaluates"
        )
    # TD-MPC2 counts env steps on fixed-length episodes of T agent steps; its
    # evaluations fall on episode boundaries only for multiples of T.
    episode_length = getattr(agent, "agent_episode_length", None)
    if episode_length is not None and check.log_frequency % (episode_length * n_envs):
        raise ValueError(
            f"{name}: log_frequency must be a multiple of T * n_envs ="
            f" {episode_length * n_envs}"
        )
    # Logging on, writing nowhere: the logged metrics come back as the
    # per-tick output of train() (NaN on the ticks that do not log).
    logging_config = LoggingConfig(
        config={"learning_check": name},
        project_name="ajax-learning-checks",
        run_name=name,
        log_frequency=check.log_frequency,
        use_wandb=False,
        use_tensorboard=False,
    )
    start = time.perf_counter()
    state, metrics = agent.train(
        seed=seeds,
        n_timesteps=check.n_timesteps,
        num_episode_test=check.num_episode_test,
        logging_config=logging_config,
    )
    jax.block_until_ready(state)
    wall = time.perf_counter() - start

    ticks = np.flatnonzero(np.isfinite(np.asarray(metrics[check.metric])[0]))
    curve = _logged(metrics, check.metric, ticks)
    train_curve = _logged(metrics, "Train/episodic mean reward", ticks)
    final = curve[:, -1]
    return {
        "check": name,
        "description": check.description,
        "smoke": smoke,
        "agent": check.agent,
        "env": check.env_id,
        "kwargs": check.kwargs,
        "seeds": seeds,
        "n_timesteps": check.n_timesteps,
        "env_frames": check.n_timesteps * agent.env_args.action_repeat,
        "metric": check.metric,
        "timesteps": np.asarray(metrics["timestep"])[0, ticks].tolist(),
        "curve": curve.tolist(),
        "train_curve": train_curve.tolist(),
        "final": final.tolist(),
        "bar": check.bar,
        "passed": bool(np.all(final > check.bar)),
        "n_updates": np.asarray(state.n_updates).tolist(),
        "resolved": {
            key: getattr(agent, key)
            for key in ("agent_episode_length", "gamma", "seed_steps")
            if hasattr(agent, key)
        },
        "wall_s": round(wall, 1),
        "git_sha": _git_sha(),
        "jax_backend": jax.default_backend(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="*", default=None, help="checks to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--out", default=None, help="append JSON lines here")
    parser.add_argument("--list", action="store_true", help="list the checks")
    parser.add_argument(
        "--smoke", action="store_true", help="tiny runs to check the script only"
    )
    args = parser.parse_args()

    checks = SMOKE if args.smoke else CHECKS
    if args.list:
        for name, check in checks.items():
            print(
                f"{name:24s} {check.description}: {check.n_timesteps} steps,"
                f" {check.metric} > {check.bar}"
            )
        return 0

    names = args.only if args.only else list(checks)
    unknown = [name for name in names if name not in checks]
    if unknown:
        raise SystemExit(f"unknown checks {unknown}; see --list")
    passed = True
    for name in names:
        check = checks[name]
        print(f"[{name}] {check.description}, seeds {args.seeds}", flush=True)
        result = run_check(name, check, args.seeds, args.smoke)
        verdict = (
            "smoke run, no verdict"
            if args.smoke
            else ("PASS" if result["passed"] else "FAIL")
        )
        print(
            f"[{name}] {check.metric} at {result['timesteps']}: {result['curve']}\n"
            f"[{name}] Train/episodic mean reward: {result['train_curve']}\n"
            f"[{name}] final {result['final']} vs bar > {check.bar}: {verdict}"
            f" ({result['wall_s']} s, {result['n_updates']} updates)",
            flush=True,
        )
        passed &= result["passed"]
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(result) + "\n")
    return 0 if args.smoke or passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
