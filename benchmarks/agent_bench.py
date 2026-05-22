"""Per-agent speed benchmark — the no-regression guardrail.

Times a fixed, standardized train run for every Ajax agent and splits
compile+warmup wall-clock from steady-state throughput (steps/s). Used to
capture a baseline before the agent-architecture refactor and to check,
after it, that no agent regressed beyond tolerance.

Modes
-----
Capture::

    python benchmarks/agent_bench.py --tag baseline \
        --out benchmarks/agent_baseline.jsonl

Compare (exit code 1 if any agent regressed past --tol)::

    python benchmarks/agent_bench.py --compare \
        benchmarks/agent_baseline.jsonl benchmarks/agent_rework.jsonl

One process, one agent at a time; the JIT cache is warmed once per agent
then a fresh agent is timed (buffer/seed state clean, cache hot) — the
same compile/measured split as ``perf_bench.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time

# agent name -> (env_id, extra constructor kwargs). Discrete-action
# agents train on CartPole-v1, continuous ones on Pendulum-v1. Only
# env_id / n_envs / n_timesteps are pinned; every other hyperparameter
# uses the agent default, so the benchmark tracks the agent's own code
# path rather than a tuned config.
AGENTS: dict[str, tuple[str, dict]] = {
    "SAC": ("Pendulum-v1", {}),
    "PPO": ("CartPole-v1", {}),
    "DQN": ("CartPole-v1", {}),
    "PQN": ("CartPole-v1", {}),
    "TD3": ("Pendulum-v1", {}),
    "REDQ": ("Pendulum-v1", {}),
    "APO": ("Pendulum-v1", {}),
    "ASAC": ("Pendulum-v1", {}),
    "AVG": ("Pendulum-v1", {}),
    "SafeSAC": ("Pendulum-v1", {}),
}

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _agent_class(name: str):
    """Resolve an agent class — exported on `ajax`, else from its module."""
    import ajax

    cls = getattr(ajax, name, None)
    if cls is None:
        import importlib

        mod = importlib.import_module(f"ajax.agents.{name}.{name}")
        cls = getattr(mod, name)
    return cls


def _build(name: str, env_id: str, kw: dict, n_envs: int):
    return _agent_class(name)(env_id=env_id, n_envs=n_envs, **kw)


def _time_agent(
    name: str,
    env_id: str,
    kw: dict,
    *,
    n_envs: int,
    warmup: int,
    timesteps: int,
    trials: int,
) -> list[dict]:
    """Warm the JIT cache, then time `trials` measured runs of `timesteps`."""
    import jax

    rows: list[dict] = []
    # Warmup: compile for this exact (shapes, n_seeds) signature.
    t0 = time.perf_counter()
    agent = _build(name, env_id, kw, n_envs)
    out = agent.train(seed=0, n_timesteps=warmup)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    del agent, out
    gc.collect()

    for trial in range(trials):
        agent = _build(name, env_id, kw, n_envs)
        t0 = time.perf_counter()
        out = agent.train(seed=1 + trial, n_timesteps=timesteps)
        jax.block_until_ready(out)
        measured_s = time.perf_counter() - t0
        rows.append(
            {
                "agent": name,
                "env": env_id,
                "trial": trial,
                "n_envs": n_envs,
                "timesteps": timesteps,
                "compile_s": round(compile_s, 3),
                "measured_s": round(measured_s, 3),
                "steps_per_s": round(timesteps / measured_s, 1)
                if measured_s > 0
                else 0.0,
            }
        )
        del agent, out
        gc.collect()
    return rows


def capture(
    tag: str,
    out_path: str,
    *,
    n_envs: int,
    warmup: int,
    timesteps: int,
    trials: int,
    only: list[str] | None,
) -> None:
    import jax

    sha = _git_sha()
    backend = jax.default_backend()
    names = only if only else list(AGENTS)
    all_rows: list[dict] = []
    for name in names:
        if name not in AGENTS:
            print(f"  {name}: unknown agent, skipped", flush=True)
            continue
        env_id, kw = AGENTS[name]
        try:
            rows = _time_agent(
                name,
                env_id,
                kw,
                n_envs=n_envs,
                warmup=warmup,
                timesteps=timesteps,
                trials=trials,
            )
        except Exception as exc:  # - one agent must not abort all
            print(
                f"  {name}: FAILED to benchmark -- {type(exc).__name__}: "
                f"{str(exc)[:160]}",
                flush=True,
            )
            all_rows.append(
                {
                    "agent": name,
                    "env": env_id,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
            )
            continue
        med = sorted(r["steps_per_s"] for r in rows)[len(rows) // 2]
        print(
            f"  {name:9s} compile={rows[0]['compile_s']:7.2f}s  "
            f"steps/s(median)={med:9.1f}",
            flush=True,
        )
        for r in rows:
            r.update(tag=tag, git_sha=sha, jax_backend=backend)
        all_rows.extend(rows)

    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(
        f"baseline -> {out_path}  ({len(all_rows)} rows, "
        f"git={sha}, backend={backend})",
        flush=True,
    )


def _median_steps(path: str) -> dict[str, float]:
    per: dict[str, list[float]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "steps_per_s" in r:
                per.setdefault(r["agent"], []).append(r["steps_per_s"])
    return {a: sorted(v)[len(v) // 2] for a, v in per.items() if v}


def compare(baseline_path: str, new_path: str, tol: float) -> int:
    base = _median_steps(baseline_path)
    new = _median_steps(new_path)
    print(f"{'agent':10s} {'baseline':>12s} {'new':>12s} {'delta':>9s}")
    regressed = []
    for agent in sorted(base):
        if agent not in new:
            print(f"{agent:10s} {base[agent]:12.1f} {'MISSING':>12s}")
            continue
        b, n = base[agent], new[agent]
        delta = (n - b) / b if b > 0 else 0.0
        flag = ""
        if delta < -tol:
            flag = "  <-- REGRESSION"
            regressed.append((agent, delta))
        print(f"{agent:10s} {b:12.1f} {n:12.1f} {delta:+8.1%}{flag}")
    if regressed:
        print(
            f"\n{len(regressed)} agent(s) regressed beyond {tol:.0%}: "
            + ", ".join(f"{a} ({d:+.1%})" for a, d in regressed)
        )
        return 1
    print(f"\nno regression beyond {tol:.0%}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "agent_baseline.jsonl"
        ),
    )
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--timesteps", type=int, default=8000)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument(
        "--only", nargs="*", default=None, help="benchmark only these agents"
    )
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "NEW"),
        default=None,
        help="compare two result files and exit",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.10,
        help="regression tolerance for --compare (default 10%%)",
    )
    args = ap.parse_args()

    if args.compare:
        raise SystemExit(compare(args.compare[0], args.compare[1], args.tol))
    capture(
        args.tag,
        args.out,
        n_envs=args.n_envs,
        warmup=args.warmup,
        timesteps=args.timesteps,
        trials=args.trials,
        only=args.only,
    )


if __name__ == "__main__":
    main()
