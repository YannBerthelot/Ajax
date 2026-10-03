"""Per-process micro-benchmark for SAC training in Ajax.

Runs a fixed-shape SAC training scenario, splitting compile vs steady-state
wall-clock and recording peak GPU memory. One process per measurement so
memory stats and JIT cache are fresh.

Usage:
    python benchmarks/perf_bench.py \
        --tag baseline \
        --scenario pure_sac \
        --timesteps 4000 \
        --warmup-timesteps 200 \
        --trials 3 \
        --out benchmarks/results.jsonl

Scenarios:
    pure_sac          Pendulum-v1, no obs norm, no expert. Sensitive to
                      patch #2 (gated q_preds_for_var).
    obs_norm_sac      Pendulum-v1, normalize_obs_running=True. Sensitive
                      to patch #1 (agent obs-norm shape).

The script prints one JSON line per trial to stdout AND appends it to
``--out`` (jsonl). A separate aggregator (``perf_compare.py``) reads the
log and produces before/after deltas.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class TrialResult:
    tag: str
    scenario: str
    trial: int
    timesteps: int
    warmup_timesteps: int
    n_envs: int
    n_seeds: int
    seed: int
    compile_plus_warmup_s: float
    measured_s: float
    steps_per_s: float
    peak_bytes_in_use: int
    bytes_in_use_after: int
    git_sha: str
    git_dirty: bool
    jax_backend: str
    device: str
    python: str


def _git_state() -> Dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                text=True,
            ).strip()
        )
    except Exception:
        sha, dirty = "unknown", False
    return {"git_sha": sha, "git_dirty": dirty}


def _build_agent(scenario: str, n_envs: int, learning_starts: int):
    from ajax.agents.SAC.SAC import SAC

    common = {
        "env_id": "Pendulum-v1",
        "n_envs": n_envs,
        "learning_starts": learning_starts,
        "actor_architecture": ("64", "relu", "64", "relu"),
        "critic_architecture": ("64", "relu", "64", "relu"),
        "batch_size": 256,
        "buffer_size": 10_000,
        "num_critics": 2,
    }
    if scenario == "pure_sac":
        return SAC(**common)
    if scenario == "obs_norm_sac":
        return SAC(normalize_obs_running=True, **common)
    if scenario == "stress_sac":
        # Amplifies patch effects: many critics (#2), bigger obs broadcast
        # via larger n_envs (#1), larger nets so forward passes dominate.
        return SAC(
            env_id="Pendulum-v1",
            n_envs=max(n_envs, 64),
            learning_starts=learning_starts,
            actor_architecture=("256", "relu", "256", "relu"),
            critic_architecture=("256", "relu", "256", "relu"),
            batch_size=512,
            buffer_size=20_000,
            num_critics=10,
            normalize_obs_running=True,
        )
    if scenario == "stress_sac_utd":
        # Update-to-data ratio > 1 (REDQ-like). Amplifies patch #4
        # (no-ys inner critic scan) by making the scan trip-count large.
        return SAC(
            env_id="Pendulum-v1",
            n_envs=max(n_envs, 64),
            learning_starts=learning_starts,
            actor_architecture=("256", "relu", "256", "relu"),
            critic_architecture=("256", "relu", "256", "relu"),
            batch_size=512,
            buffer_size=20_000,
            num_critics=10,
            num_critic_updates=10,
            normalize_obs_running=True,
        )
    if scenario == "p3dcircle_sac":
        # Plane3DCircle from target_gym: 17-dim obs, 3-dim action.
        # Heavier per-seed footprint than Pendulum, useful for testing
        # where seed scaling saturates.
        from target_gym import Plane3DCircle

        return SAC(
            env_id=Plane3DCircle(),
            n_envs=n_envs,
            learning_starts=learning_starts,
            actor_architecture=("256", "relu", "256", "relu"),
            critic_architecture=("256", "relu", "256", "relu"),
            batch_size=256,
            buffer_size=20_000,
            num_critics=2,
        )
    if scenario == "p3dcircle_edge_qa_hpo":
        # Mirrors AjaxExperiments' P1 sac_quality_aware HPO config more
        # closely: arch_width=512, num_critics=4. This is the
        # configuration where the 20-vs-50 seed regression appears in
        # real HPO data.
        import jax.numpy as _jnp
        from target_gym import Plane3DCircle

        action_dim, state_dim = 3, 3

        class _ZeroExpert:
            _zero_state = _jnp.zeros((state_dim,))
            _params = None

            def init_state(self, num_envs: int):
                return _jnp.zeros((num_envs, state_dim))

            def __call__(self, *args):
                if len(args) == 2:
                    state, obs = args
                    return _jnp.zeros(obs.shape[:-1] + (action_dim,)), state
                (obs,) = args
                return _jnp.zeros(obs.shape[:-1] + (action_dim,))

            def __hash__(self):
                return id(self)

            def __eq__(self, other):
                return self is other

        return SAC(
            env_id=Plane3DCircle(),
            n_envs=n_envs,
            learning_starts=learning_starts,
            actor_architecture=("512", "relu", "512", "relu"),
            critic_architecture=("512", "relu", "512", "relu"),
            batch_size=256,
            buffer_size=20_000,
            num_critics=4,
            expert_policy=_ZeroExpert(),
            use_expert_guidance=False,
            use_expert_guided_exploration=True,
            exploration_lcb=True,
            lcb_beta_init=1.0,
            lcb_temperature=1.0,
            augment_obs_with_expert_state=True,
            normalize_obs_running=True,
            num_critic_updates=1,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
        )
    if scenario == "p3dcircle_edge_qa":
        # Mirrors AjaxExperiments' `sac_quality_aware` config on
        # Plane3DCircle (LCB-gated SAC + obs+state aug + obs norm), with
        # a stub zero-action expert so the LCB / expert paths trace and
        # compile but the expert returns trivial actions. Used to bench
        # whether seed scaling differs from plain SAC on the same env.
        import jax.numpy as _jnp
        from target_gym import Plane3DCircle

        action_dim = 3
        state_dim = 3  # mimics PID integrator; only shape matters for bench

        class _ZeroExpert:
            _zero_state = _jnp.zeros((state_dim,))
            _params = None

            def init_state(self, num_envs: int):
                return _jnp.zeros((num_envs, state_dim))

            def __call__(self, *args):
                if len(args) == 2:
                    state, obs = args
                    return _jnp.zeros(obs.shape[:-1] + (action_dim,)), state
                (obs,) = args
                return _jnp.zeros(obs.shape[:-1] + (action_dim,))

            def __hash__(self):  # SAC hashes the expert into static args
                return id(self)

            def __eq__(self, other):
                return self is other

        expert = _ZeroExpert()
        return SAC(
            env_id=Plane3DCircle(),
            n_envs=n_envs,
            learning_starts=learning_starts,
            actor_architecture=("256", "relu", "256", "relu"),
            critic_architecture=("256", "relu", "256", "relu"),
            batch_size=256,
            buffer_size=20_000,
            num_critics=2,
            expert_policy=expert,
            use_expert_guidance=False,
            use_expert_guided_exploration=True,
            exploration_lcb=True,
            lcb_beta_init=1.0,
            lcb_temperature=1.0,
            augment_obs_with_expert_state=True,
            normalize_obs_running=True,
            num_critic_updates=1,
            expert_buffer_n_steps=0,
            expert_mix_fraction=0.0,
        )
    raise ValueError(f"unknown scenario: {scenario}")


def _seed_arg(seed_base: int, n_seeds: int):
    """Build the train()'s seed argument: int when n_seeds==1, else list.

    The list form triggers ``jax.vmap(set_key_and_train, in_axes=0)`` in
    [base.py:168](src/ajax/agents/base.py#L168), running all seeds in
    parallel under one compiled program.
    """
    if n_seeds == 1:
        return seed_base
    return [seed_base + 100 * i for i in range(n_seeds)]


def _run_one_trial(
    *,
    tag: str,
    scenario: str,
    trial: int,
    timesteps: int,
    warmup_timesteps: int,
    n_envs: int,
    n_seeds: int,
    seed: int,
) -> TrialResult:
    import jax

    device = jax.devices()[0]
    backend = jax.default_backend()
    has_mem_stats = hasattr(device, "memory_stats")
    if has_mem_stats:
        try:
            device.clear_memory_stats()
        except Exception:
            pass

    # Warmup: triggers JIT compile for this exact (shape, n_seeds)
    # signature, plus a few real training steps so the cache is hot.
    agent = _build_agent(
        scenario, n_envs=n_envs, learning_starts=min(50, warmup_timesteps // 2)
    )
    t0 = time.perf_counter()
    agent.train(seed=_seed_arg(seed, n_seeds), n_timesteps=warmup_timesteps)
    jax.block_until_ready(getattr(agent, "_last_state", 0))
    compile_plus_warmup_s = time.perf_counter() - t0

    # Measured run: rebuild the agent so seed / buffer state is clean,
    # but the JIT cache from the warmup carries over (same shapes).
    del agent
    gc.collect()
    if has_mem_stats:
        try:
            device.clear_memory_stats()
        except Exception:
            pass

    agent = _build_agent(
        scenario, n_envs=n_envs, learning_starts=min(50, timesteps // 4)
    )
    t0 = time.perf_counter()
    agent.train(seed=_seed_arg(seed + 1, n_seeds), n_timesteps=timesteps)
    jax.block_until_ready(getattr(agent, "_last_state", 0))
    measured_s = time.perf_counter() - t0

    if has_mem_stats:
        stats = device.memory_stats()
        peak = int(stats.get("peak_bytes_in_use", 0))
        in_use = int(stats.get("bytes_in_use", 0))
    else:
        peak, in_use = 0, 0

    git = _git_state()
    return TrialResult(
        tag=tag,
        scenario=scenario,
        trial=trial,
        timesteps=timesteps,
        warmup_timesteps=warmup_timesteps,
        n_envs=n_envs,
        n_seeds=n_seeds,
        seed=seed,
        compile_plus_warmup_s=compile_plus_warmup_s,
        measured_s=measured_s,
        steps_per_s=timesteps / measured_s if measured_s > 0 else 0.0,
        peak_bytes_in_use=peak,
        bytes_in_use_after=in_use,
        git_sha=git["git_sha"],
        git_dirty=git["git_dirty"],
        jax_backend=backend,
        device=str(device),
        python=platform.python_version(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tag", required=True, help="label for this run (e.g. baseline, patch1)"
    )
    p.add_argument(
        "--scenario",
        default="pure_sac",
        choices=[
            "pure_sac",
            "obs_norm_sac",
            "stress_sac",
            "stress_sac_utd",
            "p3dcircle_sac",
            "p3dcircle_edge_qa",
            "p3dcircle_edge_qa_hpo",
        ],
    )
    p.add_argument("--timesteps", type=int, default=4000)
    p.add_argument("--warmup-timesteps", type=int, default=200)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds to vmap-train in parallel.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results.jsonl"
        ),
    )
    args = p.parse_args()

    results = []
    for trial in range(args.trials):
        r = _run_one_trial(
            tag=args.tag,
            scenario=args.scenario,
            trial=trial,
            timesteps=args.timesteps,
            warmup_timesteps=args.warmup_timesteps,
            n_envs=args.n_envs,
            n_seeds=args.n_seeds,
            seed=args.seed + 1000 * trial,
        )
        line = json.dumps(asdict(r))
        print(line, flush=True)
        results.append(r)

    with open(args.out, "a") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    measured = [r.measured_s for r in results]
    peaks = [r.peak_bytes_in_use for r in results]
    measured_sorted = sorted(measured)
    peaks_sorted = sorted(peaks)
    summary = {
        "tag": args.tag,
        "scenario": args.scenario,
        "trials": args.trials,
        "measured_s_median": measured_sorted[len(measured_sorted) // 2],
        "measured_s_min": measured_sorted[0],
        "peak_bytes_median": peaks_sorted[len(peaks_sorted) // 2],
        "peak_bytes_min": peaks_sorted[0],
    }
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
