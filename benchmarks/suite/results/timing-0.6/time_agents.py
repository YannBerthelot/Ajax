"""Time the benchmark's agents for 0.6.0: environment steps per second with
10 seeds trained together, on four CPU cores and on one GPU.

usage, from the repository root, on the machine to time:
    uv run python benchmarks/suite/results/timing-0.6/time_agents.py cpu SAC 4096

The method of results/timing-gpu and PHASE0.md:
- each agent is built with the benchmark's settings (agent_suite.py), except
  that the off-policy agents start their updates by step 256 (learning_starts,
  TD-MPC2's seed_steps), so that the timed call runs updates throughout;
- a first call trains the seeds for the segment's steps (more only when the
  updates start later) and compiles the run;
- a second call, in the same process, resumes for the segment and is timed,
  the seconds JAX spends compiling taken out: a resumed call compiles a
  program of its own, its start being part of it;
- no evaluations run (logging_config=None).
Segments are picked per agent and machine so that the timed call takes one to
three minutes. Each record goes to cpu.jsonl or gpu.jsonl beside this file.
"""

import argparse
import inspect
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Optional

HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[1]))
import agent_suite  # noqa: E402
from agent_bench import compile_spans, covered_seconds  # noqa: E402

N_SEEDS = 10
#: The step by which the off-policy agents start updating in these runs.
UPDATES_BY = 256
#: The settings the record keeps from the process's environment.
ENVIRONMENT = (
    "JAX_PLATFORMS",
    "CUDA_VISIBLE_DEVICES",
    "AJAX_NO_COMPILE_CACHE",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_FLAGS",
)


def build_for_timing(label: str) -> tuple[Any, dict]:
    """The benchmark's agent and its options: the benchmark's, with
    learning_starts or seed_steps brought down to UPDATES_BY."""
    import ajax

    benchmark = agent_suite.BENCHMARKS[label]
    agent_class = getattr(ajax, label.split("-")[0])
    parameters = inspect.signature(agent_class).parameters
    options = dict(benchmark.options)
    for name in ("learning_starts", "seed_steps"):
        if name in parameters:
            default = parameters[name].default
            if default is None or default > UPDATES_BY:
                options[name] = UPDATES_BY
    return agent_class(env_id=benchmark.env_id, **options), options


def first_update_step(agent: Any) -> int:
    """The step the agent first updates at: its learning_starts or seed
    phase, DreamerV3's replay gate (batch_length x n_envs + batch_size, as
    its TrainRatio derives it), else 0 for the agents that update after
    each rollout."""
    if hasattr(agent, "learning_starts"):
        return agent.learning_starts
    if hasattr(agent, "resolved_seed_steps"):
        return agent.resolved_seed_steps
    if hasattr(agent, "train_ratio"):
        return agent.batch_length * agent.env_args.n_envs + agent.batch_size
    return 0


def gpu_processes() -> Optional[list[str]]:
    """The compute processes on the visible GPU, as nvidia-smi lists them,
    to show whether anything shared the card; None on a CPU."""
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not device:
        return None
    query = ["nvidia-smi", "-i", device, "--query-compute-apps=pid,used_memory"]
    result = subprocess.run(
        [*query, "--format=csv,noheader"], capture_output=True, text=True
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def time_agent(label: str, machine: str, steps: int) -> dict:
    benchmark = agent_suite.BENCHMARKS[label]
    agent, options = build_for_timing(label)
    unit = agent_suite.iteration_steps(agent)
    timed_steps = agent_suite.round_up(steps, unit)
    first_steps = agent_suite.round_up(
        max(timed_steps, first_update_step(agent) + unit), unit
    )
    seeds = list(range(N_SEEDS))
    run = {"total_timesteps": benchmark.budget, "logging_config": None}
    spans = compile_spans()
    spans.clear()
    began = time.perf_counter()
    state, _ = agent.train(seed=seeds, n_timesteps=first_steps, **run)
    first_s = time.perf_counter() - began
    compile_s = covered_seconds(spans)
    spans.clear()
    processes_before = gpu_processes()
    began = time.perf_counter()
    agent.train(seed=seeds, n_timesteps=timed_steps, initial_state=state, **run)
    timed_s = time.perf_counter() - began
    resume_compile_s = covered_seconds(spans)
    exec_s = timed_s - resume_compile_s
    return {
        "agent": label,
        "env": benchmark.env_id,
        "machine": machine,
        "n_seeds": N_SEEDS,
        "options": options,
        "first_update_step": first_update_step(agent),
        "first_steps": first_steps,
        "timed_steps": timed_steps,
        "compile_s": round(compile_s, 1),
        "first_s": round(first_s, 1),
        "resume_compile_s": round(resume_compile_s, 1),
        "exec_s": round(exec_s, 2),
        # A seed's steps count each of its envs; the run's count every seed.
        "run_steps_per_s": round(timed_steps / exec_s, 1),
        "env_steps_per_s": round(N_SEEDS * timed_steps / exec_s, 1),
        "peak_memory_gb": agent_suite.peak_memory_gb(),
        "device_peak_memory_gb": agent_suite.device_peak_memory_gb(),
        "cores": sorted(os.sched_getaffinity(0)),
        "environment": {name: os.environ.get(name, "") for name in ENVIRONMENT},
        "pid": os.getpid(),
        "gpu_processes": {"before": processes_before, "after": gpu_processes()},
        "manifest": agent_suite.manifest(),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("machine", choices=("cpu", "gpu"))
    parser.add_argument("label", choices=sorted(agent_suite.BENCHMARKS))
    parser.add_argument("steps", type=int, help="the segment's steps per seed")
    args = parser.parse_args(argv)
    record = time_agent(args.label, args.machine, args.steps)
    agent_suite.append(HERE / f"{args.machine}.jsonl", record)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
