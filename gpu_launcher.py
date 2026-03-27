"""
gpu_launcher.py

Distributes experiments across GPUs, launching each job only when a GPU is
truly idle — no compute processes from any user, not just this script.

Uses `nvidia-smi --query-compute-apps` to detect active CUDA processes system-wide
before every launch, so it won't steal a GPU that a colleague is using.

Usage:
    python gpu_launcher.py                             # all GPUs, plane_exps.py
    python gpu_launcher.py --script ablation_study     # run the ablation suite
    python gpu_launcher.py --gpus 0 1                  # restrict to GPUs 0 and 1
    python gpu_launcher.py --dry-run                   # print plan without running
    python gpu_launcher.py --poll 60                   # check every 60s (default: 30)
"""

import argparse
import importlib
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional



@dataclass
class Job:
    exp_index: int
    exp_name: str
    gpu_id: int
    process: Optional[subprocess.Popen] = None


# ---------------------------------------------------------------------------
# GPU availability helpers
# ---------------------------------------------------------------------------


def get_all_gpu_ids() -> List[int]:
    """Return the list of all GPU indices reported by nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi failed — is CUDA available?")
    return [int(line.strip()) for line in result.stdout.strip().splitlines() if line.strip()]


# MiB threshold below which a GPU is considered idle.
# An idle GPU typically shows ~0 MiB; any real workload uses hundreds of MiB.
FREE_MEM_THRESHOLD_MIB = 500


def get_gpu_memory_used() -> Dict[int, int]:
    """
    Return {gpu_id: memory_used_MiB} for all GPUs via nvidia-smi.
    Memory usage is always visible to every user (no privilege needed).
    Returns an empty dict if nvidia-smi fails.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    usage: Dict[int, int] = {}
    if result.returncode != 0:
        return usage
    for line in result.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            gpu_idx = int(parts[0].strip())
            mem_mib = int(parts[1].strip())
            usage[gpu_idx] = mem_mib
        except ValueError:
            continue
    return usage


def is_gpu_free(gpu_id: int, mem_used: Optional[Dict[int, int]] = None) -> bool:
    """
    True if GPU memory usage is below FREE_MEM_THRESHOLD_MIB.
    Pass a pre-fetched mem_used dict to avoid multiple nvidia-smi calls per loop.
    """
    if mem_used is None:
        mem_used = get_gpu_memory_used()
    return mem_used.get(gpu_id, 0) < FREE_MEM_THRESHOLD_MIB


# ---------------------------------------------------------------------------
# Experiment discovery and launching
# ---------------------------------------------------------------------------


def get_experiment_names(script_name: str) -> List[str]:
    """
    Discover experiment names by running the script with --list in a subprocess.

    Using a subprocess instead of importlib.import_module avoids importing JAX,
    wandb, and multiprocessing spawn contexts in the launcher process, which can
    hang when CUDA is already initialised by other processes or in poetry envs.
    """
    module_name = script_name.removesuffix(".py")
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""  # hide GPUs — we only need names
    result = subprocess.run(
        [sys.executable, f"{module_name}.py", "--list"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    names = []
    for line in result.stdout.splitlines():
        line = line.strip()
        # Lines look like: "[ 0] exp_name   project=..."
        if line.startswith("[") and "]" in line:
            after_bracket = line.split("]", 1)[1].strip()
            name = after_bracket.split()[0]
            names.append(name)
    if not names:
        raise RuntimeError(
            f"get_experiment_names: no experiments found from '{script_name} --list'.\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
    return names


def launch_experiment(
    script_name: str, exp_index: int, exp_name: str, gpu_id: int
) -> subprocess.Popen:
    """
    Spawn a subprocess for one experiment with CUDA_VISIBLE_DEVICES set so
    that JAX in the child process sees only the assigned GPU as device 0.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("JAX_PLATFORMS", None)  # child should use GPU, not CPU

    module_name = script_name.removesuffix(".py")
    cmd = [sys.executable, f"{module_name}.py", "--exp-index", str(exp_index)]

    log_path = f"logs/{exp_name}_gpu{gpu_id}.log"
    os.makedirs("logs", exist_ok=True)
    log_file = open(log_path, "w")

    print(f"  [GPU {gpu_id}] launching [{exp_index}] {exp_name} → {log_path}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # Each experiment gets its own process group so Ctrl+C in the launcher
        # doesn't kill children immediately — we handle cleanup ourselves.
        start_new_session=True,
    )
    proc._log_file = log_file
    return proc


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------


def run_sweep(
    script_name: str,
    gpu_ids: List[int],
    dry_run: bool = False,
    poll_interval: int = 30,
) -> None:
    """
    Work through the experiment queue one job at a time, launching onto a GPU
    only when nvidia-smi reports memory usage below FREE_MEM_THRESHOLD_MIB.
    Memory is always visible to all users, so this catches other people's jobs too.
    """
    exp_names = get_experiment_names(script_name)
    n_exps = len(exp_names)

    print(f"\nSweep: {n_exps} experiments  script={script_name}  GPUs={gpu_ids}")
    print(f"Polling every {poll_interval}s; only launching on GPUs free system-wide.\n")
    for i, name in enumerate(exp_names):
        print(f"  [{i:2d}] {name}")
    print()

    if dry_run:
        print("Dry run — no processes launched.")
        return

    queue: deque = deque(enumerate(exp_names))
    # gpu_id → running Job (None = our slot is empty)
    running: Dict[int, Optional[Job]] = {g: None for g in gpu_ids}
    completed: List[str] = []
    failed: List[str] = []

    try:
        while queue or any(j is not None for j in running.values()):

            # --- Collect finished jobs ---
            for gpu_id, job in list(running.items()):
                if job is None:
                    continue
                ret = job.process.poll()
                if ret is not None:
                    job.process._log_file.close()
                    if ret == 0:
                        completed.append(job.exp_name)
                        status = "✓ done"
                    else:
                        failed.append(job.exp_name)
                        status = f"✗ failed (exit {ret})"
                    print(
                        f"  [GPU {gpu_id}] {status}: {job.exp_name}  "
                        f"({len(completed) + len(failed)}/{n_exps} finished)"
                    )
                    running[gpu_id] = None

                    # Fail fast: drain queue and terminate other jobs on failure
                    if ret != 0:
                        print(f"\n  Experiment failed — aborting remaining queue ({len(queue)} pending).")
                        queue.clear()
                        for other_gpu, other_job in running.items():
                            if other_job is not None and other_job.process.poll() is None:
                                os.killpg(os.getpgid(other_job.process.pid), 15)
                                other_job.process._log_file.close()
                                print(f"  [GPU {other_gpu}] terminated: {other_job.exp_name}")
                                running[other_gpu] = None

            # --- Fill empty slots — but only on GPUs that are truly idle ---
            if queue:
                # One nvidia-smi call covers all GPUs this iteration
                mem_used = get_gpu_memory_used()

                for gpu_id in gpu_ids:
                    if not queue:
                        break
                    if running[gpu_id] is not None:
                        continue  # our job is still running on this GPU
                    used = mem_used.get(gpu_id, 0)
                    if used >= FREE_MEM_THRESHOLD_MIB:
                        print(
                            f"  [GPU {gpu_id}] busy ({used} MiB used, "
                            f"threshold {FREE_MEM_THRESHOLD_MIB} MiB) "
                            f"— will retry in {poll_interval}s"
                        )
                        continue
                    # GPU is free — claim it
                    exp_index, exp_name = queue.popleft()
                    proc = launch_experiment(script_name, exp_index, exp_name, gpu_id)
                    running[gpu_id] = Job(
                        exp_index=exp_index,
                        exp_name=exp_name,
                        gpu_id=gpu_id,
                        process=proc,
                    )

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nInterrupted — terminating running experiments...")
        for gpu_id, job in running.items():
            if job is not None and job.process.poll() is None:
                os.killpg(os.getpgid(job.process.pid), 15)  # SIGTERM to process group
                job.process._log_file.close()
                print(f"  [GPU {gpu_id}] terminated: {job.exp_name}")
        sys.exit(1)

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"Sweep complete: {len(completed)}/{n_exps} succeeded")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  ✗ {name}")
        print(f"Logs in: logs/")
        sys.exit(1)
    print(f"Logs in: logs/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch experiments one-by-one onto free GPUs (system-wide check)."
    )
    parser.add_argument(
        "--script",
        type=str,
        default="plane_exps",
        help=(
            "Experiment script to run (without .py). "
            "Must expose build_experiments() and accept --exp-index. "
            "Default: plane_exps. Use 'ablation_study' for the ablation suite."
        ),
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=None,
        help="GPU IDs to consider (default: all GPUs detected by nvidia-smi).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the experiment list without launching anything.",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=30,
        metavar="SECONDS",
        help="How often to poll GPU availability and job status (default: 30).",
    )
    args = parser.parse_args()

    gpu_ids = args.gpus if args.gpus is not None else get_all_gpu_ids()
    print(f"Candidate GPUs: {gpu_ids}")

    run_sweep(
        script_name=args.script,
        gpu_ids=gpu_ids,
        dry_run=args.dry_run,
        poll_interval=args.poll,
    )
