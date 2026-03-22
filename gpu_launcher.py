"""
gpu_launcher.py

Distributes experiments from plane_exps.py across N GPUs automatically.
Each experiment runs as a subprocess with CUDA_VISIBLE_DEVICES set before
JAX is imported, which is required for correct GPU selection.

Usage:
    python gpu_launcher.py              # uses all 4 GPUs
    python gpu_launcher.py --gpus 0 1  # use only GPUs 0 and 1
    python gpu_launcher.py --dry-run   # print what would run without running it
"""

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Job:
    exp_index: int
    exp_name: str
    gpu_id: int
    process: Optional[subprocess.Popen] = None


def get_experiment_names() -> List[str]:
    """
    Import build_experiments() from plane_exps without triggering JAX import.
    We only need the names and count here — no GPU needed.
    """
    # Temporarily suppress JAX GPU allocation in this process
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from plane_exps import build_experiments

    return [exp.name for exp in build_experiments()]


def launch_experiment(exp_index: int, exp_name: str, gpu_id: int) -> subprocess.Popen:
    """
    Spawn a subprocess for one experiment with CUDA_VISIBLE_DEVICES set.
    JAX in the child process sees only the assigned GPU as device 0.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Prevent the child from inheriting our cpu-only JAX override
    env.pop("JAX_PLATFORMS", None)

    cmd = [sys.executable, "plane_exps.py", "--exp-index", str(exp_index)]

    log_path = f"logs/{exp_name}_gpu{gpu_id}.log"
    os.makedirs("logs", exist_ok=True)
    log_file = open(log_path, "w")

    print(f"  [GPU {gpu_id}] launching exp {exp_index}: {exp_name} → {log_path}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        # Each experiment is its own process group so Ctrl+C in the launcher
        # doesn't immediately kill all children — we handle cleanup ourselves
        start_new_session=True,
    )
    proc._log_file = log_file  # keep reference to close later
    return proc


def run_sweep(gpu_ids: List[int], dry_run: bool = False):
    """
    Run all experiments, keeping each GPU busy at all times.
    Uses a simple queue: as soon as a GPU finishes, the next experiment starts.
    """
    exp_names = get_experiment_names()
    n_exps = len(exp_names)

    print(f"\nSweep: {n_exps} experiments across GPUs {gpu_ids}")
    print(f"Each experiment will be queued as a GPU becomes free.\n")

    for i, name in enumerate(exp_names):
        print(f"  [{i:2d}] {name}")
    print()

    if dry_run:
        print("Dry run — no processes launched.")
        return

    # Queue of (exp_index, exp_name) to run
    queue = deque(enumerate(exp_names))

    # Map gpu_id → running Job (or None if idle)
    running: dict[int, Optional[Job]] = {g: None for g in gpu_ids}

    completed = []
    failed = []

    try:
        while queue or any(j is not None for j in running.values()):
            # --- Check for finished jobs ---
            for gpu_id, job in list(running.items()):
                if job is None:
                    continue
                ret = job.process.poll()
                if ret is not None:
                    # Job finished
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

            # --- Fill idle GPUs from queue ---
            for gpu_id, job in running.items():
                if job is None and queue:
                    exp_index, exp_name = queue.popleft()
                    proc = launch_experiment(exp_index, exp_name, gpu_id)
                    running[gpu_id] = Job(
                        exp_index=exp_index,
                        exp_name=exp_name,
                        gpu_id=gpu_id,
                        process=proc,
                    )

            # Poll every 10 seconds — experiments take minutes each
            time.sleep(10)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribute experiments across GPUs")
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="GPU IDs to use (default: 0 1 2 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without launching anything",
    )
    args = parser.parse_args()

    run_sweep(gpu_ids=args.gpus, dry_run=args.dry_run)
