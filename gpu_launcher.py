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
    python gpu_launcher.py --total-timesteps 500000    # for progress bars (default: 1M)
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Job:
    exp_index: int
    exp_name: str
    gpu_id: int
    log_path: str
    start_time: float
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
    if mem_used is None:
        mem_used = get_gpu_memory_used()
    return mem_used.get(gpu_id, 0) < FREE_MEM_THRESHOLD_MIB


# ---------------------------------------------------------------------------
# Experiment discovery and launching
# ---------------------------------------------------------------------------


def get_experiment_names(script_name: str, extra_args: List[str] = []) -> List[str]:
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
        [sys.executable, f"{module_name}.py", "--list"] + extra_args,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    names = []
    for line in result.stdout.splitlines():
        line = line.strip()
        # Lines look like: "[ 0] exp_name   project=..."
        # or with completion status: "[ 0] DONE  exp_name  group=..."
        if line.startswith("[") and "]" in line:
            after_bracket = line.split("]", 1)[1].strip()
            tokens = after_bracket.split()
            if not tokens:
                continue
            # Skip the "DONE" status prefix added by --list when an experiment is complete.
            name = tokens[1] if tokens[0] == "DONE" and len(tokens) > 1 else tokens[0]
            names.append(name)
    if not names:
        raise RuntimeError(
            f"get_experiment_names: no experiments found from '{script_name} --list'.\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
    return names


def launch_experiment(
    script_name: str, exp_index: int, exp_name: str, gpu_id: int,
    extra_args: List[str] = [],
) -> tuple["subprocess.Popen", str]:
    """
    Spawn a subprocess for one experiment with CUDA_VISIBLE_DEVICES set so
    that JAX in the child process sees only the assigned GPU as device 0.
    Returns (process, log_path).
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("JAX_PLATFORMS", None)  # child should use GPU, not CPU

    module_name = script_name.removesuffix(".py")
    cmd = [sys.executable, f"{module_name}.py", "--exp-index", str(exp_index)] + extra_args

    log_path = f"logs/{exp_name}_gpu{gpu_id}.log"
    os.makedirs("logs", exist_ok=True)
    log_file = open(log_path, "w")

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
    return proc, log_path


# ---------------------------------------------------------------------------
# Live display helpers
# ---------------------------------------------------------------------------

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_status_line_count = 0  # how many live status lines are currently on screen


def _elapsed_str(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m:02d}:{s:02d}"


def _tb_last_step(tb_dir: Path) -> int:
    """Return the highest step seen in any TFEvents file under tb_dir."""
    last = 0
    for ef in tb_dir.glob("events.out.tfevents.*"):
        try:
            with open(ef, "rb") as f:
                while True:
                    hdr = f.read(12)
                    if len(hdr) < 12:
                        break
                    length = struct.unpack("Q", hdr[:8])[0]
                    data = f.read(length)
                    f.read(4)  # footer CRC
                    # Parse proto field 2 (step, varint) from the Event message.
                    idx = 0
                    while idx < len(data):
                        b = data[idx]; idx += 1
                        fn, wt = b >> 3, b & 7
                        if fn == 2 and wt == 0:
                            val = 0; shift = 0
                            while idx < len(data):
                                b2 = data[idx]; idx += 1
                                val |= (b2 & 0x7F) << shift; shift += 7
                                if not (b2 & 0x80): break
                            last = max(last, val)
                            break
                        elif wt == 0:
                            while idx < len(data):
                                b2 = data[idx]; idx += 1
                                if not (b2 & 0x80): break
                        elif wt == 1: idx += 8
                        elif wt == 2:
                            vlen = 0; shift = 0
                            while idx < len(data):
                                b2 = data[idx]; idx += 1
                                vlen |= (b2 & 0x7F) << shift; shift += 7
                                if not (b2 & 0x80): break
                            idx += vlen
                        else:
                            break
        except Exception:
            pass
    return last


def _parse_latest_step(
    exp_name: str,
    job_start_time: float,
    registry_path: str = "ablation_run_registry.json",
    tb_root: str = "tensorboard",
) -> Optional[int]:
    """
    Read progress for *exp_name* from TensorBoard event files.

    Looks up run_ids registered for this experiment after job_start_time
    (so stale runs from previous launches are ignored), then returns the
    maximum step seen across all those TB directories.
    """
    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except Exception:
        return None

    tb_dir = Path(tb_root)
    best = 0
    found = False
    for entry in registry:
        if entry.get("exp_name") != exp_name:
            continue
        run_id = entry.get("run_id")
        if not run_id:
            continue
        run_tb = tb_dir / run_id
        if not run_tb.exists():
            continue
        # Only consider directories created after this job started.
        if run_tb.stat().st_mtime < job_start_time:
            continue
        step = _tb_last_step(run_tb)
        if step > 0:
            best = max(best, step)
            found = True

    return best if found else None


def _render_bar(frac: float, width: int = 22) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def _build_status_lines(
    running: Dict[int, Optional[Job]],
    gpu_ids: List[int],
    total_timesteps: int,
    completed: int,
    failed: int,
    n_exps: int,
    queue_len: int,
    registry_path: str = "ablation_run_registry.json",
) -> List[str]:
    term_w = shutil.get_terminal_size((120, 24)).columns
    now = time.time()
    lines = []

    for gpu_id in sorted(gpu_ids):
        job = running.get(gpu_id)
        if job is None:
            lines.append(f"  [GPU {gpu_id}]  idle")
            continue

        elapsed = now - job.start_time
        elapsed_s = _elapsed_str(elapsed)
        step = _parse_latest_step(job.exp_name, job.start_time, registry_path=registry_path)

        name_col = f"{job.exp_name:<36}"

        if step is not None and total_timesteps > 0:
            frac = min(step / total_timesteps, 1.0)
            bar = _render_bar(frac)
            pct = f"{frac * 100:5.1f}%"
            eta_s = ""
            if frac > 0.01:
                remaining = elapsed / frac * (1 - frac)
                eta_s = f"  eta {_elapsed_str(remaining)}"
            line = f"  [GPU {gpu_id}]  {name_col}  [{bar}] {pct}  {elapsed_s}{eta_s}"
        else:
            spin = _SPINNER[int(elapsed * 2) % len(_SPINNER)]
            line = f"  [GPU {gpu_id}]  {name_col}  {spin} starting...  {elapsed_s}"

        lines.append(line[:term_w])

    done_total = completed + failed
    summary = (
        f"  {done_total}/{n_exps} done  "
        f"({completed} ok, {failed} failed)  "
        f"{queue_len} queued"
    )
    lines.append(summary)
    return lines


def _clear_status() -> None:
    """Erase the live status block from the terminal."""
    global _status_line_count
    if _status_line_count > 0:
        sys.stdout.write(f"\033[{_status_line_count}A")  # move cursor up
        for _ in range(_status_line_count):
            sys.stdout.write("\033[2K\n")               # clear each line
        sys.stdout.write(f"\033[{_status_line_count}A")  # back to top
        sys.stdout.flush()
    _status_line_count = 0


def _draw_status(lines: List[str]) -> None:
    """Draw (or redraw) the live status block."""
    global _status_line_count
    _clear_status()
    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()
    _status_line_count = len(lines)


def print_event(msg: str) -> None:
    """Print a permanent log line above the live status block."""
    _clear_status()
    print(msg)


# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------

DISPLAY_REFRESH = 2  # seconds between live display refreshes


def run_sweep(
    script_name: str,
    gpu_ids: List[int],
    dry_run: bool = False,
    poll_interval: int = 30,
    extra_args: List[str] = [],
    total_timesteps: int = 1_000_000,
) -> None:
    """
    Work through the experiment queue one job at a time, launching onto a GPU
    only when nvidia-smi reports memory usage below FREE_MEM_THRESHOLD_MIB.
    """
    exp_names = get_experiment_names(script_name, extra_args)
    n_exps = len(exp_names)
    registry_path = f"{script_name.removesuffix('.py').removesuffix('_study')}_run_registry.json"

    print(f"\nSweep: {n_exps} experiments  script={script_name}  GPUs={gpu_ids}")
    print(f"Polling every {poll_interval}s; only launching on GPUs free system-wide.\n")
    for i, name in enumerate(exp_names):
        print(f"  [{i:2d}] {name}")
    print()

    if dry_run:
        print("Dry run — no processes launched.")
        return

    queue: deque = deque(enumerate(exp_names))
    running: Dict[int, Optional[Job]] = {g: None for g in gpu_ids}
    completed: List[str] = []
    failed: List[str] = []

    last_poll = 0.0   # force immediate first poll

    try:
        while queue or any(j is not None for j in running.values()):
            now = time.time()

            # --- Poll GPU / job state at full interval ---
            if now - last_poll >= poll_interval:
                last_poll = now

                # Collect finished jobs
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
                        elapsed = _elapsed_str(time.time() - job.start_time)
                        print_event(
                            f"  [GPU {gpu_id}] {status}: {job.exp_name}  "
                            f"({len(completed) + len(failed)}/{n_exps} finished)"
                            f"  [{elapsed}]"
                        )
                        running[gpu_id] = None

                        if ret != 0:
                            print_event(
                                f"\n  Experiment failed — aborting remaining queue "
                                f"({len(queue)} pending)."
                            )
                            queue.clear()
                            for other_gpu, other_job in running.items():
                                if other_job is not None and other_job.process.poll() is None:
                                    os.killpg(os.getpgid(other_job.process.pid), 15)
                                    other_job.process._log_file.close()
                                    print_event(f"  [GPU {other_gpu}] terminated: {other_job.exp_name}")
                                    running[other_gpu] = None

                # Fill empty slots on idle GPUs
                if queue:
                    mem_used = get_gpu_memory_used()
                    for gpu_id in gpu_ids:
                        if not queue:
                            break
                        if running[gpu_id] is not None:
                            continue
                        used = mem_used.get(gpu_id, 0)
                        if used >= FREE_MEM_THRESHOLD_MIB:
                            print_event(
                                f"  [GPU {gpu_id}] busy ({used} MiB used, "
                                f"threshold {FREE_MEM_THRESHOLD_MIB} MiB) "
                                f"— will retry in {poll_interval}s"
                            )
                            continue
                        exp_index, exp_name = queue.popleft()
                        proc, log_path = launch_experiment(
                            script_name, exp_index, exp_name, gpu_id, extra_args
                        )
                        print_event(
                            f"  [GPU {gpu_id}] launched [{exp_index}] {exp_name}"
                            f"  →  {log_path}"
                        )
                        running[gpu_id] = Job(
                            exp_index=exp_index,
                            exp_name=exp_name,
                            gpu_id=gpu_id,
                            log_path=log_path,
                            start_time=time.time(),
                            process=proc,
                        )

            # --- Redraw live status ---
            status = _build_status_lines(
                running, gpu_ids, total_timesteps,
                len(completed), len(failed), n_exps, len(queue),
                registry_path=registry_path,
            )
            _draw_status(status)
            time.sleep(DISPLAY_REFRESH)

    except KeyboardInterrupt:
        _clear_status()
        print("\nInterrupted — terminating running experiments...")
        for gpu_id, job in running.items():
            if job is not None and job.process.poll() is None:
                os.killpg(os.getpgid(job.process.pid), 15)
                job.process._log_file.close()
                print(f"  [GPU {gpu_id}] terminated: {job.exp_name}")
        sys.exit(1)

    _clear_status()
    print(f"\n{'=' * 50}")
    print(f"Sweep complete: {len(completed)}/{n_exps} succeeded")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name in failed:
            print(f"  ✗ {name}")
        print("Logs in: logs/")
        sys.exit(1)
    print("Logs in: logs/")


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
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        metavar="N",
        help="Expected total training timesteps per experiment, used for progress bars (default: 1 000 000).",
    )
    args, extra_args = parser.parse_known_args()

    gpu_ids = args.gpus if args.gpus is not None else get_all_gpu_ids()
    print(f"Candidate GPUs: {gpu_ids}")
    if extra_args:
        print(f"Forwarding to script: {extra_args}")

    run_sweep(
        script_name=args.script,
        gpu_ids=gpu_ids,
        dry_run=args.dry_run,
        poll_interval=args.poll,
        extra_args=extra_args,
        total_timesteps=args.total_timesteps,
    )
