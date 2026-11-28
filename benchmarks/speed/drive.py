"""Runs one cell of the speed comparison and appends its record.

usage, from the repository root on the machine to time:
    python3 benchmarks/speed/drive.py SAC HalfCheetah-v5 SB3 --seeds 10 \\
        --cores 42-45 --warmup 2000 --segment 3000 [--gpu 0]

Each library runs from its own environment, envs/<library>/.venv (Ajax from
the repository's .venv), pinned to --cores with taskset. Ajax, PureJaxRL and
purejaxql train every seed in one program; the other libraries run one
process per seed, all at once on the CPU, torch at one thread each (JAX's
CPU pool follows the cores), so the processes share the cores;
Brax's seeds on a GPU run one after another, the GPU rule being three jobs of
ours per card. The record holds the environment steps per second over every
seed, how the segments were timed, what else ran on the cores meanwhile, the
libraries' versions and the manifest (commit, machine, date).
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib

import speed_specs as specs

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VENDOR = ROOT / "benchmarks" / "reference" / "vendor"
RECORDS = HERE / "results" / "speed.jsonl"
#: Ajax's distributions each record keeps; a library's are those its
#: envs/<library>/pyproject.toml lists.
AJAX_VERSIONS = ("ajax-rl", "jax", "jaxlib", "flax", "optax", "gymnax", "mujoco-mjx")
#: Vendored code, by the commit it was copied at.
SOURCES = {
    "CleanRL": "vwxyzjn/cleanrl@fe8d8a0",
    "PureJaxRL": "luchris429/purejaxrl@31756b1",
    "purejaxql": "mttga/purejaxql@47af6d7",
}


def project(library: str, gpu: bool) -> pathlib.Path:
    """The library's uv project: its CUDA twin on a GPU for the libraries
    whose CPU project has CPU-only builds (SB3, SBX, CleanRL)."""
    if library == "Ajax":
        return ROOT
    name = library.lower() + ("-cuda" if gpu and library in specs.PYTHON_LOOP else "")
    return HERE / "envs" / name


def python(library: str, gpu: bool = False) -> str:
    return str(project(library, gpu) / ".venv" / "bin" / "python")


def command(args, seed: int, record: str) -> list[str]:
    """The command line of one process: seed ``seed`` (a process-per-seed
    library) or every seed (the others)."""
    a, task, library = args.algorithm, args.task, args.library
    end = str(args.warmup + args.segment)
    gym_id = f"speed_clock:speed/{specs.gymnasium_id(task, library)}"
    if library in ("SB3", "SBX"):
        return [str(HERE / "run_sb3.py"), library, a, task, str(seed), end]
    if library == "CleanRL":
        s = specs.settings(a, task)
        flags = ["--env-id", gym_id, "--seed", str(seed), "--total-timesteps", end]
        if "learning_starts" in s:
            flags += ["--learning-starts", str(s["learning_starts"])]
        if a == "SAC":  # an actor and temperature step per critic step, as SB3's
            flags += ["--policy-frequency", "1", "--q-lr", str(s["learning_rate"])]
        if a == "PQN":
            _, final, fraction = s["epsilon"]
            pqn = {
                "num-envs": s["n_envs"],
                "num-steps": s["n_steps"],
                "num-minibatches": s["n_minibatches"],
                "update-epochs": s["n_epochs"],
                "learning-rate": s["learning_rate"],
                "q-lambda": s["q_lambda"],
                "end-e": final,
                "exploration-fraction": fraction,
            }
            flags += [
                x for name, value in pqn.items() for x in (f"--{name}", str(value))
            ]
        return [
            str(VENDOR / "cleanrl" / "cleanrl" / specs.cleanrl_script(a, task)),
            *flags,
        ]
    if library == "Brax":
        return [
            str(HERE / "run_brax.py"),
            a,
            task,
            str(seed),
            str(args.segment),
            record,
        ]
    runner = "run_ajax.py" if library == "Ajax" else "run_jax.py"
    tail = [a, task, str(args.seeds), str(args.warmup), str(args.segment)]
    return [str(HERE / runner), *([] if library == "Ajax" else [library]), *tail]


def environment(args, barrier: str, record: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(HERE), PYTHONUNBUFFERED="1")
    env.update(
        {
            "SPEED_START": str(args.warmup),
            "SPEED_END": str(args.warmup + args.segment),
            "SPEED_BARRIER": barrier,
            "SPEED_RECORD": record,
            "SPEED_PROCESSES": str(args.seeds if args.mode == "processes" else 1),
        }
    )
    if args.library == "CleanRL":
        env["PYTHONPATH"] += os.pathsep + str(VENDOR / "cleanrl")
    if args.library == "Ajax":
        env["AJAX_NO_COMPILE_CACHE"] = "1"
    if args.gpu is None:
        env.update({"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})
    else:
        env.update(
            {"CUDA_VISIBLE_DEVICES": args.gpu, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        )
    if args.threads == 1:  # torch's threads; JAX's CPU pool follows the cores
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return env


class CoreLoad(threading.Thread):
    """Samples /proc/stat each second: the busy jiffies of the run's cores
    and of their SMT siblings."""

    def __init__(self, cores: list[int]):
        super().__init__(daemon=True)
        self.cores, self.samples, self.done = cores, [], threading.Event()

    def run(self) -> None:
        while not self.done.is_set():
            busy = {}
            for line in pathlib.Path("/proc/stat").read_text().splitlines():
                name, *fields = line.split()
                if name[3:].isdigit():
                    values = list(map(int, fields))
                    busy[int(name[3:])] = sum(values) - values[3] - values[4]
            self.samples.append((time.time(), busy))
            self.done.wait(1.0)

    def busy_seconds(self, start: float, end: float, cores: list[int]) -> float:
        inside = [s for s in self.samples if start <= s[0] <= end]
        if len(inside) < 2:
            return float("nan")
        first, last = inside[0][1], inside[-1][1]
        span = inside[-1][0] - inside[0][0]
        hz = os.sysconf("SC_CLK_TCK")
        return sum(last[c] - first[c] for c in cores) / hz * (end - start) / span


def parse_cores(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        out += range(int(low), int(high or low) + 1)
    return out


def versions(library: str, gpu: bool) -> dict:
    """The installed versions of the library's distributions (its project's
    dependencies, the CUDA plugin's where there is one) and its lock's hash."""
    where = project(library, gpu)
    if library == "Ajax":
        wanted = list(AJAX_VERSIONS)
    else:
        listed = tomllib.loads((where / "pyproject.toml").read_text())["project"]
        wanted = [re.split(r"[=<>\[ ;]", d)[0] for d in listed["dependencies"]]
    code = (
        "import importlib.metadata as m, json, sys\n"
        "names = {d.metadata['Name'] for d in m.distributions()}\n"
        "print(json.dumps({d: m.version(d) for d in sys.argv[1:] if d in names}))"
    )
    wanted += ["jaxlib", "jax-cuda12-plugin", "jax-cuda13-plugin"]
    out = subprocess.run(
        [python(library), "-c", code, *wanted],
        capture_output=True,
        text=True,
        check=True,
    )
    found = json.loads(out.stdout)
    lock = (where / "uv.lock").read_bytes()
    found["lock_sha256"] = hashlib.sha256(lock).hexdigest()[:12]
    return found


def manifest(args) -> dict:
    def git(*a):
        return subprocess.run(
            ["git", *a], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    cpu = next(
        line.split(":", 1)[1].strip()
        for line in open("/proc/cpuinfo")
        if line.startswith("model name")
    )
    gpu = None
    if args.gpu is not None:
        query = [
            "nvidia-smi",
            "-i",
            args.gpu,
            "--query-gpu=name",
            "--format=csv,noheader",
        ]
        gpu = subprocess.run(query, capture_output=True, text=True).stdout.strip()
    return {
        "commit": git("rev-parse", "HEAD"),
        "clean_tree": not git(
            "status", "--porcelain", "--", ".", ":!benchmarks/speed/results"
        ),
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "machine": {
            "host": platform.node(),
            "cpu": cpu,
            "gpu": gpu,
            "python": platform.python_version(),
        },
    }


def planned(device: str) -> list[str]:
    """Every cell of ``device`` with its seeds and stretch, as drive.py's
    arguments, most seeds first. On a GPU a process-per-seed library runs one
    seed: ten or fifty would run one after another, at its one-seed speed."""
    out = []
    for seeds in sorted(specs.SEEDS, reverse=True):
        for algorithm, task, library in specs.cells():
            if device == "gpu" and library not in specs.ONE_PROGRAM and seeds > 1:
                continue
            warmup, segment = specs.segment(algorithm, task, library, seeds, device)
            out.append(
                f"{algorithm} {task} {library} --seeds {seeds}"
                f" --warmup {warmup} --segment {segment}"
            )
    return out


def main() -> int:
    if sys.argv[1:2] == ["--list"]:  # --list [cpu|gpu]
        print("\n".join(planned(sys.argv[2] if len(sys.argv) > 2 else "cpu")))
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("algorithm", choices=specs.TASKS)
    parser.add_argument("task")
    parser.add_argument("library")
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--cores", required=True)
    parser.add_argument(
        "--warmup", type=int, required=True, help="steps per seed before the segment"
    )
    parser.add_argument(
        "--segment", type=int, required=True, help="timed steps per seed"
    )
    parser.add_argument("--gpu", help="the GPU's index; the CPU without it")
    parser.add_argument("--out", type=pathlib.Path, default=RECORDS)
    args = parser.parse_args()
    assert (args.algorithm, args.task, args.library) in specs.cells()
    args.mode = "program" if args.library in specs.ONE_PROGRAM else "processes"
    # On a GPU, at most three of our jobs share a card: a process-per-seed
    # library's ten seeds would run one after another, at its one-seed speed.
    assert args.mode == "program" or args.gpu is None or args.seeds == 1
    args.threads = 1 if args.mode == "processes" and args.seeds > 1 else None
    if args.library == "Brax":  # its first epoch, after the prefill, is as long
        args.warmup = (
            specs.settings(args.algorithm, args.task).get("learning_starts", 0)
            + args.segment
        )
    cores = parse_cores(args.cores)
    siblings = [c + os.cpu_count() // 2 for c in cores]
    work = tempfile.mkdtemp(prefix="speed-")
    barrier, record = (
        os.path.join(work, "barrier"),
        os.path.join(work, "segments.jsonl"),
    )
    os.mkdir(barrier)
    load = CoreLoad(cores + siblings)
    load.start()
    launches = range(args.seeds) if args.mode == "processes" else [0]
    processes = []
    for seed in launches:
        log = open(os.path.join(work, f"{seed}.log"), "w")
        argv = [
            "taskset",
            "-c",
            args.cores,
            python(args.library, args.gpu is not None),
            *command(args, seed, record),
        ]
        processes.append(
            subprocess.Popen(
                argv,
                cwd=work,
                env=environment(args, barrier, record),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
    codes = [p.wait() for p in processes]
    load.done.set()
    if any(codes):
        print(f"failed: exit codes {codes}, logs in {work}", file=sys.stderr)
        return 1
    logs = {seed: pathlib.Path(work, f"{seed}.log").read_text() for seed in launches}
    if args.mode == "program":  # the runner prints its own record
        lines = logs[0].splitlines()
        result = json.loads(
            next(line for line in reversed(lines) if line.startswith("{"))
        )
        windows, steps, seconds = (
            result.pop("windows"),
            result.pop("steps"),
            result.pop("seconds"),
        )
    else:
        record_lines = pathlib.Path(record).read_text().splitlines()
        windows = [json.loads(line) for line in record_lines]
        assert len(windows) == args.seeds, windows
        steps = sum(w["steps"] for w in windows)
        # From the first start to the last end.
        seconds = max(w["end"] for w in windows) - min(w["start"] for w in windows)
        result = {}
    # The cores' busy share over the timed stretches: the processes' common
    # span, or each window of a single process; "ours" is the runs' own CPU.
    if args.mode == "processes":
        stretches = [(min(w["start"] for w in windows), max(w["end"] for w in windows))]
    else:
        stretches = [(w["start"], w["end"]) for w in windows]
    covered = len(cores) * sum(end - start for start, end in stretches)

    def busy(of: list[int]) -> float:
        """Over the stretches the 1-second samples cover (a stretch under
        two seconds may hold too few)."""
        seen = [(load.busy_seconds(s, e, of), e - s) for s, e in stretches]
        seen = [(b, d) for b, d in seen if not math.isnan(b)]
        total = len(of) * sum(d for _, d in seen)
        return round(sum(b for b, _ in seen) / total, 3) if total else None

    entry = {
        "algorithm": args.algorithm,
        "task": args.task,
        "library": args.library,
        "physics": specs.physics(args.task, args.library),
        "seeds": args.seeds,
        "device": "gpu" if args.gpu is not None else "cpu",
        "how": args.mode,
        "threads": args.threads,
        "env_steps_per_s": round(steps / seconds, 1),
        "warmup": args.warmup,
        "segment": args.segment,
        "seconds": round(seconds, 3),
        **result,
        "cores": cores,
        "load": {
            "cores_busy": busy(cores),
            "siblings_busy": busy(siblings),
            "ours": round(sum(w["cpu_s"] for w in windows) / covered, 3),
            "loadavg": os.getloadavg(),
        },
        "windows": windows,
        "settings": specs.settings(args.algorithm, args.task),
        "exception": specs.EXCEPTIONS.get((args.algorithm, args.library)),
        "source": SOURCES.get(args.library),
        "versions": versions(args.library, args.gpu is not None),
        "manifest": manifest(args),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as file:
        file.write(json.dumps(entry, default=str) + "\n")
    shutil.rmtree(work)  # the processes' logs, and CleanRL's TensorBoard files
    keys = ("algorithm", "task", "library", "seeds", "device", "env_steps_per_s")
    print(json.dumps({k: entry[k] for k in keys}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
