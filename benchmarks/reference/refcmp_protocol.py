"""The reproduction protocol: AdaStop decides, interim by interim, whether an
Ajax arm learns like its reference (docs/reproduction/protocol.md).

usage, from the repository root:
    uv run python benchmarks/reference/refcmp_protocol.py run NAME [ARM ...]
    uv run python benchmarks/reference/refcmp_protocol.py report [NAME ...]

``--cache DIR`` keeps the records in DIR instead of ``results/protocol``:
``run NAME --cache DIR`` with an empty DIR reproduces a comparison from
scratch.

``run`` replays AdaStop over the runs already recorded and trains only the
seeds it lacks: ``n`` per side and interim (the spec's ``adastop``), the
reference in its own uv project (``envs/<library>``), Ajax in the
repository's. Ajax and purejaxql train a batch of seeds in one program;
SB3 and Brax, whose trainers have no seed axis, one process per seed. Each
batch appends one record to ``results/protocol/<NAME>.jsonl``: the per-seed
curves, the speed (wall, CPU and compile seconds, environment steps per
second, the machine) and a manifest (Ajax's commit and whether the tree was
clean, the lockfile's hash, versions, platform, resolved settings). A
record is reused while its ``fingerprint``, the hash of the settings and the
code it ran, stays the same.
"""

import datetime
import hashlib
import json
import os
import pathlib
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from refcmp_specs import COMPARISONS, spec_sha256

from ajax.compare import AdaStop, difference, resolution

# docs/reproduction/protocol.md's version, recorded with every batch.
PROTOCOL_VERSION = 1
HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
CACHE = HERE / "results" / "protocol"
# Library -> (uv project, runner, one process per seed, the code it runs).
# refcmp_jax.py holds purejaxql's runner and the evaluator of final policies.
IMPLEMENTATIONS = {
    "ajax": (
        ROOT,
        "refcmp_ajax.py",
        False,
        ("src/ajax", "uv.lock", "refcmp_ajax.py", "refcmp_jax.py"),
    ),
    "sb3": (
        HERE / "envs" / "sb3",
        "refcmp_sb3.py",
        True,
        ("envs/sb3", "refcmp_sb3.py"),
    ),
    "purejaxql": (
        HERE / "envs" / "purejaxql",
        "refcmp_jax.py",
        False,
        ("envs/purejaxql", "vendor/purejaxql", "refcmp_jax.py"),
    ),
    "brax": (
        HERE / "envs" / "brax",
        "refcmp_brax.py",
        True,
        ("envs/brax", "refcmp_brax.py", "refcmp_jax.py"),
    ),
}


def implementation(name: str, side: str) -> str:
    return COMPARISONS[name].reference if side == "reference" else "ajax"


def digest(paths) -> str:
    """The SHA-256 of the files under ``paths`` (relative to this directory,
    or the repository's ``src/ajax`` and ``uv.lock``), their paths included;
    caches and virtual environments left out."""
    h = hashlib.sha256()
    for path in paths:
        base = ROOT if path in ("src/ajax", "uv.lock") else HERE
        files = (
            sorted((base / path).rglob("*"))
            if (base / path).is_dir()
            else [base / path]
        )
        for f in files:
            parts = f.relative_to(base).parts
            if f.is_file() and not {"__pycache__", ".venv"} & set(parts):
                h.update("/".join(parts).encode() + f.read_bytes())
    return h.hexdigest()


def fingerprint(name: str, side: str) -> str:
    spec, lib = COMPARISONS[name], implementation(name, side)
    settings = {
        "agent": spec.agent,
        "env_id": spec.env_id,
        "n_timesteps": spec.n_timesteps,
        "eval_every": spec.eval_every,
        "n_eval_episodes": spec.n_eval_episodes,
        "config": spec.reference_config if side == "reference" else getattr(spec, side),
        "code": digest(IMPLEMENTATIONS[lib][3]),
    }
    text = json.dumps(settings, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def eval_steps(spec) -> list:
    if spec.eval_every == 0:
        return [spec.n_timesteps]
    return list(range(spec.eval_every, spec.n_timesteps + 1, spec.eval_every))


def hardware() -> dict:
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = _run("sysctl", "-n", "machdep.cpu.brand_string")
    elif pathlib.Path("/proc/cpuinfo").exists():
        lines = pathlib.Path("/proc/cpuinfo").read_text().splitlines()
        cpu = next(
            (ln.split(":", 1)[1].strip() for ln in lines if "model name" in ln), cpu
        )
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": cpu,
        "cores": os.cpu_count(),
        "memory_gb": round(memory, 1),
    }


def _run(*command) -> str:
    return subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def train(name: str, side: str, seeds: list) -> dict:
    """Train ``seeds`` of one side on the CPU; the batch's record."""
    spec, lib = COMPARISONS[name], implementation(name, side)
    project, runner, per_seed, _ = IMPLEMENTATIONS[lib]
    groups = [[s] for s in seeds] if per_seed else [seeds]
    # Each project's own environment, not the caller's.
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env |= {"JAX_PLATFORMS": "cpu"}
    before, start = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    command = ["uv", "run", "--locked", "--project", str(project), "python"]
    processes = [
        subprocess.Popen(
            [*command, str(HERE / runner), name, side, str(group[0]), str(len(group))],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        for group in groups
    ]
    outputs = []
    for process in processes:
        out, _ = process.communicate()
        if process.returncode:
            raise RuntimeError(f"{runner} {name} {side}: exit {process.returncode}")
        outputs.append(json.loads(out.splitlines()[-1]))
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    return {
        "comparison": name,
        "spec_version": spec.version,
        "side": side,
        "implementation": lib,
        "fingerprint": fingerprint(name, side),
        "seeds": seeds,
        "steps": eval_steps(spec),
        "curve": [row for out in outputs for row in out["curve"]],
        "speed": {
            "wall_s": round(wall, 2),
            "cpu_s": round(cpu, 2),
            "compile_s": max(out["compile_s"] for out in outputs),
            "env_steps_per_s": round(len(seeds) * spec.n_timesteps / wall),
            "processes": len(groups),
        },
        "manifest": {
            "protocol_version": PROTOCOL_VERSION,
            "spec_sha256": spec_sha256(spec),
            "runner": f"benchmarks/reference/{runner}",
            "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "ajax_commit": _run("git", "rev-parse", "HEAD"),
            "ajax_clean": not _run("git", "status", "--porcelain"),
            "lock_sha256": hashlib.sha256(
                (project / "uv.lock").read_bytes()
            ).hexdigest(),
            "versions": outputs[0]["versions"],
            "platform": platform.platform(),
            "jax_platforms": env["JAX_PLATFORMS"],
            "hardware": hardware(),
            "config": outputs[0]["config"],
        },
    }


def records(name: str) -> list:
    path = CACHE / f"{name}.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    return [json.loads(line) for line in lines]


def current(name: str, side: str) -> list:
    """The side's records whose fingerprint matches the code and settings now."""
    key = fingerprint(name, side)
    return [r for r in records(name) if (r["side"], r["fingerprint"]) == (side, key)]


def curves(name: str, side: str) -> dict:
    """Seed -> evaluation curve, from the current records (the first run of a
    seed counts)."""
    by_seed: dict = {}
    for r in current(name, side):
        for seed, curve in zip(r["seeds"], r["curve"], strict=True):
            by_seed.setdefault(seed, curve)
    return by_seed


def final_scores(name: str, side: str, seeds: list, train_missing: bool) -> np.ndarray:
    """Each seed's last evaluation, from the cache, training the missing seeds
    (or, with ``train_missing`` False, refusing them)."""
    recorded = curves(name, side)
    missing = [s for s in seeds if s not in recorded]
    if missing and not train_missing:
        raise LookupError(f"{name} {side}: no runs of seeds {missing}")
    if missing:
        record = train(name, side, missing)
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(CACHE / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        recorded.update(zip(missing, record["curve"], strict=True))
    return np.array([recorded[s][-1] for s in seeds])


def decide(name: str, arms=("matched",), train_missing: bool = True):
    """AdaStop on the final returns, each arm against the reference: the
    test and each side's scores, in seed order."""
    n, K, alpha = COMPARISONS[name].adastop
    sides = ["reference", *arms]
    pairs = [(arm, "reference") for arm in arms]
    test = AdaStop(sides, n=n, K=K, alpha=alpha, comparisons=pairs)
    scores: dict = {side: [] for side in sides}
    while not test.finished:
        seeds = list(range(test.k * n, (test.k + 1) * n))
        new = {
            s: final_scores(name, s, seeds, train_missing)
            for s in sides
            if s in test.active
        }
        for side, x in new.items():
            scores[side] += x.tolist()
        test.update(new)
    return test, scores


def speed(name: str, side: str) -> dict:
    """Medians over the side's current batches."""
    steps = COMPARISONS[name].n_timesteps
    batches = current(name, side)
    per_million = [1e6 * b["speed"]["cpu_s"] / len(b["seeds"]) / steps for b in batches]
    median = lambda key: float(np.median([b["speed"][key] for b in batches]))  # noqa: E731
    return {
        "env_steps_per_s": median("env_steps_per_s"),
        "cpu_s_per_1m_steps": float(np.median(per_million)),
        "compile_s": median("compile_s"),
    }


def summary(name: str, arms=("matched",), reps: int = 200) -> list:
    """One row per arm: AdaStop's verdict, the seeds it took, the arm's mean
    minus the reference's with its 95% interval, the smallest gap the budget
    resolves (80% power), and each side's speed (medians over batches)."""
    test, scores = decide(name, arms, train_missing=False)
    n, K, alpha = COMPARISONS[name].adastop
    rows = []
    for arm in arms:
        a, ref = scores[arm], scores["reference"]
        gap, interval = difference(a, ref)
        rows.append(
            {
                "comparison": name,
                "arm": arm,
                "decision": test.decisions[(arm, "reference")],
                "seeds": len(a),
                "reference_mean": float(np.mean(ref)),
                "ajax_mean": float(np.mean(a)),
                "difference": gap,
                "interval": interval,
                "resolution": resolution(a, ref, n, K, alpha, reps=reps),
                "speed": {
                    "reference": speed(name, "reference"),
                    "ajax": speed(name, arm),
                },
            }
        )
    return rows


if __name__ == "__main__":
    command, *names = sys.argv[1:]
    if "--cache" in names:  # another record directory: a fresh reproduction
        i = names.index("--cache")
        CACHE = pathlib.Path(names.pop(i + 1)).resolve()
        names.pop(i)
    if command == "run":
        test, scores = decide(names[0], tuple(names[1:]) or ("matched",))
        print(test.decisions, {side: len(x) for side, x in scores.items()})
    elif command == "report":
        for name in names or sorted(p.stem for p in CACHE.glob("*.jsonl")):
            for row in summary(name):
                print(json.dumps(row))
