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
            if f.is_file() and not {"__pycache__", ".venv"}.intersection(parts):
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


def clean_tree() -> bool:
    """No uncommitted change in the repository, the records aside: each
    batch appends to them."""
    records = [f":!{CACHE.relative_to(ROOT)}"] if CACHE.is_relative_to(ROOT) else []
    return not _run("git", "status", "--porcelain", "--", ".", *records)


#: What the caller sets that belongs to the caller's project: its virtualenv, and the uv resolution
#: settings compat.yml's lowest job exports (UV_RESOLUTION) for Ajax's own `uv run --locked`. A
#: reference subproject resolves in the mode its own uv.lock records, or --locked refuses it.
CALLER_SETTINGS = ("VIRTUAL_ENV", "UV_RESOLUTION", "UV_PRERELEASE", "UV_EXCLUDE_NEWER")


def reference_environment(project: pathlib.Path) -> pathlib.Path:
    """Where a reference subproject's environment is built: one per lock file, named after the
    subproject and the lock's hash, so every checkout on the same lock shares it (each is 0.5 to
    1 GB) and a changed lock gets its own. The root is AJAX_REFERENCE_ENVS, or else
    ~/.cache/ajax/reference-envs."""
    default = pathlib.Path.home() / ".cache" / "ajax" / "reference-envs"
    root = pathlib.Path(os.environ.get("AJAX_REFERENCE_ENVS", default))
    lock = hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest()[:12]
    return root / f"{project.name}-{lock}"


def subproject_env(project: pathlib.Path, **extra: str) -> dict[str, str]:
    """The environment of a uv run --locked --project child: the caller's, without its
    virtualenv, plus extra. A reference subproject (benchmarks/reference/envs/...) also goes
    without the rest of CALLER_SETTINGS and runs in its shared reference_environment. Ajax's own
    project keeps its resolution settings: its uv.lock is the one they resolved (the lowest
    job's lowest-direct sync)."""
    own = project.resolve() == ROOT.resolve()
    dropped = ("VIRTUAL_ENV",) if own else CALLER_SETTINGS
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    if not own:
        env["UV_PROJECT_ENVIRONMENT"] = str(reference_environment(project))
    env.update(extra)
    return env


def train(name: str, side: str, seeds: list) -> dict:
    """Train ``seeds`` of one side on the CPU; the batch's record."""
    spec, lib = COMPARISONS[name], implementation(name, side)
    project, runner, per_seed, _ = IMPLEMENTATIONS[lib]
    groups = [[s] for s in seeds] if per_seed else [seeds]
    env = subproject_env(project, JAX_PLATFORMS="cpu")
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
            "ajax_clean": clean_tree(),
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


def current(name: str, side: str, key: str | None = None) -> list:
    """The side's records of fingerprint ``key``: by default, the one of
    the code and settings now."""
    key = key or fingerprint(name, side)
    return [r for r in records(name) if (r["side"], r["fingerprint"]) == (side, key)]


def curves(name: str, side: str, key: str | None = None) -> dict:
    """Seed -> evaluation curve, from the records of ``key`` (the first run
    of a seed counts)."""
    by_seed: dict = {}
    for r in current(name, side, key):
        for seed, curve in zip(r["seeds"], r["curve"], strict=True):
            by_seed.setdefault(seed, curve)
    return by_seed


def final_scores(
    name: str, side: str, seeds: list, key: str | None = None
) -> np.ndarray:
    """Each seed's last evaluation: from the records of ``key`` (refusing
    seeds they lack), or, without ``key``, from the current ones, training
    the missing seeds."""
    recorded = curves(name, side, key)
    missing = [s for s in seeds if s not in recorded]
    if missing and key:
        raise LookupError(f"{name} {side}: no runs of seeds {missing}")
    if missing:
        record = train(name, side, missing)
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(CACHE / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        recorded.update(zip(missing, record["curve"], strict=True))
    return np.array([recorded[s][-1] for s in seeds])


def decide(name: str, arms=("matched",), keys: dict | None = None, spec=None):
    """AdaStop on the final returns, each arm against the reference: the
    test and each side's scores, in seed order. ``keys`` replays the records
    of those fingerprints (one per side) under ``spec``; without them, the
    current code and spec run, training what is missing."""
    n, K, alpha = (spec or COMPARISONS[name]).adastop
    sides = ["reference", *arms]
    pairs = [(arm, "reference") for arm in arms]
    test = AdaStop(sides, n=n, K=K, alpha=alpha, comparisons=pairs)
    scores: dict = {side: [] for side in sides}
    while not test.finished:
        seeds = list(range(test.k * n, (test.k + 1) * n))
        new = {
            s: final_scores(name, s, seeds, keys and keys[s])
            for s in sides
            if s in test.active
        }
        for side, x in new.items():
            scores[side] += x.tolist()
        test.update(new)
    return test, scores


def reported(name: str, version: int | None = None) -> tuple:
    """The spec version to report (the latest with records when None), the
    fingerprint of each side's latest batch at that version, and the
    matched arm's first record there."""
    rs = records(name)
    version = version or max(r["spec_version"] for r in rs)
    keys = {r["side"]: r["fingerprint"] for r in rs if r["spec_version"] == version}
    first = next(r for r in rs if r["fingerprint"] == keys["matched"])
    return version, keys, first


def spec_of(name: str, record: dict):
    """The spec a record ran: the current one when the hashes agree, else
    refcmp_specs.py at the record's commit, checked against the hash."""
    spec, manifest = COMPARISONS[name], record.get("manifest", {})
    if manifest.get("spec_sha256", spec_sha256(spec)) == spec_sha256(spec):
        return spec
    path = "benchmarks/reference/refcmp_specs.py"
    namespace: dict = {}
    exec(_run("git", "show", f"{manifest['ajax_commit']}:{path}"), namespace)
    old = namespace["COMPARISONS"][name]
    if namespace["spec_sha256"](old) != manifest["spec_sha256"]:
        raise LookupError(f"{name}: no spec at {manifest['ajax_commit']} has its hash")
    return old


def speed(name: str, side: str, steps: int, key: str) -> dict:
    """Medians over the side's batches of fingerprint ``key``."""
    batches = current(name, side, key)
    per_million = [1e6 * b["speed"]["cpu_s"] / len(b["seeds"]) / steps for b in batches]
    median = lambda k: float(np.median([b["speed"][k] for b in batches]))  # noqa: E731
    return {
        "env_steps_per_s": median("env_steps_per_s"),
        "cpu_s_per_1m_steps": float(np.median(per_million)),
        "compile_s": median("compile_s"),
    }


def verdict(spec, interval: tuple, reference_mean: float) -> tuple:
    """``(margin, word)``: "similar" when the gap's interval lies inside
    the margin, "different" when outside it, else "inconclusive"."""
    margin = spec.margin
    if margin is None and spec.relative_margin is not None:
        margin = spec.relative_margin * abs(reference_mean)
    if margin is None:
        return None, "no margin registered"
    low, high = interval
    if -margin <= low and high <= margin:
        return margin, "similar"
    return margin, "different" if high < -margin or low > margin else "inconclusive"


def summary(name: str, reps: int = 200, version: int | None = None) -> list:
    """The matched arm's row, from its records at spec ``version`` (the
    latest when None), judged under the spec they ran: AdaStop's decision
    and the seeds it took, the arm's mean minus the reference's with its
    95% interval, the verdict against the margin, the smallest gap the
    budget resolves (80% power), and each side's speed (medians over
    batches)."""
    version, keys, first = reported(name, version)
    spec = spec_of(name, first)
    test, scores = decide(name, keys=keys, spec=spec)
    n, K, alpha = spec.adastop
    a, ref = scores["matched"], scores["reference"]
    gap, interval = difference(a, ref)
    margin, word = verdict(spec, interval, float(np.mean(ref)))
    sides = {"reference": "reference", "ajax": "matched"}
    return [
        {
            "comparison": name,
            "arm": "matched",
            "spec_version": version,
            "ajax_commit": first.get("manifest", {}).get("ajax_commit"),
            "decision": test.decisions[("matched", "reference")],
            "seeds": len(a),
            "reference_mean": float(np.mean(ref)),
            "ajax_mean": float(np.mean(a)),
            "difference": gap,
            "interval": interval,
            "margin": margin,
            "verdict": word,
            "resolution": resolution(a, ref, n, K, alpha, reps=reps),
            "speed": {
                label: speed(name, side, spec.n_timesteps, keys[side])
                for label, side in sides.items()
            },
        }
    ]


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
