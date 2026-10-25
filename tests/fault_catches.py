"""Which planted faults the fast tier catches (the two-tier test plan).

A probe family may leave the pull-request gate once the fast tests catch
every fault its catalogue lists (tests/probing/faults/<family>.jsonl). This
measures it: one run of the fast tests under coverage records which test
executes which source line (pytest-cov's per-test contexts); then each fault
is planted in a scratch copy of the repository (the catalogue's own
``faults.plant``) and the fast tests executing one of its lines run there,
cheapest first, stopping at the first failure: a failure is a catch::

    JAX_PLATFORMS=cpu uv run python -m tests.fault_catches SCRATCH [FAMILY ...]

It writes SCRATCH/catches.jsonl (a line per fault) and prints each family's
rate. Coverage sees a line only in the test that traced it, so a test reusing
a program traced earlier is not selected: the rates are lower bounds.
"""

from __future__ import annotations

import concurrent.futures
import glob
import json
import os
import subprocess
import sys
from collections import Counter

from tests import shards
from tests.probing import faults

REPO = faults.REPO
#: The fast tier: tests/conftest.py marks tests/probing slow but its harness checks.
FAST = ["-m", "not slow", "tests"]


def families() -> dict[str, str]:
    """Each fault's family: the catalogue file that lists it."""
    out = {}
    for path in sorted(glob.glob(os.path.join(faults.HERE, "faults", "*.jsonl"))):
        for line in open(path):
            if line.strip():
                out[json.loads(line)["id"]] = os.path.basename(path)[: -len(".jsonl")]
    return out


def fault_lines(text: str, old: str) -> list[int]:
    """The 1-based lines of the one occurrence of ``old`` in ``text``."""
    first = text[: text.index(old)].count("\n") + 1
    return list(range(first, first + max(1, old.count("\n"))))


def covering(contexts_by_line: dict[int, list[str]], lines: list[int]) -> set[str]:
    """The node ids whose context ("nodeid|run", "|setup", ...) ran a line."""
    return {
        c.rsplit("|", 1)[0] for ln in lines for c in contexts_by_line.get(ln, []) if c
    }


def record(scratch: str) -> str:
    """Run the fast tests once, contexts per test; return the data file."""
    data = os.path.join(scratch, ".coverage")
    env = dict(os.environ, COVERAGE_FILE=data)
    cmd = [sys.executable, "-m", "pytest", "-q", "-n", "auto", "-p", "no:cacheprovider"]
    cmd += ["--cov=src/ajax", "--cov-context=test", "--cov-report=", *FAST]
    subprocess.run(cmd, cwd=REPO, env=env, check=False)
    return data


def failed_nodes(stdout: str) -> list[str]:
    """The tests pytest's short summary names: failed, or errored in setup (a
    fault that breaks a fixture is caught too)."""
    lines = stdout.splitlines()
    return [ln.split()[1] for ln in lines if ln.startswith(("FAILED ", "ERROR "))]


def catch(fault: faults.Fault, nodes: list[str], scratch: str) -> dict:
    root = os.path.join(scratch, fault.id)
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"), JAX_PLATFORMS="cpu")
    env["AJAX_JAX_COMPILE_CACHE_DIR"] = os.path.join(scratch, "jax-cache")
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *nodes]
    with faults.scratch_copy(REPO, root, os.listdir(REPO)):
        faults.plant(fault, root)
        run = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    failed = failed_nodes(run.stdout)
    first_fail = failed[0] if failed else None
    return {"caught": run.returncode not in (0, 5), "first_fail": first_fail}


def main(scratch: str, wanted: list[str]) -> None:
    import coverage

    os.makedirs(scratch, exist_ok=True)
    data_file = os.path.join(scratch, ".coverage")
    data = coverage.CoverageData(
        data_file if os.path.exists(data_file) else record(scratch)
    )
    data.read()
    durations = shards.load()
    family = families()
    todo = {i: f for i, f in faults.load().items() if not wanted or family[i] in wanted}
    plans = {}
    for fid, fault in todo.items():
        path = os.path.join(REPO, fault.file)
        lines = fault_lines(open(path).read(), fault.old)
        nodes = covering(data.contexts_by_lineno(path) or {}, lines)
        plans[fid] = sorted(nodes, key=lambda n: (durations.get(n, shards.DEFAULT), n))
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futures = {
            i: pool.submit(catch, todo[i], n, scratch) for i, n in plans.items() if n
        }
    results = {i: f.result() for i, f in futures.items()}
    with open(os.path.join(scratch, "catches.jsonl"), "w") as out:
        for fid in todo:
            got = results.get(fid, {"caught": False, "first_fail": None})
            row = {"id": fid, "family": family[fid], "selected": len(plans[fid]), **got}
            out.write(json.dumps(row) + "\n")
    caught = Counter(family[i] for i, r in results.items() if r["caught"])
    for name, total in sorted(Counter(family[i] for i in todo).items()):
        print(f"{name:14s} {caught[name]:3d} / {total:3d} caught by the fast tier")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
