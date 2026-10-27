"""Planted faults: one catalogue line per fault, one runner.

``faults/<module>.jsonl`` holds the faults a scenario module catches (an
edit others catch too: once, all its catches listed): an ``id``, a ``path``
under src/ajax, the ``old`` text (exactly one occurrence), its ``new``
replacement, ``catches``, the probe nodes that catch it (none where no probe
can see it), and ``fast``, the fast tests that catch it (``python -m
tests.fault_catches`` finds them), which every fault lists: the pull-request
gate runs the fast tier alone. Nodes are named with their parameters. ``run``
plants the fault in a scratch copy of the repository (the checkout is never
modified), runs those nodes there and removes the copy; ``missed`` names those
that did not catch it, and the command prints OK or CHECK per fault::

    JAX_PLATFORMS=cpu uv run python -m tests.probing.faults SCRATCH ID [ID ...]

A node catches a fault by failing on an assertion (CAUGHT), not by crashing:
a probe that crashed measured nothing (its checks on the library assert), and
a fast test may crash only because the planted text meets its stand-in state,
which the bug would not. ``raises`` names the error a fault's fast tests may
fail with instead where the bug cannot run in their setting: TD-MPC2's
multi-task tests swap tasks of different shapes (q10's tasks share one).
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from subprocess import CompletedProcess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
COPIED = ("src", "tests", "benchmarks", "docs", "pyproject.toml")
#: How a node catching a fault fails: an assert (pytest names a bare one
#: "assert"), or pytest's Failed (a refusal pytest.raises expected did not come).
CAUGHT = ("AssertionError", "assert", "Failed")
#: Never copied, at any depth: the run needs none of them, and the reference
#: implementations' environments (benchmarks/reference/envs/*/.venv) are 1-3.5 GB.
SKIPPED = frozenset({".git", ".venv", ".claude", "site", "__pycache__"}).union(
    {".mypy_cache", ".pytest_cache", ".ruff_cache"}
)


@dataclasses.dataclass(frozen=True)
class Fault:
    id: str
    path: str
    old: str
    new: str
    catches: tuple[str, ...]
    fast: tuple[str, ...] = ()
    raises: str = ""

    @property
    def file(self) -> str:
        return os.path.join("src", "ajax", self.path)

    @property
    def nodes(self) -> list[str]:
        return [f"tests/probing/{n}" for n in self.catches] + list(self.fast)


def load() -> dict[str, Fault]:
    """Every catalogue's faults by id (ids are unique across modules)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "faults", "*.jsonl"))):
        with open(path) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    lists = {"catches": (), "fast": ()}
    faults = {
        r["id"]: Fault(**{**r, **{k: tuple(r.get(k, d)) for k, d in lists.items()}})
        for r in rows
    }
    if len(faults) != len(rows):
        raise RuntimeError("duplicate fault ids")
    return faults


def plant(fault: Fault, root: str) -> None:
    with open(os.path.join(root, fault.file)) as fh:
        text = fh.read()
    if text.count(fault.old) != 1:
        raise RuntimeError(
            f"{fault.id}: {text.count(fault.old)} matches in {fault.file}"
        )
    with open(os.path.join(root, fault.file), "w") as fh:
        fh.write(text.replace(fault.old, fault.new))


@contextlib.contextmanager
def scratch_copy(repo: str, root: str, names: Iterable[str]) -> Iterator[None]:
    """Copy the named files and directories of repo into root, and remove the
    copy on exit, whether the run inside passed, failed or raised."""
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    skip = shutil.ignore_patterns(*SKIPPED)
    try:
        for name in set(names).difference(SKIPPED):
            source, target = os.path.join(repo, name), os.path.join(root, name)
            if os.path.isdir(source):
                shutil.copytree(source, target, ignore=skip)
            else:
                shutil.copy2(source, target)
        yield
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run(fault: Fault | None, scratch: str, *pytest_args: str) -> CompletedProcess:
    """pytest, its output captured, on a scratch copy with the fault planted
    (None: the clean copy): the fault's nodes, then pytest_args."""
    root = os.path.join(scratch, fault.id if fault else "clean")
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"), JAX_PLATFORMS="cpu")
    env.update(AJAX_NO_COMPILE_CACHE="1", COLUMNS="400")  # summary lines whole
    probe = [sys.executable, "-c", "import ajax; print(ajax.__file__)"]
    nodes = fault.nodes if fault else []
    if fault is not None and not nodes:
        raise ValueError(f"{fault.id} lists no node")
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-rfEp"]
    cmd += ["--tb=line", *nodes, *pytest_args]
    cmd += ["-n", str(min(len(nodes), 4))] if len(nodes) > 1 else []
    with scratch_copy(REPO, root, COPIED):
        if fault is not None:
            plant(fault, root)
        where = subprocess.run(probe, env=env, cwd=root, capture_output=True, text=True)
        if not where.stdout.strip().startswith(root):
            raise RuntimeError(f"imported {where.stdout.strip()}, not the copy")
        return subprocess.run(cmd, env=env, cwd=root, capture_output=True, text=True)


def missed(fault: Fault, stdout: str) -> dict[str, str]:
    """Each of the fault's nodes that did not catch it, and how it ended
    (pytest's summary): PASSED, not run, or the error it crashed with."""
    summary = re.findall(r"^(PASSED|FAILED|ERROR) (\S+)(?: - (\w+))?", stdout, re.M)
    ended = {node: error or outcome for outcome, node, error in summary}
    ends = {node: ended.get(node, "not run") for node in fault.nodes}
    excused = {node: fault.raises for node in fault.fast if fault.raises}
    return {n: e for n, e in ends.items() if e not in CAUGHT and e != excused.get(n)}


if __name__ == "__main__":
    catalogue = load()
    for i in sys.argv[2:]:
        missing = missed(catalogue[i], run(catalogue[i], sys.argv[1]).stdout)
        print(f"{i}: CHECK {missing}" if missing else f"{i}: OK", flush=True)
