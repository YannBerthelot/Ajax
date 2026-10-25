"""Planted faults: one catalogue line per fault, one runner.

``faults/<module>.jsonl`` holds the faults a scenario module catches (an
edit others catch too: once, all its catches listed): an ``id``, a ``path``
under src/ajax, the ``old`` text (exactly one occurrence), its ``new``
replacement, ``catches``, the nodes the old measured record shows failing
with it planted, and ``fast``, the fast tests measured to catch it
(``python -m tests.fault_catches``), which every fault lists: the
pull-request gate runs the fast tier alone. ``run`` copies the repository to a scratch
directory, plants the fault there and runs those nodes, so the checkout is
never modified, then removes the copy; a non-zero exit means caught::

    JAX_PLATFORMS=cpu uv run python -m tests.probing.faults SCRATCH ID [ID ...]
"""

from __future__ import annotations

import contextlib
import dataclasses
import glob
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
COPIED = ("src", "tests", "benchmarks", "docs", "pyproject.toml")
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

    @property
    def file(self) -> str:
        return os.path.join("src", "ajax", self.path)


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


def run(fault: Fault | None, scratch: str, *pytest_args: str) -> int:
    """Run the fault's catches (or pytest_args) on a scratch copy with the fault
    planted (None: the clean copy)."""
    root = os.path.join(scratch, fault.id if fault else "clean")
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"), JAX_PLATFORMS="cpu")
    env["AJAX_NO_COMPILE_CACHE"] = "1"
    probe = [sys.executable, "-c", "import ajax; print(ajax.__file__)"]
    nodes = [f"tests/probing/{n}" for n in fault.catches] if fault else []
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *nodes]
    cmd += pytest_args
    with scratch_copy(REPO, root, COPIED):
        if fault is not None:
            plant(fault, root)
        where = subprocess.run(probe, env=env, cwd=root, capture_output=True, text=True)
        if not where.stdout.strip().startswith(root):
            raise RuntimeError(f"imported {where.stdout.strip()}, not the copy")
        return subprocess.run(cmd, env=env, cwd=root).returncode


if __name__ == "__main__":
    catalogue = load()
    codes = {i: run(catalogue[i], sys.argv[1]) for i in sys.argv[2:]}
    print({i: "caught" if code else "SURVIVED" for i, code in codes.items()})
