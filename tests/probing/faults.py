"""Planted faults: one catalogue line per fault, one runner.

``faults/<module>.jsonl`` holds the faults a scenario module catches (an
edit others catch too: once, all its catches listed): an ``id``, a ``path``
under src/ajax, the ``old`` text (exactly one occurrence), its ``new``
replacement, and ``catches``, the nodes the old measured record shows
failing with it planted. ``run`` copies the repository to a scratch
directory, plants the fault there and runs those nodes, so the checkout is
never modified; a non-zero exit means caught::

    JAX_PLATFORMS=cpu poetry run python -m tests.probing.faults SCRATCH ID [ID ...]
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
COPIED = ("src", "tests", "benchmarks", "docs", "pyproject.toml")


@dataclasses.dataclass(frozen=True)
class Fault:
    id: str
    path: str
    old: str
    new: str
    catches: tuple[str, ...]

    @property
    def file(self) -> str:
        return os.path.join("src", "ajax", self.path)


def load() -> dict[str, Fault]:
    """Every catalogue's faults by id (ids are unique across modules)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "faults", "*.jsonl"))):
        with open(path) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    faults = {r["id"]: Fault(**{**r, "catches": tuple(r["catches"])}) for r in rows}
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


def run(fault: Fault | None, scratch: str, *pytest_args: str) -> int:
    """Run ``fault.catches`` (or ``pytest_args``) on a copy with ``fault``
    planted (``None``: the clean copy)."""
    root = os.path.join(scratch, fault.id if fault else "clean")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for name in COPIED:
        src, dst = os.path.join(REPO, name), os.path.join(root, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)
    if fault is not None:
        plant(fault, root)
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"), JAX_PLATFORMS="cpu")
    env["AJAX_NO_COMPILE_CACHE"] = "1"
    probe = [sys.executable, "-c", "import ajax; print(ajax.__file__)"]
    where = subprocess.run(probe, env=env, cwd=root, capture_output=True, text=True)
    if not where.stdout.strip().startswith(root):
        raise RuntimeError(f"imported {where.stdout.strip()}, not the copy")
    nodes = [f"tests/probing/{n}" for n in fault.catches] if fault else []
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *nodes]
    cmd += pytest_args
    return subprocess.run(cmd, env=env, cwd=root).returncode


if __name__ == "__main__":
    catalogue = load()
    codes = {i: run(catalogue[i], sys.argv[1]) for i in sys.argv[2:]}
    print({i: "caught" if code else "SURVIVED" for i, code in codes.items()})
