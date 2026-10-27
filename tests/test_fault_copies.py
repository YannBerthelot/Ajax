"""A planted fault's run (tests/probing/faults.py): its scratch copy leaves
out the Python environments and is removed once the run ends, and a node
catches the fault only by failing on an assertion."""

import dataclasses
import os
import subprocess
from pathlib import Path

import pytest

from tests.probing import faults


def test_a_copy_leaves_out_environments_at_any_depth_and_goes_when_the_run_fails(
    tmp_path: Path,
) -> None:
    repo, root = tmp_path / "repo", tmp_path / "copy"
    (repo / "benchmarks" / "envs" / "torch" / ".venv" / "lib").mkdir(parents=True)
    (repo / ".venv").mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "module.py").write_text("")
    with pytest.raises(RuntimeError, match="the run failed"):
        with faults.scratch_copy(str(repo), str(root), os.listdir(repo)):
            copied = [path.relative_to(root).as_posix() for path in root.rglob("*")]
            raise RuntimeError("the run failed")
    assert "src/module.py" in copied and "benchmarks/envs/torch" in copied
    assert not [path for path in copied if ".venv" in path]
    assert not root.exists()


def test_a_run_leaves_nothing_in_its_scratch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def pytest_fails(cmd: list[str], env: dict, cwd: str, **kwargs: object) -> object:
        assert os.path.isdir(os.path.join(cwd, "src", "ajax"))
        # The first call asks where ajax imports from: the copy.
        where = os.path.join(cwd, "src", "ajax", "__init__.py")
        return subprocess.CompletedProcess(cmd, 1, stdout=where)

    monkeypatch.setattr(faults.subprocess, "run", pytest_fails)
    assert faults.run(None, str(tmp_path)).returncode == 1
    assert list(tmp_path.iterdir()) == []


def test_a_fault_listing_no_node_has_no_run(tmp_path: Path) -> None:
    fault = faults.Fault("unseen", "module.py", "old", "new", catches=())
    with pytest.raises(ValueError, match="unseen lists no node"):
        faults.run(fault, str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_a_node_catches_a_fault_only_by_failing_on_an_assertion() -> None:
    """A crash is no catch, but for the error an entry's fast tests may fail
    with (``raises``); a node missing from the summary did not run."""
    probes = ("p.py::t[a]", "p.py::t[b]", "p.py::t[c]", "p.py::t[d]")
    fault = faults.Fault("f", "m.py", "o", "n", probes, ("tests/f.py::u", "f.py::v"))
    summary = """PASSED tests/probing/p.py::t[a]
FAILED tests/probing/p.py::t[b] - AssertionError: 0 of 8 seeds within
FAILED tests/probing/p.py::t[c] - ValueError: a guard
FAILED tests/f.py::u - Failed: DID NOT RAISE <class 'ValueError'>
ERROR f.py::v - ValueError: Incompatible shapes for broadcasting"""
    missed = {"tests/probing/p.py::t[a]": "PASSED", "f.py::v": "ValueError"}
    missed["tests/probing/p.py::t[c]"] = "ValueError"  # a probe: never excused
    missed["tests/probing/p.py::t[d]"] = "not run"
    assert faults.missed(fault, summary) == missed
    allowed = faults.missed(dataclasses.replace(fault, raises="ValueError"), summary)
    missed.pop("f.py::v")
    assert allowed == missed
