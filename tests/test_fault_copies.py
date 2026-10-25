"""A planted fault's scratch copy (tests/probing/faults.py) leaves out the
Python environments and is removed once its run ends."""

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
    assert faults.run(None, str(tmp_path)) == 1
    assert list(tmp_path.iterdir()) == []
