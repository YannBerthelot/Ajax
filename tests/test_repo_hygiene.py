"""Repository-level invariants that CI must enforce: filesystem case,
which a Linux-only CI is otherwise blind to, and the test layout."""

import subprocess
from collections import defaultdict
from pathlib import Path


def _tracked() -> list[str]:
    """The paths git tracks, relative to the repository root."""
    return subprocess.run(
        ["git", "ls-files"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()


def test_no_case_insensitive_path_collisions():
    """No two tracked paths may differ only by case.

    On case-insensitive filesystems (macOS APFS default, Windows NTFS)
    such paths collapse onto a single file at checkout, silently
    corrupting the working tree. This happened once with
    ``SAC/SAC.py`` vs ``SAC/sac.py`` (PR #32) — Linux CI cannot see it.
    """
    by_folded = defaultdict(list)
    for path in _tracked():
        by_folded[path.lower()].append(path)

    collisions = {k: v for k, v in by_folded.items() if len(v) > 1}
    assert not collisions, (
        f"Tracked paths collide on case-insensitive filesystems: {collisions}"
    )


def test_every_test_directory_is_a_package():
    """Every directory of ``tests/`` holding Python files is a package, so
    pytest imports each module as ``tests.<dir>.<module>``, the name shared
    helpers and the probing scripts import it by. pytest puts a directory
    without ``__init__.py`` on ``sys.path`` itself, and its modules become
    top-level names that can collide (two ``common.py``)."""
    tracked = _tracked()
    missing = sorted(
        {
            str(Path(path).parent)
            for path in tracked
            if path.startswith("tests/") and path.endswith(".py")
        }
        - {str(Path(path).parent) for path in tracked if path.endswith("/__init__.py")}
    )
    assert not missing, f"test directories without __init__.py: {missing}"
