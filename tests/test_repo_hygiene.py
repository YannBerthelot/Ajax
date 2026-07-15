"""Repository-level invariants that CI must enforce.

These catch classes of breakage that a Linux-only CI is otherwise blind
to (e.g. filesystem case-sensitivity differences with macOS/Windows).
"""

import subprocess
from collections import defaultdict
from pathlib import Path


def test_no_case_insensitive_path_collisions():
    """No two tracked paths may differ only by case.

    On case-insensitive filesystems (macOS APFS default, Windows NTFS)
    such paths collapse onto a single file at checkout, silently
    corrupting the working tree. This happened once with
    ``SAC/SAC.py`` vs ``SAC/sac.py`` (PR #32) — Linux CI cannot see it.
    """
    repo_root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    by_folded = defaultdict(list)
    for path in tracked:
        by_folded[path.lower()].append(path)

    collisions = {k: v for k, v in by_folded.items() if len(v) > 1}
    assert not collisions, (
        "Tracked paths collide on case-insensitive filesystems:" f" {collisions}"
    )
