"""How the harness records where things were and which code ran (README.md,
"Recorded locations").

A recorded location is relative to a named root, ``<name>/relative/path``
(``<reference-checkout>/dreamerv3/agent.py``, ``<ajax-src>/ajax/__init__.py``,
``<out>/ref_s0``), never an absolute path: the provenance that matters is the
git commit of the tree the code was imported from and whether that tree had
uncommitted changes, recorded next to the location. Runs in both venvs (the
reference's Python 3.11 and Ajax's), standard library only.

Self-test: ``python -m doctest provenance.py``.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

ELSEWHERE = "<elsewhere>"


def located(path: str | os.PathLike, roots: dict[str, str | os.PathLike]) -> str:
    """``path`` as ``<name>/relative/path`` under the deepest of ``roots`` holding it.

    A path under none of the roots is recorded as ``<elsewhere>/<file name>``:
    its directory is not recorded (it would be a local path), and a provenance
    check that expects a named root fails on it. Symbolic links are resolved
    on both sides first.

    >>> located("/a/b/c.py", {"x": "/a", "y": "/a/b"})
    '<y>/c.py'
    >>> located("/a/b", {"out": "/a"})
    '<out>/b'
    >>> located("/a", {"out": "/a"})
    '<out>'
    >>> located("/ab/c.py", {"x": "/a"})
    '<elsewhere>/c.py'
    """
    p = pathlib.Path(os.path.realpath(path))
    best: tuple[int, str] | None = None
    for name, root in roots.items():
        r = pathlib.Path(os.path.realpath(root))
        if p == r or r in p.parents:
            rel = p.relative_to(r).as_posix()
            text = f"<{name}>" + ("" if rel == "." else f"/{rel}")
            if best is None or len(r.parts) > best[0]:
                best = (len(r.parts), text)
    return best[1] if best else f"{ELSEWHERE}/{p.name}"


def git_state(directory: str | os.PathLike) -> dict:
    """``{"commit": HEAD, "dirty": bool}`` of the git tree holding ``directory``.

    ``dirty``: ``git status --porcelain`` lists something under ``directory``
    (a modified, staged, deleted or untracked file; ignored files do not
    count). ``{"commit": None, "dirty": None}`` when ``directory`` is not in a
    git tree or git is unavailable.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(directory), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--", ".").strip())
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def shown(path: str | os.PathLike, repo: str | os.PathLike) -> str:
    """A directory as the analysis outputs name it: relative to the repository
    root ``repo`` when inside it, else only its name.

    >>> shown("/r/docs/x/results/round1", "/r")
    'docs/x/results/round1'
    >>> shown("/s/refrun/full", "/r")
    'full'
    """
    p = pathlib.Path(os.path.realpath(path))
    r = pathlib.Path(os.path.realpath(repo))
    return p.relative_to(r).as_posix() if r in p.parents else p.name
