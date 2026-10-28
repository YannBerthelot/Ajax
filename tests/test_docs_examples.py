"""Every Python example of the documentation runs.

A reader runs a page's examples in order, each building on the ones before,
so a page's ``python`` blocks run in order in one namespace. A block whose
first line is ``# doc: skip`` is left out, as in the README
(tests/test_readme.py). A page with a block that trains (``.train(`` or
``.train_offline(``) runs in the slow tier, word for word; the others in the
fast tier. docs/world_models/ holds the reference implementations' code,
quoted beside Ajax's, not examples.
"""

from __future__ import annotations

import pathlib
import re
import sys
import types

import pytest

from tests.test_readme import BLOCK, ROOT

DOCS = ROOT / "docs"
SKIP = "# doc: skip"
TRAINS = re.compile(r"\.train(_offline)?\(")


def runnable_blocks(page: pathlib.Path) -> list[str]:
    return [
        code
        for code in BLOCK.findall(page.read_text())
        if not code.lstrip().startswith(SKIP)
    ]


def pages() -> list:
    params = []
    for page in sorted(DOCS.rglob("*.md")):
        if "world_models" in page.relative_to(DOCS).parts:
            continue
        blocks = runnable_blocks(page)
        if not blocks:
            continue
        trains = any(TRAINS.search(code) for code in blocks)
        params.append(
            pytest.param(
                page,
                id=page.relative_to(DOCS).as_posix(),
                marks=[pytest.mark.slow] if trains else [],
            )
        )
    return params


@pytest.mark.parametrize("page", pages())
def test_a_pages_examples_run_in_order(page, tmp_path, monkeypatch):
    """Each block executes without raising, after the page's blocks before
    it, in a scratch directory so that nothing it writes lands in the
    checkout."""
    monkeypatch.chdir(tmp_path)
    # A module of its own, as a script is: a dataclass finds its module's
    # names (string annotations) through sys.modules.
    module = types.ModuleType("__docs__")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for index, code in enumerate(runnable_blocks(page)):
        exec(compile(code, f"{page}, block {index}", "exec"), module.__dict__)
