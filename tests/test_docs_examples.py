"""Every Python example of the documentation runs.

A reader runs a page's examples in order, each building on the ones before,
so a page's ``python`` blocks run in order in one namespace. A block whose
first line is ``# doc: skip`` is left out, as in the README
(tests/test_readme.py). A page with a block that trains (``.train(`` or
``.train_offline(``) runs in the slow tier, word for word; the others in the
fast tier. tests/doc_examples.py says which blocks run and runs them; the
Release workflow runs it as a script against the built wheel.
"""

from __future__ import annotations

import re

import pytest

from tests.doc_examples import ROOT, example_pages, main, run_page, runnable_blocks

DOCS = ROOT / "docs"
TRAINS = re.compile(r"\.train(_offline)?\(")


def pages() -> list:
    return [
        pytest.param(
            page,
            id=page.relative_to(DOCS).as_posix(),
            marks=[pytest.mark.slow]
            if any(TRAINS.search(code) for code in runnable_blocks(page))
            else [],
        )
        for page in example_pages(DOCS)
    ]


@pytest.mark.parametrize("page", pages())
def test_a_pages_examples_run_in_order(page, tmp_path, monkeypatch):
    """Each block executes without raising, after the page's blocks before
    it, in a scratch directory so that nothing it writes lands in the
    checkout."""
    monkeypatch.chdir(tmp_path)
    run_page(page)


FAILS = "```python\nraise ValueError\n```\n"
WRITES = "```python\nopen('written', 'w').close()\n```\n"
SKIPPED = "```python\n# doc: skip\nraise ValueError\n```\n"


def test_the_script_fails_naming_each_failing_example(tmp_path, capsys, monkeypatch):
    """As the release gate runs it, from the checkout: a failing README
    block and a failing page make the exit status non-zero and are named; a
    skipped block and docs/world_models/ do not run; nothing is written in
    the checkout."""
    monkeypatch.chdir(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(WRITES + SKIPPED + FAILS)
    (tmp_path / "docs" / "guide").mkdir(parents=True)
    (tmp_path / "docs" / "world_models").mkdir()
    (tmp_path / "docs" / "guide" / "broken.md").write_text(WRITES + FAILS)
    (tmp_path / "docs" / "fine.md").write_text(WRITES + WRITES)
    (tmp_path / "docs" / "world_models" / "quoted.md").write_text(FAILS)

    status = main([readme, tmp_path / "docs"])

    assert status == 1
    last_line = capsys.readouterr().out.splitlines()[-1]
    assert last_line == "failed: README.md block2, docs/guide/broken.md"
    assert not list(tmp_path.rglob("written"))


def test_the_script_passes_when_every_example_runs(tmp_path):
    """A page's later block sees the names its earlier blocks defined."""
    readme = tmp_path / "README.md"
    readme.write_text("```python\nx = 1\n```\n")
    (tmp_path / "docs").mkdir()
    page = "```python\nx = 1\n```\n```python\nassert x == 1\n```\n"
    (tmp_path / "docs" / "page.md").write_text(page)
    assert main([readme, tmp_path / "docs"]) == 0
