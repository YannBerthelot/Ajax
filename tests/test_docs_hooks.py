"""Contract of the documentation build hook (docs/hooks.py) for versioned
docs: a site built for a release's tag (docs-deploy.yml names it in
AJAX_DOCS_REF) sends its readers to the source as that release has it, where
GitHub shows a tag's file at blob/<tag>/<path> and cannot edit it; main's site,
the default, links to main and edits there. The expected URLs follow GitHub's
URL scheme, written out by hand."""

from __future__ import annotations

import pathlib
import runpy
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "https://github.com/YannBerthelot/Ajax"


def _hooks() -> types.SimpleNamespace:
    """The hook read afresh, as MkDocs loads it, after AJAX_DOCS_REF is set."""
    return types.SimpleNamespace(**runpy.run_path(str(ROOT / "docs" / "hooks.py")))


class _Config(dict):
    """MkDocs's config, read both as a mapping and by attribute."""

    __getattr__ = dict.__getitem__


def _built_for(monkeypatch, ref: str | None) -> tuple[types.SimpleNamespace, _Config]:
    if ref is None:
        monkeypatch.delenv("AJAX_DOCS_REF", raising=False)
    else:
        monkeypatch.setenv("AJAX_DOCS_REF", ref)
    hooks = _hooks()
    config = _Config(
        extra={},
        config_file_path=str(ROOT / "mkdocs.yml"),
        docs_dir=str(ROOT / "docs"),
    )
    return hooks, hooks.on_config(config)


@pytest.mark.parametrize(
    "ref, edit_uri",
    [(None, "edit/main/docs/"), ("v0.4.0", "blob/v0.4.0/docs/")],
    ids=["main", "release"],
)
def test_the_edit_button_opens_the_page_at_the_built_ref(monkeypatch, ref, edit_uri):
    _, config = _built_for(monkeypatch, ref)
    assert config["edit_uri"] == edit_uri


@pytest.mark.parametrize(
    "ref, url",
    [
        (None, f"{REPO}/blob/main/src/ajax/agents/loop.py"),
        ("v0.4.0", f"{REPO}/blob/v0.4.0/src/ajax/agents/loop.py"),
    ],
    ids=["main", "release"],
)
def test_a_link_to_a_source_file_opens_it_at_the_built_ref(monkeypatch, ref, url):
    hooks, config = _built_for(monkeypatch, ref)
    page_file = types.SimpleNamespace(
        abs_src_path=str(ROOT / "docs" / "developer" / "architecture.md")
    )
    page = types.SimpleNamespace(file=page_file)
    markdown = "The [loop](../../src/ajax/agents/loop.py) runs it."
    got = hooks.on_page_markdown(markdown, page, config, files=None)
    assert got == f"The [loop]({url}) runs it."
