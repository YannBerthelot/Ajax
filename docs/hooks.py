"""MkDocs build hooks: an include and a link rewrite, both before MkDocs
resolves links.

**Include.** The contributing guide lives at the repository root, where GitHub
shows it. ``docs/developer/contributing.md`` names it with a ``--8<--`` line,
which this hook replaces with the file. ``pymdownx.snippets`` would do that
during Markdown conversion, after this hook, too late to fix its links.

**Links.** Pages link to source files, benchmarks and the root guides by their
repository path (``../../src/ajax/agents/loop.py``): right for a reader on
GitHub, dead on the site, which holds only ``docs/``. Each such link is
rewritten to the file on GitHub, so the sources stay right for both readers.

**Source ref.** Those links and the edit button point at main for the
development version, and at the release's tag for a release's site, so a
release's pages show the source they describe. docs-deploy.yml names the ref
in AJAX_DOCS_REF; a build without it is main's.
"""

import os
import pathlib
import re

_REPO = "https://github.com/YannBerthelot/Ajax"
_INCLUDE = re.compile(r'^--8<--\s+"([^"]+)"\s*$', re.MULTILINE)
# An inline link or image target with no scheme; an anchor and title may follow.
_LINK = re.compile(r'(\]\()([^)\s#:]+)((?:#[^)\s]*)?(?:\s+"[^"]*")?\))')


def on_config(config):
    ref = os.environ.get("AJAX_DOCS_REF", "main")
    config.extra["source_ref"] = ref
    # A tag's pages can be read on GitHub, not edited.
    action = "edit" if ref == "main" else "blob"
    config["edit_uri"] = f"{action}/{ref}/docs/"
    return config


def _rewrite(markdown, base, page_dir, root, docs, files, ref):
    """Retarget the links of text written in ``base`` for the page in ``page_dir``."""

    def sub(match):
        target = (base / match.group(2)).resolve()
        if not (target.exists() and target.is_relative_to(root)):
            return match.group(0)
        if target.is_relative_to(docs):
            site = files.get_file_from_path(target.relative_to(docs).as_posix())
            if site is not None and not site.inclusion.is_excluded():
                link = pathlib.Path(os.path.relpath(target, page_dir)).as_posix()
                return f"{match.group(1)}{link}{match.group(3)}"
        kind = "tree" if target.is_dir() else "blob"
        url = f"{_REPO}/{kind}/{ref}/{target.relative_to(root).as_posix()}"
        return f"{match.group(1)}{url}{match.group(3)}"

    return _LINK.sub(sub, markdown)


def on_page_markdown(markdown, page, config, files):
    root = pathlib.Path(config.config_file_path).parent.resolve()
    docs = pathlib.Path(config.docs_dir).resolve()
    page_dir = pathlib.Path(page.file.abs_src_path).resolve().parent
    ref = config.extra["source_ref"]

    def include(match):
        target = root / match.group(1)
        if not target.is_file():
            raise FileNotFoundError(
                f"{page.file.src_uri}: included file not found: {match.group(1)}"
            )
        text = target.read_text()
        return _rewrite(text, target.parent, page_dir, root, docs, files, ref)

    markdown = _INCLUDE.sub(include, markdown)
    return _rewrite(markdown, page_dir, page_dir, root, docs, files, ref)
