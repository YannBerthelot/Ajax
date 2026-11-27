"""The README is checked, not merely written.

It is the first page a reader sees and its examples are the first code they
run, so each ``python`` block is executed; a block whose first line is
``# doc: skip`` (one that would train for minutes, say) is left out. A block
that trains (calls ``.train(``) runs in the slow tier, word for word. The agent list and the
links are checked too: both go stale quietly when an agent or a
documentation page is added, renamed or removed. So are the totals the
documentation states (agents, extensions), the algorithms index, the
extensions catalogue and the memory table, against the code.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import ajax
from tests.agents.test_compatibility import extension_classes
from tests.doc_examples import ROOT, readme_blocks, run_readme_block

README = ROOT / "README.md"
DOCS = ROOT / "docs"

# A Markdown link or image target, or an HTML href or src, up to an anchor, a
# quote or a space.
LINK = re.compile(r'\]\(([^)\s#]+)|(?:href|src)="([^"#]+)"')
# PyPI shows the README as the project page, where relative links and images
# break: a link into the repository takes one of these absolute forms.
REPOSITORY_FILES = (
    "https://raw.githubusercontent.com/YannBerthelot/Ajax/main/",  # an image
    "https://github.com/YannBerthelot/Ajax/blob/main/",  # a file
    "https://github.com/YannBerthelot/Ajax/tree/main/",  # a folder
)
DOCS_SITE = "https://yannberthelot.github.io/Ajax/latest/"  # a documentation page
# The README's "## Agents" section, and each agent's link in it:
# "[SAC](https://yannberthelot.github.io/Ajax/latest/algorithms/sac/)".
AGENTS_SECTION = re.compile(r"^## Agents\n(.*?)^## ", re.MULTILINE | re.DOTALL)
AGENT_LINK = re.compile(r"\[(\w+)\]\(" + re.escape(DOCS_SITE) + "algorithms/")
NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve"
    " thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}
# A table row's first cell: an agent's link, "| [SAC](sac.md) |", or an
# extension's name, "| `OnlineBC` |".
AGENT_ROW = re.compile(r"^\| \[(\w+)\]\(", re.MULTILINE)
EXTENSION_ROW = re.compile(r"^\| `(\w+)` \|", re.MULTILINE)
TABLE_ROW = re.compile(r"^\| ([^|]+) \| ([^|]+) \|$", re.MULTILINE)


def _runnable_blocks() -> list:
    return [
        pytest.param(
            code, id=f"block{i}", marks=[pytest.mark.slow] if ".train(" in code else []
        )
        for i, code in readme_blocks(README)
    ]


@pytest.mark.parametrize("code", _runnable_blocks())
def test_readme_example_runs(code, tmp_path, monkeypatch):
    """Every runnable example executes without raising, in a scratch
    directory so that nothing it writes lands in the checkout."""
    monkeypatch.chdir(tmp_path)
    run_readme_block(code, README)


def test_readme_agent_list_is_complete():
    """The Agents section lists exactly the agents ``ajax`` exports, and the
    prose count matches the list."""
    readme = README.read_text()
    section = AGENTS_SECTION.search(readme)
    assert section, "the README has no '## Agents' section"
    listed = AGENT_LINK.findall(section.group(1))
    assert len(listed) == len(set(listed)), f"an agent is listed twice: {listed}"
    assert set(listed) == set(ajax.__all__), (
        f"README agents {sorted(listed)} differ from ajax.__all__"
        f" {sorted(ajax.__all__)}"
    )
    assert f"{NUMBER_WORDS[len(listed)]} agents" in readme


def local_file(url: str) -> pathlib.Path | None:
    """The checkout's file or folder that an absolute link names; None for a
    link elsewhere. A documentation page is docs/<path>.md, or
    docs/<path>/index.md for a folder's page (the site's directory URLs)."""
    for prefix in REPOSITORY_FILES:
        if url.startswith(prefix):
            return ROOT / url.removeprefix(prefix)
    if not url.startswith(DOCS_SITE):
        return None
    page = url.removeprefix(DOCS_SITE).strip("/")
    markdown = DOCS / f"{page}.md"
    return markdown if markdown.exists() else DOCS / page / "index.md"


def test_readme_links_are_absolute_and_name_files_that_exist():
    """No link or image is relative, which PyPI's project page would break,
    and each into the repository (a docs page, an image, LICENSE, ...) names
    a file of the checkout that exists."""
    targets = {a or b for a, b in LINK.findall(README.read_text())}
    relative = sorted(t for t in targets if not re.match(r"[a-z]+:", t))
    assert not relative, f"README links relatively, which breaks on PyPI: {relative}"
    files = {t: local_file(t) for t in targets}
    repository = {t: f for t, f in files.items() if f is not None}
    assert repository, "no link into the repository found: has a pattern drifted?"
    missing = sorted(t for t, f in repository.items() if not f.exists())
    assert not missing, f"README links to missing files: {missing}"


def in_words(n: int) -> str:
    """n as the documentation writes it: eighteen, thirty-two."""
    if n <= 20:
        return NUMBER_WORDS[n]
    tens, units = divmod(n, 10)
    return TENS[tens] + (f"-{NUMBER_WORDS[units]}" if units else "")


def section(path: pathlib.Path, heading: str) -> str:
    """The text under a "## heading", up to the next one."""
    text = path.read_text()
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else None]


@pytest.mark.parametrize(
    "page, phrase, counted",
    [
        ("docs/index.md", "{} agents", "agents"),
        ("docs/algorithms/index.md", "{} agents", "agents"),
        ("README.md", "{} extensions", "extensions"),
        ("README.md", "{} research features", "extensions"),
        ("docs/guide/extensions.md", "ships {} of them", "extensions"),
    ],
)
def test_a_stated_total_is_the_codes(page, phrase, counted):
    """Each total the documentation keeps is the number of exported agents
    (ajax.__all__) or of extension classes (found by import, as the
    compatibility matrix finds them)."""
    total = len(ajax.__all__) if counted == "agents" else len(extension_classes())
    text = " ".join((ROOT / page).read_text().lower().split())
    assert phrase.format(in_words(total)) in text, (page, total)


def test_the_algorithms_index_lists_every_exported_agent_once():
    listed = AGENT_ROW.findall(section(DOCS / "algorithms" / "index.md", "Overview"))
    assert len(listed) == len(set(listed)), f"an agent is listed twice: {listed}"
    assert set(listed) == set(ajax.__all__)


def test_the_docs_home_page_links_every_agent_page():
    """The home page's table of agents links the page of every agent the
    algorithms index lists: REINFORCE's was missing there after it shipped."""
    home = re.findall(
        r"\]\(algorithms/(\w+)\.md", section(DOCS / "index.md", "The agents")
    )
    index = re.findall(
        r"^\| \[\w+\]\((\w+)\.md",
        section(DOCS / "algorithms" / "index.md", "Overview"),
        re.MULTILINE,
    )
    assert set(home) == set(index), sorted(set(index).symmetric_difference(home))


def test_the_extensions_catalogue_lists_every_extension_once():
    catalogue = section(DOCS / "guide" / "extensions.md", "Catalogue")
    listed = EXTENSION_ROW.findall(catalogue)
    assert len(listed) == len(set(listed)), f"an extension is listed twice: {listed}"
    assert set(listed) == set(extension_classes())


def test_the_memory_table_says_which_agents_take_memory():
    """Every exported agent is in one row; a row granting `memory=` holds
    exactly the agents that support it. DreamerV3, whose world model is
    recurrent, is the one agent listed as built in."""
    rows = TABLE_ROW.findall(section(DOCS / "guide" / "memory.md", "Which agents"))
    where = {
        name: what
        for agents, what in rows[2:]  # past the header and its rule
        for name in agents.split(", ")
    }
    names = [name for agents, _ in rows[2:] for name in agents.split(", ")]
    assert len(names) == len(set(names)), f"an agent is listed twice: {names}"
    assert set(names) == set(ajax.__all__)
    for name, what in where.items():
        supports = getattr(getattr(ajax, name), "supports_memory", False)
        assert what.startswith("`memory=`") == supports, name
        assert ("built in" in what) == (name == "DreamerV3"), name
