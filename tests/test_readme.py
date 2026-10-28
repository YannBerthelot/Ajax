"""The README is checked, not merely written.

It is the first page a reader sees and its examples are the first code they
run, so each ``python`` block is executed; a block whose first line is
``# doc: skip`` (one that would train for minutes, say) is left out. A block
that trains (calls ``.train(``; the quickstart's PPO takes 1,920 gradient
steps) runs in the slow tier, word for word. The agent list and the
relative links are checked too: both go stale quietly when an agent or a
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

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
DOCS = ROOT / "docs"

BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
# A Markdown link target or an HTML href, up to an anchor, a quote or a space.
LINK = re.compile(r'\]\(([^)\s#]+)|href="([^"#]+)"')
# The README's "## Agents" section, and each agent's link in it: "[SAC](docs/algorithms/sac.md)".
AGENTS_SECTION = re.compile(r"^## Agents\n(.*?)^## ", re.MULTILINE | re.DOTALL)
AGENT_LINK = re.compile(r"\[(\w+)\]\(docs/algorithms/")
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
        for i, code in enumerate(BLOCK.findall(README.read_text()))
        if not code.lstrip().startswith("# doc: skip")
    ]


@pytest.mark.parametrize("code", _runnable_blocks())
def test_readme_example_runs(code, tmp_path, monkeypatch):
    """Every runnable example executes without raising, in a scratch
    directory so that nothing it writes lands in the checkout."""
    monkeypatch.chdir(tmp_path)
    exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})


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


def test_readme_relative_links_resolve():
    """Each link into the repository (a docs page, LICENSE, ...) names a file
    that exists."""
    targets = {
        a or b
        for a, b in LINK.findall(README.read_text())
        if not re.match(r"[a-z]+:", a or b)
    }
    assert targets, "no relative links found: has the link pattern drifted?"
    missing = sorted(t for t in targets if not (ROOT / t).exists())
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
