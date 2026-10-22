"""The README is checked, not merely written.

It is the first page a reader sees and its examples are the first code they
run, so each ``python`` block is executed; a block whose first line is
``# doc: skip`` (one that would train for minutes, say) is left out. A block
that trains (calls ``.train(``; the quickstart's PPO takes 1,920 gradient
steps) runs in the slow tier, word for word. The agent list and the
relative links are checked too: both go stale quietly when an agent or a
documentation page is added, renamed or removed.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import ajax

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

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
