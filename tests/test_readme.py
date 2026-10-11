"""The README is checked, not merely written.

It is the first page a reader sees and its examples are the first code they
run, so each ``python`` block is executed; a block whose first line is
``# doc: skip`` (one that would train for minutes, say) is left out. The agent
table and the relative links are checked too: both go stale quietly when an
agent or a documentation page is added, renamed or removed.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import pytest

import ajax

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

BLOCK = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)
# A Markdown link target or an HTML href, up to an anchor, a quote or a space.
LINK = re.compile(r'\]\(([^)\s#]+)|href="([^"#]+)"')
# The agent column of the README's agent table: "| [`SAC`](docs/...) |".
AGENT_ROW = re.compile(r"^\| \[`(\w+)`\]\(", re.MULTILINE)
# The agents ``ajax`` does not export, with the module the README names.
NOT_EXPORTED = {"TD3": "ajax.agents.TD3.TD3", "UDRL": "ajax.agents.UDRL.UDRL"}
NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve"
    " thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()


def _runnable_blocks() -> list:
    return [
        pytest.param(code, id=f"block{i}")
        for i, code in enumerate(BLOCK.findall(README.read_text()))
        if not code.lstrip().startswith("# doc: skip")
    ]


@pytest.mark.parametrize("code", _runnable_blocks())
def test_readme_example_runs(code, tmp_path, monkeypatch):
    """Every runnable example executes without raising, in a scratch
    directory so that nothing it writes lands in the checkout."""
    monkeypatch.chdir(tmp_path)
    exec(compile(code, str(README), "exec"), {"__name__": "__readme__"})


def test_readme_agent_table_is_complete():
    """The table lists exactly the agents Ajax has, each importable where the
    README says, and the prose count matches the table."""
    readme = README.read_text()
    listed = AGENT_ROW.findall(readme)
    assert len(listed) == len(set(listed)), f"an agent is listed twice: {listed}"
    assert set(listed) == set(ajax.__all__) | set(NOT_EXPORTED), (
        f"README agents {sorted(listed)} differ from the package's"
        f" {sorted(set(ajax.__all__) | set(NOT_EXPORTED))}"
    )
    for name, module in NOT_EXPORTED.items():
        assert name not in ajax.__all__, f"{name} is exported now: update the README"
        assert f"`{module}`" in readme
        assert hasattr(importlib.import_module(module), name)
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
