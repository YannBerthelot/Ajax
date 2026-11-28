"""A GitHub Release's notes (.github/scripts/release_notes.py).

Contract: the notes are the version's CHANGELOG.md entry, from its "## X.Y.Z"
heading to the next "## " one, with each paragraph and list item on one line,
since GitHub's release page breaks a line wherever the source does. Only line
breaks change: nested items keep their indentation, and blank lines,
headings, table rows and code blocks stay as they are.
"""

import os
import pathlib
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".github", "scripts"))
import release_notes

CHANGELOG_PATH = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"
VERSIONS = re.findall(r"^## (\d+\.\d+\.\d+)$", CHANGELOG_PATH.read_text(), re.M)

CHANGELOG = """\
# Changelog

## 0.7.0

- **New.** Something
  new.

## 0.6.0

What a user
of 0.5.0 meets:

- **A first run.** `train` with
  no `logging_config` now
  evaluates.
  - A nested item
    that wraps.
- Second item.

### Upgrading from 0.5.0

Text right
after a heading.

```python
x = 1
  y = 2
```

| a | b |
| - | - |

## 0.5.0

- Old.
"""


def test_each_paragraph_and_list_item_is_one_line():
    notes = release_notes.unwrap(release_notes.entry(CHANGELOG, "0.6.0"))

    assert notes == (
        "What a user of 0.5.0 meets:\n"
        "\n"
        "- **A first run.** `train` with no `logging_config` now evaluates.\n"
        "  - A nested item that wraps.\n"
        "- Second item.\n"
        "\n"
        "### Upgrading from 0.5.0\n"
        "\n"
        "Text right after a heading.\n"
        "\n"
        "```python\n"
        "x = 1\n"
        "  y = 2\n"
        "```\n"
        "\n"
        "| a | b |\n"
        "| - | - |\n"
    )


def test_a_version_without_an_entry_stops_with_a_message(tmp_path, monkeypatch):
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="no '## 9.9.9' entry"):
        release_notes.main("9.9.9")


@pytest.mark.parametrize("version", VERSIONS)
def test_the_notes_keep_every_word_of_the_entry(version):
    """Catches text dropped or doubled while joining lines, on every entry."""
    lines = release_notes.entry(CHANGELOG_PATH.read_text(), version)

    notes = release_notes.unwrap(lines)

    assert notes.split() == "\n".join(lines).split()
