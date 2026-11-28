"""A GitHub Release's notes: the version's CHANGELOG.md entry, each paragraph
and list item on one line. GitHub's release page breaks a line wherever the
source does, so the entry's wrapping at 76 columns broke its sentences.

    python3 .github/scripts/release_notes.py 0.6.0 > notes.md
"""

import re
import sys
from pathlib import Path

#: A line that starts a block of its own: a list item, a heading, a table row.
BLOCK_START = re.compile(r"\s*([-*+] |\d+\. |#|\|)")


def entry(changelog: str, version: str) -> list[str]:
    """The lines between the "## <version>" heading and the next "## " one."""
    lines = changelog.splitlines()
    start = lines.index(f"## {version}") + 1
    ends = [i for i in range(start, len(lines)) if lines[i].startswith("## ")]
    return lines[start : ends[0] if ends else len(lines)]


def unwrap(lines: list[str]) -> str:
    """Joins each line that continues a paragraph or list item to the line
    before; blank lines, headings, table rows and code blocks stay as they
    are."""
    joined: list[str] = []
    in_code = continues = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code, continues = not in_code, False
            joined.append(line)
        elif in_code or not line.strip() or BLOCK_START.match(line):
            joined.append(line)
            text = line.lstrip()
            continues = not in_code and bool(text) and not text.startswith(("#", "|"))
        elif continues:
            joined[-1] += " " + line.strip()
        else:
            joined.append(line)
            continues = True
    return "\n".join(joined).strip("\n") + "\n"


def main(version: str) -> None:
    changelog = Path("CHANGELOG.md").read_text()
    if f"## {version}" not in changelog.splitlines():
        sys.exit(f"CHANGELOG.md has no '## {version}' entry")
    notes = unwrap(entry(changelog, version))
    if not notes.strip():
        sys.exit(f"CHANGELOG.md's '## {version}' entry is empty")
    sys.stdout.write(notes)


if __name__ == "__main__":
    main(sys.argv[1])
