"""Re-point the planted faults whose quoted source moved to another file (a
restructure moves code without changing it): a fault whose ``old`` text no
longer occurs exactly once in its ``path`` takes the one file under src/ajax
where it does. The faults it cannot place (their text changed, or matches in
no file or in several) are printed for a hand edit::

    uv run python -m tests.fault_requote
"""

from __future__ import annotations

import glob
import json
import os

from tests.probing import faults

SRC = os.path.join(faults.REPO, "src", "ajax")
CATALOGUES = os.path.join(faults.HERE, "faults")


def moves(src: str, catalogue: dict[str, faults.Fault]) -> tuple[dict, list]:
    """The new path of each fault whose text moved file, and the ids of those
    no single file holds."""
    texts = {}
    for path in glob.glob(os.path.join(src, "**", "*.py"), recursive=True):
        with open(path) as fh:
            texts[os.path.relpath(path, src)] = fh.read()
    moved, lost = {}, []
    for fault in catalogue.values():
        if texts.get(fault.path, "").count(fault.old) == 1:
            continue
        found = [p for p, text in texts.items() if text.count(fault.old) == 1]
        if len(found) == 1:
            moved[fault.id] = (fault.path, found[0])
        else:
            lost.append(fault.id)
    return moved, sorted(lost)


def rewrite(catalogues: str, moved: dict) -> None:
    """Each moved fault's catalogue line with its new path, nothing else."""
    for path in glob.glob(os.path.join(catalogues, "*.jsonl")):
        with open(path) as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            fid = json.loads(line)["id"]
            if fid in moved:
                old, new = moved[fid]
                lines[i] = line.replace(f'"path": "{old}"', f'"path": "{new}"', 1)
        with open(path, "w") as fh:
            fh.writelines(lines)


if __name__ == "__main__":
    moved, lost = moves(SRC, faults.load())
    rewrite(CATALOGUES, moved)
    for fid, (old, new) in sorted(moved.items()):
        print(f"{fid}: {old} -> {new}")
    print("to edit by hand:", lost)
