"""A planted fault whose quoted text moved file is re-pointed at it; one the
tool cannot place is listed (tests/fault_requote.py)."""

import json
from pathlib import Path

from tests import fault_requote
from tests.probing import faults


def test_a_moved_quote_is_re_pointed_and_an_ambiguous_one_listed(
    tmp_path: Path,
) -> None:
    src, catalogues = tmp_path / "src", tmp_path / "faults"
    (src / "agents").mkdir(parents=True)
    catalogues.mkdir()
    (src / "agents" / "old.py").write_text("kept = 1\n")
    (src / "agents" / "new.py").write_text("moved = 2\ntwice = 3\n")
    (src / "agents" / "other.py").write_text("twice = 3\n")
    rows = [
        {"id": "kept", "path": "agents/old.py", "old": "kept = 1", "new": "kept = 0"},
        {"id": "moved", "path": "agents/old.py", "old": "moved = 2", "new": "x"},
        {"id": "twice", "path": "agents/old.py", "old": "twice = 3", "new": "x"},
    ]
    (catalogues / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    catalogue = {
        r["id"]: faults.Fault(r["id"], r["path"], r["old"], r["new"], ()) for r in rows
    }

    moved, lost = fault_requote.moves(str(src), catalogue)
    assert moved == {"moved": ("agents/old.py", "agents/new.py")}
    assert lost == ["twice"]
    fault_requote.rewrite(str(catalogues), moved)
    paths = [json.loads(line)["path"] for line in (catalogues / "a.jsonl").open()]
    assert paths == ["agents/old.py", "agents/new.py", "agents/old.py"]
