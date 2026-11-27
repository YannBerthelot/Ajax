"""The originals' kept records (tests/conformance/originals.py): each is
recorded again exactly when its own inputs change."""

import pathlib

from tests.conformance.originals import inputs_hash


def _tree(root: pathlib.Path) -> None:
    ref = root / "benchmarks" / "reference"
    (ref / "vendor" / "orig_a").mkdir(parents=True)
    (ref / "vendor" / "orig_b").mkdir(parents=True)
    (ref / "vendor" / "orig_a" / "agent.py").write_text("A = 1\n")
    (ref / "vendor" / "orig_b" / "agent.py").write_text("B = 1\n")
    (ref / "helper.py").write_text("H = 1\n")
    (ref / "shared.py").write_text("from helper import H\n")
    (ref / "rec_a.py").write_text('from shared import H\nVENDOR = "vendor/orig_a"\n')
    (ref / "rec_b.py").write_text('VENDOR = "vendor/orig_b"\n')
    (root / "env").mkdir()
    (root / "env" / "uv.lock").write_text("lock 1\n")


def test_a_change_records_again_only_the_originals_it_reaches(tmp_path) -> None:
    """A change to one vendored original, or to a helper one recorder
    imports through another, moves that recorder's hash and leaves the
    other's: one key over every original's inputs re-recorded all of them
    for any change (Conformance 7 to 16 min). The env's lock reaches both."""
    _tree(tmp_path)
    ref = tmp_path / "benchmarks" / "reference"

    def hashes() -> tuple[str, str]:
        return inputs_hash("rec_a.py", "env", tmp_path), inputs_hash(
            "rec_b.py", "env", tmp_path
        )

    a0, b0 = hashes()
    (ref / "vendor" / "orig_a" / "agent.py").write_text("A = 2\n")
    a1, b1 = hashes()
    assert a1 != a0 and b1 == b0
    (ref / "helper.py").write_text("H = 2\n")
    a2, b2 = hashes()
    assert a2 != a1 and b2 == b0
    (tmp_path / "env" / "uv.lock").write_text("lock 2\n")
    a3, b3 = hashes()
    assert a3 != a2 and b3 != b0
