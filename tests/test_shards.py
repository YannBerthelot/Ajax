"""Shards by recorded duration (tests/shards.py, ``pytest --shard I/N``)."""

import subprocess
import sys

from tests import shards


def test_every_test_lands_in_one_shard_longest_first() -> None:
    """a (10) -> 0, b (6) -> 1, c (5) -> 1, d (1) -> 0, e (unknown: 1) -> 0
    (a tie: the lower shard). Loads 12 and 11."""
    durations = {"a": 10.0, "b": 6.0, "c": 5.0, "d": 1.0}
    assert shards.assign(["d", "c", "b", "a", "e"], durations, 2) == [0, 1, 1, 0, 0]


def test_the_split_is_balanced_and_the_same_in_every_process() -> None:
    durations = {f"t{i}": float(i % 7 + 1) for i in range(50)}
    ids = list(durations)
    picked = shards.assign(ids, durations, 3)
    assert picked == shards.assign(list(reversed(ids)), durations, 3)[::-1]
    loads = [
        sum(durations[i] for i, p in zip(ids, picked, strict=False) if p == s)
        for s in range(3)
    ]
    assert max(loads) - min(loads) <= max(durations.values())


def test_the_option_partitions_a_collection() -> None:
    def collected(*args: str) -> set[str]:
        cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q", *args]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return {line for line in out.splitlines() if "::" in line}

    whole = collected("tests/test_shards.py", "tests/test_tiers.py")
    parts = [
        collected("tests/test_shards.py", "tests/test_tiers.py", f"--shard={i}/3")
        for i in (1, 2, 3)
    ]
    assert set().union(*parts) == whole
    assert sum(len(p) for p in parts) == len(whole)
