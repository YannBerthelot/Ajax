"""Shards by recorded duration (``pytest --shard I/N``, tests/conftest.py).

Tests go longest first, each to the least loaded shard (ties to the lowest),
so every process that collects the same tests computes the same shards: the
CI jobs and their xdist workers. A test missing from ``tests/.durations``
counts ``DEFAULT`` seconds; refresh the file from CI's junit reports with
``make durations``.

The slow tier splits by group (``--shard-by group``, ``assign_groups``): a
probe case's tests stay together.

The fast tier splits by file (``--shard-by file``, ``assign_files``): a
module's tests share its runs (module fixtures), which every shard holding
one of them would build again. Its times are per file, in
``tests/.file_durations``, since most of its tests take well under the
second ``tests/.durations`` rounds up to; a file missing from it counts
``FILE_DEFAULT`` seconds.
"""

from __future__ import annotations

import json
import os

DURATIONS = os.path.join(os.path.dirname(__file__), ".durations")
FILE_DURATIONS = os.path.join(os.path.dirname(__file__), ".file_durations")
DEFAULT = 1.0
FILE_DEFAULT = 10.0


def load(path: str = DURATIONS) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def assign(
    nodeids: list[str], durations: dict[str, float], n: int, default: float = DEFAULT
) -> list[int]:
    """The shard (0 .. n - 1) of each node id, in the given order."""
    loads = [0.0] * n
    shard = {}
    for nodeid in sorted(nodeids, key=lambda i: (-durations.get(i, default), i)):
        best = min(range(n), key=lambda s: (loads[s], s))
        shard[nodeid] = best
        loads[best] += durations.get(nodeid, default)
    return [shard[i] for i in nodeids]


def assign_groups(
    nodeids: list[str], groups: list[str], durations: dict[str, float], n: int
) -> list[int]:
    """The shard of each node id when the tests of a group (``groups``, one
    per node id: the slow tier's probe cases, tests/conftest.py) stay
    together: a probe's tests read one cached training, which a shard holding
    one of them trains again (2026-11-26: 1,075 s of 6,726 s of probes). A
    group counts its tests' recorded seconds."""
    seconds: dict[str, float] = {}
    for nodeid, group in zip(nodeids, groups, strict=True):
        seconds[group] = seconds.get(group, 0.0) + durations.get(nodeid, DEFAULT)
    shard = dict(zip(seconds, assign(list(seconds), seconds, n), strict=True))
    return [shard[group] for group in groups]


def assign_files(nodeids: list[str], durations: dict[str, float], n: int) -> list[int]:
    """The shard of each node id when a file's tests stay together:
    ``durations`` holds seconds per file."""
    files = sorted({nodeid.split("::")[0] for nodeid in nodeids})
    shard = dict(zip(files, assign(files, durations, n, FILE_DEFAULT), strict=True))
    return [shard[nodeid.split("::")[0]] for nodeid in nodeids]
