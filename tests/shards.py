"""Shards by recorded duration (``pytest --shard I/N``, tests/conftest.py).

Tests go longest first, each to the least loaded shard (ties to the lowest),
so every process that collects the same tests computes the same shards: the
CI jobs and their xdist workers. A test missing from ``tests/.durations``
counts ``DEFAULT`` seconds; refresh the file from CI's junit reports with
``make durations``.
"""

from __future__ import annotations

import json
import os

DURATIONS = os.path.join(os.path.dirname(__file__), ".durations")
DEFAULT = 1.0


def load(path: str = DURATIONS) -> dict[str, float]:
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def assign(nodeids: list[str], durations: dict[str, float], n: int) -> list[int]:
    """The shard (0 .. n - 1) of each node id, in the given order."""
    loads = [0.0] * n
    shard = {}
    for nodeid in sorted(nodeids, key=lambda i: (-durations.get(i, DEFAULT), i)):
        best = min(range(n), key=lambda s: (loads[s], s))
        shard[nodeid] = best
        loads[best] += durations.get(nodeid, DEFAULT)
    return [shard[i] for i in nodeids]
