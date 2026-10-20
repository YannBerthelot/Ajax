"""Compile statistics of a pytest run, for the Compile preview workflow
(.github/workflows/compile-preview.yml): persistent-cache hits and misses,
compile requests (JAX reports one per program, a cache hit included, at
about 0 s), XLA compile time, and the time spent reading cache entries.

``pytest -p tests.compile_stats`` writes them as JSON to ``$COMPILE_STATS``
(one file per xdist worker, suffixed with its id); without the variable it
records nothing."""

from __future__ import annotations

import json
import os
from typing import Any

import jax

COUNTS = {"/jax/compilation_cache/cache_hits": "hits"}
COUNTS["/jax/compilation_cache/cache_misses"] = "misses"
DURATIONS = {"/jax/core/compile/backend_compile_duration": "compile_s"}
DURATIONS["/jax/compilation_cache/cache_retrieval_time_sec"] = "retrieval_s"
stats: dict[str, float] = {"hits": 0, "misses": 0, "requests": 0}
stats |= {"compile_s": 0.0, "retrieval_s": 0.0}


def on_event(event: str, **_: str | int) -> None:
    if event in COUNTS:
        stats[COUNTS[event]] += 1


def on_duration(event: str, duration_secs: float, **_: str | int) -> None:
    if event in DURATIONS:
        stats[DURATIONS[event]] += duration_secs
        stats["requests"] += event.endswith("backend_compile_duration")


def pytest_sessionstart(session: Any) -> None:
    if os.environ.get("COMPILE_STATS"):
        jax.monitoring.register_event_listener(on_event)
        jax.monitoring.register_event_duration_secs_listener(on_duration)


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    path = os.environ.get("COMPILE_STATS")
    if not path:
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    with open(f"{path}.{worker}" if worker else path, "w") as fh:
        json.dump({**stats, "tests": session.testscollected}, fh)
