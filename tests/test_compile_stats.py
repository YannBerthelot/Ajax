"""The Compile preview workflow's statistics plugin (tests/compile_stats.py)
counts the events it is sent and writes them where it is told."""

import json
from types import SimpleNamespace

import pytest

from tests import compile_stats


def test_the_plugin_writes_the_events_it_counted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(compile_stats, "stats", dict.fromkeys(compile_stats.stats, 0))
    compile_stats.on_event("/jax/compilation_cache/cache_hits")
    compile_stats.on_event("/jax/compilation_cache/cache_misses")
    compile_stats.on_event("/jax/compilation_cache/cache_misses")
    compile_stats.on_event("/jax/other")
    compile_stats.on_duration("/jax/core/compile/backend_compile_duration", 1.5)
    compile_stats.on_duration("/jax/compilation_cache/cache_retrieval_time_sec", 0.25)
    out = tmp_path / "stats.json"
    monkeypatch.setenv("COMPILE_STATS", str(out))
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    compile_stats.pytest_sessionfinish(SimpleNamespace(testscollected=3), 0)
    written = json.loads((tmp_path / "stats.json.gw1").read_text())
    assert written == {
        "hits": 1,
        "misses": 2,
        "requests": 1,
        "compile_s": pytest.approx(1.5),
        "retrieval_s": pytest.approx(0.25),
        "tests": 3,
    }


def test_without_a_path_the_plugin_writes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("COMPILE_STATS", raising=False)
    monkeypatch.chdir(tmp_path)
    compile_stats.pytest_sessionfinish(SimpleNamespace(testscollected=3), 0)
    assert list(tmp_path.iterdir()) == []
