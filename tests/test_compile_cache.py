"""The persistent JAX compile cache keeps one subdirectory per host
fingerprint and never deletes anything on import; ``prune`` trims it when
called (see ``ajax/_compile_cache.py``).

The fingerprint must change with the CPU and the Python/JAX/jaxlib
versions, so an entry built elsewhere is never loaded here, and must not
change with the hostname, which macOS changes on every network switch.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys

import jax
import pytest

from ajax import _compile_cache

_CACHE_OPTIONS = (
    "jax_compilation_cache_dir",
    "jax_persistent_cache_min_compile_time_secs",
    "jax_persistent_cache_min_entry_size_bytes",
)


@pytest.fixture
def restore_jax_config():
    """``configure`` sets global JAX options; put them back so later tests
    do not write to a deleted temporary directory."""
    saved = {name: getattr(jax.config, name) for name in _CACHE_OPTIONS}
    yield
    for name, value in saved.items():
        jax.config.update(name, value)


@pytest.fixture
def cache_root(tmp_path, monkeypatch, restore_jax_config):
    root = tmp_path / "jax_compile_cache"
    monkeypatch.setenv("AJAX_JAX_COMPILE_CACHE_DIR", str(root))
    monkeypatch.delenv("AJAX_NO_COMPILE_CACHE", raising=False)
    return root


def test_fingerprint_ignores_the_hostname(monkeypatch):
    before = _compile_cache.host_fingerprint()
    monkeypatch.setattr(platform, "node", lambda: "another-network-name")
    assert _compile_cache.host_fingerprint() == before
    assert "another-network-name" not in before


def test_fingerprint_names_cpu_and_versions():
    fingerprint = _compile_cache.host_fingerprint()
    fields = dict(part.split("=", 1) for part in fingerprint.split("|"))
    assert set(fields) == {"cpu", "arch", "py", "jax", "jaxlib"}
    assert fields[
        "cpu"
    ], "the CPU must be identified, or entries built on another CPU could load"
    assert fields["jax"] == jax.__version__


def test_fingerprint_changes_with_the_cpu(monkeypatch):
    before = _compile_cache.host_fingerprint()
    monkeypatch.setattr(_compile_cache, "_cpu_description", lambda: "Other CPU")
    assert _compile_cache.host_fingerprint() != before


def test_configure_uses_the_fingerprint_subdirectory(cache_root):
    _compile_cache.configure()

    expected = _compile_cache.cache_dir_for(
        str(cache_root), _compile_cache.host_fingerprint()
    )
    assert jax.config.jax_compilation_cache_dir == expected
    assert os.path.isdir(expected) and os.path.dirname(expected) == str(cache_root)
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
    assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0


def test_configure_never_deletes_entries(cache_root, monkeypatch):
    """Entries of other fingerprints and of the old single-directory
    layout survive, so each environment keeps its warm cache."""
    cache_root.mkdir()
    (cache_root / "legacy_entry-cache").write_text("old layout")
    (cache_root / ".host_fingerprint").write_text("host=oldbox|cpu=oldcpu")
    other = cache_root / "0123456789abcdef"
    other.mkdir()
    (other / "entry-cache").write_text("other environment")

    _compile_cache.configure()
    monkeypatch.setattr(_compile_cache, "_cpu_description", lambda: "Other CPU")
    _compile_cache.configure()

    assert (cache_root / "legacy_entry-cache").exists()
    assert (cache_root / ".host_fingerprint").exists()
    assert (other / "entry-cache").exists()
    # The two fingerprints got two subdirectories, plus the pre-existing one.
    assert len([p for p in cache_root.iterdir() if p.is_dir()]) == 3


def test_no_compile_cache_env_var_leaves_disk_and_config_alone(cache_root, monkeypatch):
    monkeypatch.setenv("AJAX_NO_COMPILE_CACHE", "1")
    before = jax.config.jax_compilation_cache_dir

    _compile_cache.configure()

    assert not cache_root.exists()
    assert jax.config.jax_compilation_cache_dir == before


def test_importing_ajax_points_jax_at_the_fingerprint_subdirectory(tmp_path):
    """``ajax/__init__`` configures the cache before importing the agents.
    In a fresh interpreter: this session imported ajax long ago."""
    env = {**os.environ, "AJAX_JAX_COMPILE_CACHE_DIR": str(tmp_path)}
    env.pop("AJAX_NO_COMPILE_CACHE", None)
    code = "import ajax, jax; print(jax.config.jax_compilation_cache_dir)"
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == _compile_cache.cache_dir_for(
        str(tmp_path), _compile_cache.host_fingerprint()
    )


def write_entry(directory, name: str, written: float, read: float = 0.0) -> None:
    path = directory / name
    path.write_bytes(b"x" * 100)
    os.utime(path, (read, written))  # (atime, mtime)


def test_prune_deletes_the_least_recently_used_down_to_the_bound(tmp_path):
    subdirectory = tmp_path / "0123456789abcdef"
    subdirectory.mkdir()
    write_entry(subdirectory, "a-cache", written=1000)
    write_entry(subdirectory, "b-cache", written=2000)
    write_entry(subdirectory, "c-cache", written=500, read=3000)  # read lately

    assert _compile_cache.prune(str(tmp_path), max_bytes=250) == 100
    assert sorted(p.name for p in subdirectory.iterdir()) == ["b-cache", "c-cache"]
    assert _compile_cache.prune(str(tmp_path), max_bytes=250) == 0
