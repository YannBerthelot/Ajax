"""Host-fingerprint test for the JAX compile cache configuration.

Phase-5/relaunch-day fallout: pilots stalled for ~4 hours because the
on-disk JAX compile cache had entries built on a different CPU. JAX
emitted "Target machine feature ... not supported on the host machine.
... could lead to SIGILL" for every load and silently fell back to
fresh compile, manifesting as 100%-GPU-util-but-no-output.

The fix in ``ajax/__init__.py``:
  * Record a ``.host_fingerprint`` sentinel inside the cache dir on
    every import (hostname + CPU model + Python/JAX version).
  * If the sentinel changes between launches, wipe the cache and warn.
  * ``AJAX_KEEP_STALE_CACHE=1`` overrides (skip wipe, keep cache).

These tests pin that behaviour so a future refactor of the cache
configuration doesn't quietly drop the host-check.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    """Point ajax at a temp cache dir and ensure a fresh import."""
    cache_dir = tmp_path / "jax_compile_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("AJAX_JAX_COMPILE_CACHE_DIR", str(cache_dir))
    # Force ajax to re-import so _configure_jax_compile_cache runs again.
    for name in list(sys.modules):
        if name == "ajax" or name.startswith("ajax."):
            sys.modules.pop(name, None)
    return cache_dir


def test_host_fingerprint_written_on_first_import(tmp_cache_dir):
    """Fresh cache dir → sentinel written, no warning."""
    import ajax  # noqa: F401

    sentinel = tmp_cache_dir / ".host_fingerprint"
    assert sentinel.is_file()
    content = sentinel.read_text()
    assert "host=" in content
    assert "cpu=" in content
    assert "py=" in content
    assert "jax=" in content


def test_host_fingerprint_wipes_cache_on_mismatch(tmp_cache_dir, capfd):
    """Pre-populated stale fingerprint → cache wiped, warning emitted."""
    (tmp_cache_dir / ".host_fingerprint").write_text(
        "host=oldbox|cpu=oldcpu|py=3.9|jax=0.5.0"
    )
    (tmp_cache_dir / "stale_entry").write_text("would-trigger-SIGILL")

    import ajax  # noqa: F401

    out, _ = capfd.readouterr()
    assert "[ajax]" in out
    assert "fingerprint changed" in out
    assert "wiping" in out
    assert not (tmp_cache_dir / "stale_entry").exists()
    # Sentinel rewritten with the current fingerprint.
    sentinel = tmp_cache_dir / ".host_fingerprint"
    assert sentinel.is_file()
    assert sentinel.read_text() != "host=oldbox|cpu=oldcpu|py=3.9|jax=0.5.0"


def test_keep_stale_cache_env_var_skips_wipe(tmp_cache_dir, capfd, monkeypatch):
    """AJAX_KEEP_STALE_CACHE=1 → no wipe even on mismatch (escape hatch)."""
    monkeypatch.setenv("AJAX_KEEP_STALE_CACHE", "1")
    (tmp_cache_dir / ".host_fingerprint").write_text(
        "host=oldbox|cpu=oldcpu|py=3.9|jax=0.5.0"
    )
    (tmp_cache_dir / "stale_entry").write_text("preserved-by-escape-hatch")

    import ajax  # noqa: F401

    out, _ = capfd.readouterr()
    assert "AJAX_KEEP_STALE_CACHE" in out
    assert (tmp_cache_dir / "stale_entry").exists()


def test_no_warning_when_fingerprint_matches(tmp_cache_dir, capfd):
    """Sentinel matches current host → silent (no warning, no wipe)."""
    from ajax import _host_fingerprint  # type: ignore[attr-defined]

    (tmp_cache_dir / ".host_fingerprint").write_text(_host_fingerprint())
    (tmp_cache_dir / "warm_entry").write_text("kept")

    for name in list(sys.modules):
        if name == "ajax" or name.startswith("ajax."):
            sys.modules.pop(name, None)
    importlib.import_module("ajax")

    out, _ = capfd.readouterr()
    assert "fingerprint changed" not in out
    assert (tmp_cache_dir / "warm_entry").exists()


def test_no_compile_cache_env_var_skips_setup(tmp_path, monkeypatch):
    """AJAX_NO_COMPILE_CACHE=1 → don't touch the cache dir at all."""
    cache_dir = tmp_path / "untouched"
    monkeypatch.setenv("AJAX_JAX_COMPILE_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("AJAX_NO_COMPILE_CACHE", "1")
    for name in list(sys.modules):
        if name == "ajax" or name.startswith("ajax."):
            sys.modules.pop(name, None)

    import ajax  # noqa: F401

    # Cache dir was not created, sentinel does not exist.
    assert not cache_dir.exists()


# Helper: expose _host_fingerprint for the test above to access.
def test_host_fingerprint_includes_jax_version():
    """The fingerprint must include the JAX version (major-cache poisoning)."""
    from ajax import _host_fingerprint  # type: ignore[attr-defined]

    fp = _host_fingerprint()
    assert "jax=" in fp
    # Doesn't bind to a specific JAX version — just check the format.
    assert "|" in fp
    assert fp.count("|") >= 3
