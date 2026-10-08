"""Persistent JAX compile cache, one subdirectory per host fingerprint.

HPO sweeps and any workflow that re-launches Ajax in a fresh Python
process pay the JIT compile cost once per launch. Pointing JAX at a
persistent cache directory lets later launches with identical shapes and
code load the compiled executable instead of compiling it again.

Behaviour:
  * Root directory: ``~/.cache/ajax/jax_compile_cache``, overridden by
    ``AJAX_JAX_COMPILE_CACHE_DIR``.
  * Entries live in ``<root>/<fingerprint hash>``. The fingerprint is the
    CPU model (plus the CPU feature flags on Linux), the machine
    architecture, and the Python, JAX and jaxlib versions.
  * ``AJAX_NO_COMPILE_CACHE=1`` disables the cache and leaves the disk
    untouched.

Why a fingerprint: JAX's cache key covers the HLO, the JAX/jaxlib
versions and the device kind, but on CPU not the instruction-set
features. An entry built on another CPU loads with "Target machine
feature ... could lead to SIGILL" warnings and falls back to a fresh
compile, which once looked like a multi-hour stall. Keying the
subdirectory by CPU keeps such entries apart.

Why a subdirectory rather than a wipe on mismatch: the previous design
recorded one fingerprint in the cache root and deleted the whole cache
whenever it changed. The hostname was part of it, so switching networks
wiped the cache, and every project importing Ajax from an environment
with another JAX or Python version wiped it for all the others. Keeping
one subdirectory per fingerprint, never deleted, lets each environment
keep its own warm cache.

JAX never writes programs that contain host callbacks to this cache
(``jax._src.compiler._cache_write``), so runs that log through
``jax.debug.callback`` recompile on every launch.
"""

import hashlib
import os
import platform
import subprocess
import sys

_TRUTHY = ("1", "true", "yes")


def _cpu_description() -> str:
    """The CPU model, plus its feature flags where the OS reports them.

    ``platform.processor()`` is not enough: it returns ``"arm"`` on every
    Apple Silicon Mac and is often empty on Linux.
    """
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    model, flags = "", ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                key = key.strip().lower()
                if key == "model name" and not model:
                    model = value.strip()
                elif key in ("flags", "features") and not flags:
                    flags = hashlib.sha256(value.strip().encode()).hexdigest()[:12]
                if model and flags:
                    break
    except OSError:
        pass
    return f"{model} flags={flags}" if model or flags else platform.processor()


def host_fingerprint() -> str:
    """What must match for a cached CPU executable to be safe to load.

    Deliberately excludes the hostname: it changes with the network on
    macOS and says nothing about the CPU.
    """
    from importlib.metadata import version

    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"cpu={_cpu_description()}|arch={platform.machine()}|py={py}"
        f"|jax={version('jax')}|jaxlib={version('jaxlib')}"
    )


def cache_dir_for(root: str, fingerprint: str) -> str:
    """The subdirectory of ``root`` that holds this fingerprint's entries."""
    return os.path.join(root, hashlib.sha256(fingerprint.encode()).hexdigest()[:16])


def configure() -> None:
    """Point JAX at this host's cache subdirectory (see the module docstring)."""
    if os.environ.get("AJAX_NO_COMPILE_CACHE", "").lower() in _TRUTHY:
        return

    import jax

    root = os.environ.get("AJAX_JAX_COMPILE_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "ajax", "jax_compile_cache"
    )
    try:
        cache_dir = cache_dir_for(root, host_fingerprint())
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        # Cache every compile. JAX's default skips compiles under one
        # second, which are cheap to store and add up over a sweep.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except Exception:
        # A read-only home or an unusual filesystem must not block
        # importing Ajax; run without the cache instead.
        pass
