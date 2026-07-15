from importlib.metadata import version


def _host_fingerprint() -> str:
    """A stable host identifier for cache validity.

    Combines hostname, CPU model name, and a Python+JAX major version
    pair. If two launches see different fingerprints, they were on
    different machines (or after a major JAX upgrade) and the cache
    built by one will mostly miss / mis-load on the other — which is the
    failure mode that bit us once (Phase 5 of the architecture rework:
    pilots stalled for ~4 hours on a cache built by a different CPU,
    JAX emitting "Target machine feature ... is not supported on the
    host machine. ... could lead to SIGILL." for every cached entry).
    """
    import platform
    import sys

    cpu = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu = platform.processor() or platform.machine()
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    try:
        import jax

        jver = jax.__version__
    except Exception:
        jver = "unknown"
    return f"host={platform.node()}|cpu={cpu}|py={py}|jax={jver}"


def _configure_jax_compile_cache() -> None:
    """Enable JAX's on-disk compile cache so HLO survives process exits.

    HPO sweeps and any workflow that re-launches Ajax in a fresh Python
    process pay the JIT compile cost once per launch. Pointing JAX at a
    persistent cache directory lets subsequent launches with identical
    shapes / code hit warm HLO and skip the compile (saves ~10-20s on
    Plane3DCircle-class workloads at jax 0.10.0).

    Behaviour:
      * Default cache dir: ``~/.cache/ajax/jax_compile_cache``.
      * Override with env var ``AJAX_JAX_COMPILE_CACHE_DIR``.
      * Disable with ``AJAX_NO_COMPILE_CACHE=1`` (cache stays untouched).

    Host-fingerprint sentinel: a ``.host_fingerprint`` file inside the
    cache dir records the host that built the cache. On every import we
    compare the recorded fingerprint to the current host's. If they
    differ, the cache is from another machine (or a major JAX upgrade);
    JAX would emit "could lead to SIGILL" warnings for every load and
    silently fall back to fresh compile, which manifests as multi-hour
    "stuck" training. We wipe the cache and recreate the sentinel in
    that case, with a one-line warning. Override with
    ``AJAX_KEEP_STALE_CACHE=1`` (skip the wipe; accept the SIGILL risk).

    Caching aggressively (no min-compile-time / min-size threshold) is
    safe: the cache key includes JAX/JAXlib version + HLO hash, so
    cache poisoning across versions is not possible.
    """
    import os

    if os.environ.get("AJAX_NO_COMPILE_CACHE", "").lower() in ("1", "true", "yes"):
        return

    import jax

    cache_dir = os.environ.get("AJAX_JAX_COMPILE_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "ajax", "jax_compile_cache"
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)

        # Host-fingerprint check: wipe on mismatch unless explicitly told
        # not to. Cheap (one small file read) and prevents the multi-hour
        # silent stall this mechanism was once involved in.
        sentinel = os.path.join(cache_dir, ".host_fingerprint")
        current = _host_fingerprint()
        try:
            with open(sentinel) as fh:
                recorded = fh.read().strip()
        except OSError:
            recorded = ""

        if recorded and recorded != current:
            if os.environ.get("AJAX_KEEP_STALE_CACHE", "").lower() not in (
                "1",
                "true",
                "yes",
            ):
                import shutil

                print(
                    f"[ajax] Compile cache fingerprint changed (was "
                    f"{recorded!r}, now {current!r}); wiping "
                    f"{cache_dir} to avoid stale-HLO stalls. Set "
                    "AJAX_KEEP_STALE_CACHE=1 to skip.",
                    flush=True,
                )
                shutil.rmtree(cache_dir, ignore_errors=True)
                os.makedirs(cache_dir, exist_ok=True)
            else:
                print(
                    f"[ajax] Compile cache fingerprint changed (was "
                    f"{recorded!r}, now {current!r}) but "
                    "AJAX_KEEP_STALE_CACHE=1; keeping cache. "
                    "If training stalls without producing output, "
                    "wipe it manually: rm -rf "
                    f"{cache_dir}",
                    flush=True,
                )
        if recorded != current:
            try:
                with open(sentinel, "w") as fh:
                    fh.write(current)
            except OSError:
                pass

        jax.config.update("jax_compilation_cache_dir", cache_dir)
        # Cache every compile, regardless of HLO size or compile duration.
        # Defaults (in jax 0.10) skip very small / very fast compiles which
        # is a footgun: those tiny ones aren't the wall-clock dominators,
        # but the few large ones are, and the threshold logic adds key
        # overhead per lookup. Setting both to 0 makes cache hits maximal.
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except Exception:
        # If anything goes wrong (read-only home, weird filesystem),
        # silently fall back to no cache. Don't block agent imports.
        pass


_configure_jax_compile_cache()


from ajax.agents.APO.APO import APO  # noqa: E402
from ajax.agents.ASAC.ASAC import ASAC  # noqa: E402
from ajax.agents.AVG.AVG import AVG  # noqa: E402
from ajax.agents.DQN.DQN import DQN  # noqa: E402
from ajax.agents.PPO.PPO import PPO  # noqa: E402
from ajax.agents.PQN.PQN import PQN  # noqa: E402
from ajax.agents.REDQ.REDQ import REDQ  # noqa: E402
from ajax.agents.SAC.SAC import SAC  # noqa: E402

__all__ = ["APO", "ASAC", "AVG", "DQN", "PPO", "PQN", "REDQ", "SAC"]
__version__ = version("ajax")
