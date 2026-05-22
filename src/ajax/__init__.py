from importlib.metadata import version


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
