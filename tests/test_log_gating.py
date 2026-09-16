"""The logging side effect must respect the gate under vmap (select lowering)."""

import jax
import jax.numpy as jnp

from ajax.log import gated_log_callback


def test_gated_callback_fires_once_per_true_flag_under_vmap_select():
    calls = []

    def log_fn(metrics, index):
        calls.append((int(index), float(metrics["x"])))

    def body(flag, x, index):
        def run(x):
            gated_log_callback(log_fn, flag, {"x": x}, index)
            return x

        # batched predicate -> lowered to select under vmap: both branches run
        return jax.lax.cond(flag, run, lambda x: x, x)

    flags = jnp.array([True, False, True, False])
    jax.vmap(body)(flags, jnp.arange(4.0), jnp.arange(4))
    jax.effects_barrier()
    assert sorted(calls) == [(0, 0.0), (2, 2.0)]


def test_ungated_callback_would_fire_for_every_element_under_vmap_select():
    """Documents the failure mode the helper exists for."""
    calls = []

    def body(flag, x):
        def run(x):
            jax.debug.callback(lambda v: calls.append(float(v)), x)
            return x

        return jax.lax.cond(flag, run, lambda x: x, x)

    jax.vmap(body)(jnp.array([True, False]), jnp.arange(2.0))
    jax.effects_barrier()
    assert len(calls) == 2  # fires for the False element too
