"""Shared JAX-pure performance helpers for Ajax agents.

These primitives encapsulate patterns the Round-1/Round-2/Round-3 perf
audit found across multiple agents (see ``PERFORMANCE_LOG.md``). Use
them from any agent's ``train_*.py`` so a fix lands in one place
instead of being duplicated per-agent.

Public API:
    train_jit            Decorator for top-level per-seed train functions.
    final_aux_scan       lax.scan that exposes only the last-step aux,
                         without materializing the full ys axis.
"""

import inspect
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# train_jit
# ---------------------------------------------------------------------------
def train_jit(fn: Callable) -> Callable:
    """Decorate a per-seed train function with sane default jit options.

    For agents that follow the full convention

        def train(key, index=None, initial_state=None, resume_from_state=False):
            ...

    this applies:
      * ``static_argnames=("resume_from_state",)`` so the bool selects
        the init-fresh vs load-from-checkpoint branch at trace time
        (dead-code elimination on the unused side).
      * ``donate_argnames=("initial_state",)`` so XLA reuses the
        incoming agent-state buffers on the resume path; saves ~one
        ``agent_state`` worth of peak memory at the jit boundary.
        Donating ``None`` (the init path) is a safe no-op.

    Agents that don't yet take ``initial_state`` / ``resume_from_state``
    just get a plain ``jax.jit`` (decorator introspects the signature).
    Adding those parameters in future is a purely-additive change that
    automatically picks up the donation/static treatment.
    """
    params = inspect.signature(fn).parameters
    static = tuple(n for n in ("resume_from_state",) if n in params)
    donate = tuple(n for n in ("initial_state",) if n in params)
    kwargs = {}
    if static:
        kwargs["static_argnames"] = static
    if donate:
        kwargs["donate_argnames"] = donate
    return jax.jit(fn, **kwargs)


# ---------------------------------------------------------------------------
# final_aux_scan
# ---------------------------------------------------------------------------
def final_aux_scan(
    body: Callable[[Any, Any], Tuple[Any, Any]],
    init_carry: Any,
    length: int,
    xs: Any = None,
):
    """Run ``lax.scan`` and expose only the final-step aux from the body.

    Replaces the common idiom

        carry, ys = jax.lax.scan(body, init, xs, length=length)
        last_aux = jax.tree.map(lambda x: x[-1], ys)

    with a carry-only variant that never materializes the leading scan
    axis on device for the aux. Critical under vmap-over-seeds because
    the discarded ys dimension would otherwise grow as
    ``[seeds, length, *aux_shape]`` purely to be thrown away.

    The body must follow the standard ``(carry, x) -> (new_carry, aux)``
    signature where ``aux`` is a pytree of arrays (any structure). We
    use ``jax.eval_shape`` to derive a zeros placeholder for the
    initial aux carry without actually executing the body.

    Args:
        body: ``(carry, x) -> (new_carry, aux)``.
        init_carry: initial scan carry.
        length: number of scan iterations (static int).
        xs: optional pytree of per-step inputs; if None, the body
            receives ``None`` as ``x`` each step (matches scan semantics).

    Returns:
        ``(final_carry, last_aux)`` where ``last_aux`` is the body's
        return value on the final iteration only.
    """

    def _wrapped(state, x):
        carry, _prev_aux = state
        new_carry, aux = body(carry, x)
        return (new_carry, aux), None

    # Sample xs at index 0 if it is a per-step pytree, otherwise use None.
    sample_x = (
        jax.tree.map(lambda a: a[0], xs) if xs is not None else None
    )
    aux_shape = jax.eval_shape(lambda c: body(c, sample_x)[1], init_carry)
    init_aux = jax.tree.map(
        lambda s: jnp.zeros(s.shape, s.dtype), aux_shape
    )
    (final_carry, last_aux), _ = jax.lax.scan(
        _wrapped, (init_carry, init_aux), xs=xs, length=length
    )
    return final_carry, last_aux


__all__ = ["train_jit", "final_aux_scan"]
