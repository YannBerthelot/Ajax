"""Shared JAX-pure performance helpers for Ajax agents.

These primitives encapsulate patterns the Round-1/Round-2/Round-3 perf
audit found across multiple agents (see ``PERFORMANCE_LOG.md``). Use
them from any agent's ``train_*.py`` so a fix lands in one place
instead of being duplicated per-agent.

Public API:
    train_jit            Decorator for top-level per-seed train functions.
    build_resumable_train
                         Builds the canonical init-or-resume + lax.scan
                         inner train function shared by every agent.
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
# build_resumable_train
# ---------------------------------------------------------------------------
def build_resumable_train(
    *,
    init_fn: Callable[[Any, Any], Any],
    scan_fn: Callable[..., Tuple[Any, Any]] | None = None,
    make_scan_fn: Callable[..., Callable[..., Tuple[Any, Any]]] | None = None,
    num_updates: int,
    init_transform: Callable[[Any, Any], Any] | None = None,
) -> Callable:
    """Build the canonical init-or-resume + ``lax.scan`` inner train fn.

    Every Ajax agent shares the exact same training-loop skeleton:

      1. either build a fresh ``agent_state`` from the agent's ``init_*``
         function, or — when ``resume_from_state=True`` — reuse the
         ``initial_state`` handed in from a checkpoint;
      2. on a fresh run only, optionally apply a one-shot
         ``init_transform`` (skipped on resume so expensive one-time
         initialization is not re-run when continuing a previous run);
      3. ``jax.lax.scan`` the per-iteration scan body for
         ``num_updates`` steps;
      4. return ``(agent_state, out)``.

    This helper owns that skeleton so it is written exactly once instead
    of being duplicated (and drifting) across every agent's
    ``train_*.py``. Each agent's ``make_train`` keeps its own setup
    (network/optimizer/action-pipeline construction, the per-iteration
    partial, the ``num_updates`` computation, logging) and simply hands
    the resulting pieces to this function.

    The returned ``train`` function has the signature

        train(key, index=None, initial_state=None, resume_from_state=False)

    and is decorated with :func:`train_jit`, so ``resume_from_state``
    becomes a trace-time static (init-fresh vs load-from-checkpoint
    branch is dead-code-eliminated) and ``initial_state`` buffers are
    donated on the resume path.

    Args:
        init_fn: ``(key, index) -> agent_state``. Builds a fresh agent
            state. Called only on the fresh-init path. ``index`` is the
            per-seed index (or ``None``); agents that don't need it
            simply ignore it.
        scan_fn: the per-iteration body ``(carry, x) -> (carry, aux)``.
            Use this when the body needs neither the resolved
            ``agent_state`` nor the per-call ``key`` / ``index``.
            Mutually exclusive with ``make_scan_fn``.
        make_scan_fn: a *builder*
            ``(agent_state, resume_from_state, key, index) -> body``
            invoked once at trace time, after init/resume is resolved.
            Use this when the body must be finished from values produced
            during ``init_fn`` (e.g. SAC's value-box bounds), must
            branch on whether this is a resume, or needs the per-call
            ``key`` / per-seed ``index``. Mutually exclusive with
            ``scan_fn``.
        num_updates: number of scan iterations (static int).
        init_transform: optional one-shot ``(agent_state, key) ->
            agent_state`` applied on the fresh-init path only. Pass
            ``None`` for agents that have no such transform.

    Returns:
        The jit-decorated inner ``train`` function.
    """
    if (scan_fn is None) == (make_scan_fn is None):
        raise ValueError(
            "build_resumable_train: pass exactly one of `scan_fn` or " "`make_scan_fn`."
        )

    @train_jit
    def train(
        key: Any,
        index: Any = None,
        initial_state: Any = None,
        resume_from_state: bool = False,
    ) -> Tuple[Any, Any]:
        if resume_from_state:
            agent_state = initial_state
        else:
            agent_state = init_fn(key, index)
            if init_transform is not None:
                agent_state = init_transform(agent_state, key)

        body = (
            scan_fn
            if make_scan_fn is None
            else make_scan_fn(agent_state, resume_from_state, key, index)
        )

        agent_state, out = jax.lax.scan(
            f=body,
            init=agent_state,
            xs=None,
            length=num_updates,
        )
        return agent_state, out

    return train


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
    sample_x = jax.tree.map(lambda a: a[0], xs) if xs is not None else None
    aux_shape = jax.eval_shape(lambda c: body(c, sample_x)[1], init_carry)
    init_aux = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_shape)
    (final_carry, last_aux), _ = jax.lax.scan(
        _wrapped, (init_carry, init_aux), xs=xs, length=length
    )
    return final_carry, last_aux


__all__ = ["train_jit", "build_resumable_train", "final_aux_scan"]
