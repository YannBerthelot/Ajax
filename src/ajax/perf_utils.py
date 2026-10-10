"""Shared JAX-pure performance helpers for Ajax agents.

These primitives encapsulate patterns the Round-1/Round-2/Round-3 perf
audit found across multiple agents (see ``PERFORMANCE_LOG.md``). Use
them from any agent's ``train_*.py`` so a fix lands in one place
instead of being duplicated per-agent.

Public API:
    build_resumable_train
                         Builds the canonical init-or-resume + lax.scan
                         inner train function shared by every agent
                         (directly, or through ``ajax.agents.loop``).
    final_aux_scan       lax.scan that exposes only the last-step aux,
                         without materializing the full ys axis.
    final_aux_fori       fori_loop with a traced trip count that exposes
                         only the last iteration's aux.
"""

from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


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
    resume_transform: Callable[[Any, Any], Any] | None = None,
    carry_out: Callable[[Any, Any], Any] | None = None,
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
         ``num_updates`` steps, feeding it the iteration index as ``x``;
      4. return ``(agent_state, out)``.

    This helper owns that skeleton so it is written exactly once instead
    of being duplicated (and drifting) across every agent's
    ``train_*.py``. Each agent's ``make_train`` keeps its own setup
    (network/optimizer/action-pipeline construction, the per-iteration
    partial, the ``num_updates`` computation, logging) and simply hands
    the resulting pieces to this function.

    The returned ``train`` function has the signature

        train(key, index=None, initial_state=None, resume_from_state=False,
              iteration_offset=0, shared=None)

    and is jitted with ``resume_from_state`` static (the init-fresh vs
    load-from-checkpoint branch is dead-code-eliminated) and
    ``initial_state`` donated: XLA reuses the incoming agent-state buffers
    on the resume path, saving one agent state of peak memory (donating
    ``None``, the init path, is a no-op).

    The scan input is the iteration index ``iteration_offset +
    arange(num_updates)``: a resumed run passes the number of iterations
    already done so the body sees *absolute* iteration indices and every
    schedule computed from them (seed phase, update bursts, static resets,
    logging cadence) continues instead of restarting. A passed offset is a
    traced scalar (one compilation serves every offset) and must be
    unbatched -- pass it through ``jax.vmap`` with ``in_axes=None`` -- so
    the index, and every predicate derived from it, stays unbatched under
    the seed vmap. The default (the Python int 0, not passed) adds nothing
    to the program: it is exactly the one before offsets existed.

    ``shared`` is an optional pytree the body reads but never carries -- an
    offline dataset, say -- handed to ``make_scan_fn`` as the keyword
    argument ``shared`` (only when it is not ``None``, so builders that do
    not take it are unchanged). Pass it through the seed vmap with
    ``in_axes=None``: it then enters the compiled program once, as an
    argument shared by every seed. Closing over it instead would bake it into
    the program as a constant (HLO bloat, a recompilation per dataset), and
    a batched argument would copy it per seed.

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
            during ``init_fn``, must branch on whether this is a resume,
            or needs the per-call ``key`` / per-seed ``index`` or the
            ``shared`` input (passed as ``shared=`` when given). Mutually
            exclusive with ``scan_fn``.
        num_updates: number of scan iterations (static int).
        init_transform: optional one-shot ``(agent_state, key) ->
            agent_state`` applied on the fresh-init path only. Pass
            ``None`` for agents that have no such transform.
        resume_transform: optional one-shot ``(agent_state, key) ->
            agent_state`` applied on the resume path only, the mirror of
            ``init_transform`` (e.g. re-initialising the optimizer for a
            new curriculum stage while keeping the learned parameters).
        carry_out: optional ``(agent_state, index) -> out``, the initial
            value of an output the body writes into as it goes (in place)
            instead of stacking one per iteration. The body then maps
            ``((agent_state, out), x)`` to ``((agent_state, out), None)``
            and ``train`` returns the final ``(agent_state, out)``.

    Returns:
        The jit-decorated inner ``train`` function.
    """
    if (scan_fn is None) == (make_scan_fn is None):
        raise ValueError(
            "build_resumable_train: pass exactly one of `scan_fn` or " "`make_scan_fn`."
        )

    @partial(
        jax.jit,
        static_argnames=("resume_from_state",),
        donate_argnames=("initial_state",),
    )
    def train(
        key: Any,
        index: Any = None,
        initial_state: Any = None,
        resume_from_state: bool = False,
        iteration_offset: Any = 0,
        shared: Any = None,
    ) -> Tuple[Any, Any]:
        if resume_from_state:
            agent_state = initial_state
            if resume_transform is not None:
                agent_state = resume_transform(agent_state, key)
        else:
            agent_state = init_fn(key, index)
            if init_transform is not None:
                agent_state = init_transform(agent_state, key)

        if make_scan_fn is None:
            body = scan_fn
        elif shared is None:
            body = make_scan_fn(agent_state, resume_from_state, key, index)
        else:
            body = make_scan_fn(
                agent_state, resume_from_state, key, index, shared=shared
            )

        # The body receives the (unbatched) iteration index as its scan input.
        # Every existing body ignores it (``_``); bodies that gate periodic
        # work on it get a trace-time-shaped predicate that stays a real
        # ``lax.cond`` under vmap, unlike anything derived from the (possibly
        # batched, on resume) agent state.
        iterations = jnp.arange(num_updates)
        if not (isinstance(iteration_offset, int) and iteration_offset == 0):
            iterations = jnp.asarray(iteration_offset, iterations.dtype) + iterations
        if carry_out is None:
            return jax.lax.scan(body, agent_state, iterations, length=num_updates)
        carry = (agent_state, carry_out(agent_state, index))
        carry, _ = jax.lax.scan(body, carry, iterations, length=num_updates)
        return carry

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


# ---------------------------------------------------------------------------
# final_aux_fori
# ---------------------------------------------------------------------------
def final_aux_fori(
    body: Callable[[jax.Array, Any], Tuple[Any, Any]],
    carry: Any,
    n: Any,
) -> Tuple[Any, Any]:
    """Run ``body`` ``n`` times (``n`` may be traced); keep only the last aux.

    The loop-with-a-data-dependent-trip-count counterpart of
    :func:`final_aux_scan`, for a variable number of updates per tick
    (DESIGN §5.3)::

        carry, aux = final_aux_fori(update, carry, n_updates(tick))

    ``body(i, carry) -> (carry, aux)`` receives the int32 iteration index
    ``i`` in ``[0, n)``. The result is ``(carry, aux)`` with ``aux`` from
    the final iteration, or zeros of its shape when ``n == 0`` (the aux
    structure comes from ``jax.eval_shape``). Only the last aux is carried,
    never a stacked ``[n, ...]`` history.

    ``n`` must be computed from *unbatched* values (the absolute tick
    index, static hyperparameters): ``lax.fori_loop`` then lowers to one
    ``while`` whose predicate stays unbatched under the seed ``vmap``, so
    the loop runs exactly ``n`` times instead of becoming a select-masked
    loop to the maximum over seeds.

    ``body`` is traced once: it is wrapped in ``jax.jit`` so the abstract
    evaluation that derives the aux structure and the loop body share
    jit's trace cache (the index and carry have the same types in both).
    A carry whose types change across iterations (e.g. Python-scalar weak
    types) would only cost one extra trace, never correctness.
    """
    body_jit = jax.jit(body)
    index_struct = jax.ShapeDtypeStruct((), jnp.int32)
    aux_struct = jax.eval_shape(body_jit, index_struct, carry)[1]
    init_aux = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_struct)

    def step(i, state):
        inner, _prev_aux = state
        return body_jit(i, inner)

    return jax.lax.fori_loop(
        jnp.int32(0), jnp.asarray(n, jnp.int32), step, (carry, init_aux)
    )


__all__ = ["build_resumable_train", "final_aux_fori", "final_aux_scan"]
