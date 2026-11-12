"""Shared JAX-pure performance helpers for Ajax agents.

Patterns the perf audits found across agents (``PERFORMANCE_LOG.md``),
written once so a fix lands in one place instead of in every agent's
``train_*.py``.
"""

import warnings
from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def build_resumable_train(
    *,
    init_fn: Callable[[Any, Any], Any],
    make_scan_fn: Callable[..., Callable[..., Tuple[Any, Any]]],
    num_updates: int,
    resume_transform: Callable[[Any, Any], Any] | None = None,
    carry_out: Callable[[Any, Any], Any] | None = None,
) -> Callable:
    """Build the canonical init-or-resume + ``lax.scan`` inner train fn.

    Every Ajax agent shares this skeleton, written once here so it cannot
    drift between agents: build a fresh ``agent_state`` with ``init_fn``
    (its one-shot initialisation included, so a continued run does not
    re-run it) or, with ``resume_from_state=True``, reuse the
    ``initial_state`` handed in from a checkpoint; then ``jax.lax.scan``
    the per-iteration body ``num_updates`` times, its input the iteration
    index. Each agent's ``make_train`` keeps its own setup (networks,
    optimisers, action pipeline, logging) and hands the pieces over.

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
    to the program.

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
        make_scan_fn: a *builder*
            ``(agent_state, resume_from_state, key, index) -> body`` of
            the per-iteration body ``(carry, x) -> (carry, aux)``, invoked
            once at trace time, after init/resume is resolved, so the body
            can be finished from the resolved state, branch on whether
            this is a resume, or use the per-call ``key`` / per-seed
            ``index`` or the ``shared`` input (passed as ``shared=`` when
            given).
        num_updates: number of scan iterations (static int).
        resume_transform: optional one-shot ``(agent_state, key) ->
            agent_state`` applied on the resume path only (e.g.
            re-initialising the optimizer for a new curriculum stage while
            keeping the learned parameters).
        carry_out: optional ``(agent_state, index) -> out``, the initial
            value of an output the body writes into as it goes (in place)
            instead of stacking one per iteration. The body then maps
            ``((agent_state, out), x)`` to ``((agent_state, out), None)``
            and ``train`` returns the final ``(agent_state, out)``.
    """

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

        if shared is None:
            body = make_scan_fn(agent_state, resume_from_state, key, index)
        else:
            body = make_scan_fn(
                agent_state, resume_from_state, key, index, shared=shared
            )

        # The body receives the (unbatched) iteration index as its scan input:
        # a body that gates periodic work on it gets a predicate that stays a
        # real ``lax.cond`` under vmap, unlike anything derived from the
        # (possibly batched, on resume) agent state.
        iterations = jnp.arange(num_updates)
        if not (isinstance(iteration_offset, int) and iteration_offset == 0):
            iterations = jnp.asarray(iteration_offset, iterations.dtype) + iterations
        if carry_out is None:
            return jax.lax.scan(body, agent_state, iterations, length=num_updates)
        carry = (agent_state, carry_out(agent_state, index))
        carry, _ = jax.lax.scan(body, carry, iterations, length=num_updates)
        return carry

    return train


def build_chunked_train(
    *,
    init_fn: Callable[[Any, Any], Any],
    tick_fn: Callable[[Any, Any, Any], Tuple[Any, Any]],
    log_fn: Callable[[Any, Any, Any, Any, Any], Tuple[Any, Any]],
    due: Callable[[np.ndarray], np.ndarray],
    num_updates: int,
    carry_out: Callable[[Any, Any], Any],
    resume_transform: Callable[[Any, Any], Any] | None = None,
) -> Callable:
    """:func:`build_resumable_train`'s train with the evaluations between
    chunks of ticks, as Brax trains: the ticks up to the next one that
    evaluates are one jitted scan of ``tick_fn``, and ``log_fn`` runs on the
    state the chunk ends with, as a program of its own. The training program
    then holds no evaluation, which slowed every tick of it (about 4% for SAC
    on MuJoCo Warp, evaluations aside); the states, and so the evaluations,
    are those of one scan.

    ``tick_fn(agent_state, index, tick) -> (agent_state, aux)`` is one tick,
    ``tick`` its absolute, unbatched index. ``log_fn(agent_state, aux, out,
    index, tick) -> (agent_state, out)`` evaluates after tick ``tick``,
    reading the tick's ``aux``, and writes into ``out``, which starts as
    ``carry_out(agent_state, index)``. ``due(ticks)``, of a NumPy array of
    absolute ticks, marks those after which ``log_fn`` runs: it must be known
    before the run, so the chunks are. Each chunk length compiles once; they
    are one or two when the evaluations come every so many ticks or steps.

    The train function has build_resumable_train's signature but ``shared``,
    and returns ``(agent_state, out)``. It is a Python loop over jitted
    programs: vmap it over the seeds or call it per seed, as
    ``ActorCritic.train`` does. Jitting it whole would put every chunk into
    one program, and it reads ``iteration_offset`` as a number.
    """

    @partial(
        jax.jit,
        static_argnames=("resume_from_state",),
        donate_argnames=("initial_state",),
    )
    def start(key: Any, index: Any, initial_state: Any, resume_from_state: bool) -> Any:
        if resume_from_state:
            agent_state = initial_state
            if resume_transform is not None:
                agent_state = resume_transform(agent_state, key)
        else:
            agent_state = init_fn(key, index)
        return agent_state, carry_out(agent_state, index)

    # The state is donated: a chunk updates the replay in place instead of
    # holding two copies of it.
    @partial(jax.jit, static_argnames=("length",), donate_argnums=(0,))
    def chunk(agent_state: Any, index: Any, first_tick: Any, length: int) -> Any:
        ticks = first_tick + jnp.arange(length)
        return final_aux_scan(
            lambda state, tick: tick_fn(state, index, tick), agent_state, length, ticks
        )

    log = jax.jit(log_fn)

    def train(
        key: Any,
        index: Any = None,
        initial_state: Any = None,
        resume_from_state: bool = False,
        iteration_offset: Any = 0,
    ) -> Tuple[Any, Any]:
        first = int(iteration_offset)
        ticks = np.arange(first, first + num_updates)
        evaluates = set(ticks[np.asarray(due(ticks), dtype=bool)].tolist())
        agent_state, out = start(key, index, initial_state, resume_from_state)
        ends = evaluates.union(ticks[-1:].tolist())
        begin = first
        for end in sorted(ends):
            with warnings.catch_warnings():
                # A leaf a chunk leaves as it was (the parameters before the
                # first update) is passed through, not copied, so its donated
                # buffer has no use: nothing is lost.
                warnings.filterwarnings(
                    "ignore", message="Some donated buffers were not usable"
                )
                agent_state, aux = chunk(agent_state, index, begin, end - begin + 1)
            if end in evaluates:
                agent_state, out = log(agent_state, aux, out, index, end)
            begin = end + 1
        return agent_state, out

    return train


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

    The body maps ``(carry, x) -> (new_carry, aux)``, ``aux`` a pytree of
    arrays (any structure).

    Args:
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

    sample_x = jax.tree.map(lambda a: a[0], xs) if xs is not None else None
    aux_shape = jax.eval_shape(lambda c: body(c, sample_x)[1], init_carry)
    init_aux = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), aux_shape)
    (final_carry, last_aux), _ = jax.lax.scan(
        _wrapped, (init_carry, init_aux), xs=xs, length=length
    )
    return final_carry, last_aux


__all__ = ["build_chunked_train", "build_resumable_train", "final_aux_scan"]
