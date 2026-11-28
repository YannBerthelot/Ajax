"""Shared JAX-pure performance helpers for Ajax agents.

Patterns the perf audits found across agents (``PERFORMANCE_LOG.md``),
written once so a fix lands in one place instead of in every agent's
``train_*.py``.
"""

import math
import warnings
from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

#: About how many times a call reports its progress (build_chunked_train's
#: progress): a bar that moves every hundredth of the run.
PROGRESS_REPORTS = 100


def start_state(
    init_fn: Callable[[Any, Any], Any],
    resume_transform: Callable[[Any, Any], Any] | None,
    key: Any,
    index: Any,
    initial_state: Any,
    resume_from_state: bool,
) -> Any:
    """A train call's first state: ``init_fn(key, index)`` on a fresh run;
    on a resumed one, the state passed in, through ``resume_transform`` when
    given."""
    if not resume_from_state:
        return init_fn(key, index)
    agent_state = initial_state
    if resume_transform is not None:
        agent_state = resume_transform(agent_state, key)
    return agent_state


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
        agent_state = start_state(
            init_fn, resume_transform, key, index, initial_state, resume_from_state
        )

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
    progress: Optional[Callable[[float, Any], None]] = None,
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
    The start, every chunk and every evaluation take the state by donation,
    so a run updates one copy of it in place.

    ``progress(done, index)``, when given, is called in the training process
    as training goes on, about :data:`PROGRESS_REPORTS` times a call, with
    the fraction of the call's ticks done (1.0 after the last) and the
    member's ``index``: each chunk reports from inside its program
    (:func:`reporting_chunk`), so the chunks and the evaluations stay those
    of the run without it.

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
        agent_state = start_state(
            init_fn, resume_transform, key, index, initial_state, resume_from_state
        )
        return agent_state, carry_out(agent_state, index)

    # The state is donated: a chunk updates the replay in place instead of
    # holding two copies of it.
    @partial(jax.jit, static_argnames=("length",), donate_argnums=(0,))
    def chunk(agent_state: Any, index: Any, first_tick: Any, length: int) -> Any:
        ticks = first_tick + jnp.arange(length)
        return final_aux_scan(
            lambda state, tick: tick_fn(state, index, tick), agent_state, length, ticks
        )

    # So are the evaluation's state and rows: it returns the state it reads,
    # which without donation is a copy of all of it, replay included
    # (DreamerV3's replay in host memory ran out of its 64 GB pool there).
    log = jax.jit(log_fn, donate_argnums=(0, 2))
    reporting = (
        None if progress is None else reporting_chunk(tick_fn, progress, num_updates)
    )

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
                # A leaf a chunk or an evaluation leaves as it was (the
                # parameters before the first update) is passed through, not
                # copied, so its donated buffer has no use: nothing is lost.
                warnings.filterwarnings(
                    "ignore", message="Some donated buffers were not usable"
                )
                if reporting is None:
                    agent_state, aux = chunk(agent_state, index, begin, end - begin + 1)
                else:
                    agent_state, aux = reporting(
                        agent_state, index, begin, end - begin + 1, first
                    )
                if end in evaluates:
                    agent_state, out = log(agent_state, aux, out, index, end)
            begin = end + 1
        return agent_state, out

    return train


def reporting_chunk(
    tick_fn: Callable[[Any, Any, Any], Tuple[Any, Any]],
    progress: Callable[[float, Any], None],
    num_updates: int,
) -> Callable:
    """build_chunked_train's chunk as ``chunk(agent_state, index, first_tick,
    length, call_first)``, calling ``progress(done / num_updates, index)`` in
    the training process each time a piece of its ticks has trained, with
    ``done`` the call's ticks so far (from ``call_first``, its first tick).
    The pieces are of one length, the divisor of the chunk's length nearest
    the call's ticks / :data:`PROGRESS_REPORTS`, so about that many reports a
    call; a chunk with no divisor near (a prime length) reports at its end.
    Under the seed vmap it calls ``progress`` once per member.

    The chunk is one scan of its ticks, as without reports: a tick that ends
    a piece reports under a cond, and the chunk's last report comes after
    the scan. More, shorter chunks would report as well, but MuJoCo Warp
    captures its CUDA graphs anew each time a program runs (two of about
    65 ms on SAC Hopper): a hundred chunks a call made SAC on Hopper about
    20% slower. Pieces scanned one by one inside the chunk, with a barrier
    or a callback between them, changed SAC's and TD3's bits on CPU, and a
    barrier on every tick AVG's; the cond does not."""

    def reported(done: Any, index: Any) -> None:
        progress(int(done) / num_updates, None if index is None else int(index))

    @partial(jax.jit, static_argnames=("length",), donate_argnums=(0,))
    def chunk(
        agent_state: Any, index: Any, first_tick: Any, length: int, call_first: Any
    ) -> Any:
        piece = nearest_divisor(length, num_updates / PROGRESS_REPORTS)

        def reporting_tick(carry: Any, tick: Any) -> Tuple[Any, Any]:
            state, reports = carry
            state, aux = tick_fn(state, index, tick)
            trained = tick - first_tick + 1

            def report(reports: Any) -> Any:
                jax.debug.callback(reported, tick - call_first + 1, index)
                return reports + 1

            reports = jax.lax.cond(
                (trained % piece == 0) & (trained < length),
                report,
                lambda reports: reports,
                reports,
            )
            return (state, reports), aux

        ticks = first_tick + jnp.arange(length)
        (agent_state, reports), aux = final_aux_scan(
            reporting_tick, (agent_state, jnp.int32(0)), length, ticks
        )
        # The last report reads no tick, so it could come first, before the
        # chunk has run or the reports inside it (callbacks with no data
        # between them run in any order: on CI's aarch64 runner it did, and
        # the bar, which keeps the furthest, stood still until it). Through
        # the barrier its count waits for the whole state and for the count
        # of reports, which a cond returns once its callback has run.
        agent_state, _, done = jax.lax.optimization_barrier(
            (agent_state, reports, first_tick + length - call_first)
        )
        jax.debug.callback(reported, done, index)
        return agent_state, aux

    return chunk


def nearest_divisor(n: int, target: float) -> int:
    """The divisor of ``n`` nearest ``target`` by ratio, the larger on a tie,
    among those of at least a quarter of it, so a chunk reports at most
    about four times as often as it aims to; ``n`` when none is."""
    small = [d for d in range(1, math.isqrt(n) + 1) if n % d == 0]
    divisors = [d for d in small + [n // d for d in small] if d >= target / 4]
    if not divisors:
        return n
    return min(divisors, key=lambda d: (abs(math.log(d / target)), -d))


def final_aux_scan(
    body: Callable[[Any, Any], Tuple[Any, Any]],
    init_carry: Any,
    length: int,
    xs: Any = None,
):
    """``length`` steps of ``body(carry, x) -> (new_carry, aux)``, returning
    the final carry and the last step's aux alone (``aux`` a pytree of
    arrays, any structure).

    It replaces ``lax.scan`` and keeping ``ys[-1]``, which would stack every
    step's aux, ``[seeds, length, *aux_shape]`` under the seed vmap, to throw
    all but one away. The steps before the last run in a scan that keeps no
    aux, so XLA computes none of it there (a training chunk's per-update
    logs, which only its last tick's evaluation reads); the last step runs
    once more in the program, keeping its own. The body is traced twice.

    Args:
        xs: optional pytree of per-step inputs; if None, the body
            receives ``None`` as ``x`` each step (matches scan semantics).

    Returns:
        ``(final_carry, last_aux)``; with ``length`` 0, ``init_carry`` and
        an aux of zeros.
    """
    if length < 1:
        sample_x = jax.tree.map(lambda a: a[0], xs) if xs is not None else None
        aux_shape = jax.eval_shape(lambda c: body(c, sample_x)[1], init_carry)
        return init_carry, jax.tree.map(
            lambda s: jnp.zeros(s.shape, s.dtype), aux_shape
        )

    def step_keeping_no_aux(carry: Any, x: Any) -> Tuple[Any, None]:
        return body(carry, x)[0], None

    if xs is None:
        carry, _ = jax.lax.scan(
            step_keeping_no_aux, init_carry, None, length=length - 1
        )
        return body(carry, None)
    before_last = jax.tree.map(lambda a: a[:-1], xs)
    carry, _ = jax.lax.scan(step_keeping_no_aux, init_carry, before_last)
    return body(carry, jax.tree.map(lambda a: a[-1], xs))


__all__ = ["build_chunked_train", "build_resumable_train", "final_aux_scan"]
