"""Test-only seams for conformance tests: a tape of random draws, and the capture of training steps.

A conformance test runs one training step of an Ajax agent and one of its original implementation
from the same state with the same random numbers, and compares them sub-step by sub-step. Two seams
make that possible without changing the code under test:

* The tape wraps the samplers of :mod:`jax.random` while a program is traced. :func:`recording` logs
  each draw with its role and key, the value passed through a host callback so that the log follows
  program order (a callback returning nothing may run anywhere, even after the main loop).
  :func:`replaying` returns, for each draw, the value a table holds under the draw's own key, with no
  callback on the value path. A role is the sampler, its first caller in an owned package, and the
  draw's shape and dtype; a permutation of an array is taped as its index permutation, and a run in
  float64 replays a float32 tape (a 64-bit draw takes its role's 32-bit entry, cast). Samplers decide
  at trace time, so both clear JAX's trace caches: an inner ``jax.jit`` would otherwise keep the trace
  of the other mode.
* :func:`capture_step` returns an agent's loop ``iteration``, its ``update`` and its ``init``, by
  intercepting :meth:`TrainLoop.train` while the agent builds its train function, before any
  tracing; :func:`capture_scan` does the same for a ``lax.scan`` body (an original's step function)
  and :func:`stashing` keeps the outputs of named scans (carry and stacked), such as the rollout;
  :func:`substituting` gives named scans such outputs in place of running them.

A second tape, recording_key_uses, logs every use of a key (a draw, a split, a fold_in), from which
tests/state/test_key_uniqueness.py finds the keys that hand out related numbers twice.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import inspect
import pathlib
from typing import Any, Callable, Iterator, Mapping

import jax
import jax._src.random as _random
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

Role = tuple[str, str, tuple[int, ...], str]
Table = Mapping[Role, tuple[np.ndarray, np.ndarray]]
SAMPLERS = (
    "normal",
    "uniform",
    "randint",
    "permutation",
    "choice",
    "gumbel",
    "bernoulli",
    "truncated_normal",
    "categorical",
    "exponential",
    "beta",
    "gamma",
    "bits",
    "orthogonal",
)
#: Packages whose callers name a role (the first such frame); vendored originals live under vendor/.
OWNERS = ("/src/ajax/", "/gymnax/", "/brax/", "/mujoco_playground/", "/vendor/")
#: The functions whose draws start an episode. An auto-reset evaluates the next episode's start at
#: every step and keeps it only where the episode ends, and with resets keyed by episode it draws
#: it on one key throughout: the value the env kept is that key's last draw, and the redraws are
#: one use of the key.
RESET_FUNCTIONS = ("reset_env", "reset")
_HERE = pathlib.Path(__file__).resolve()
_NARROW = {"float64": "float32", "int64": "int32", "uint64": "uint32"}


@dataclasses.dataclass
class Session:
    """What a recording or replay saw: the draws in program order, the roles a replay had no table
    entry for (shape-only traces, or a missed draw, which a comparison then shows), and the draws whose
    key was not in their role's table."""

    log: list = dataclasses.field(default_factory=list)
    unreplayed: set = dataclasses.field(default_factory=set)
    misses: list = dataclasses.field(default_factory=list)


@functools.cache
def _resolved(filename: str) -> pathlib.Path:
    return pathlib.Path(filename).resolve()


def _call_site(passed_over: tuple[str, ...] = ()) -> str:
    # Frames walked directly: inspect.stack() reads every frame's source lines, which made the tape
    # two thirds of a traced step's time.
    here = inspect.currentframe()
    f = here.f_back.f_back if here is not None and here.f_back is not None else None
    frames = []
    while f is not None:  # the sampler's callers, innermost first
        if _resolved(f.f_code.co_filename) != _HERE:
            frames.append((f.f_code.co_filename, f.f_code.co_name))
        f = f.f_back
    owned = [
        f
        for f in frames
        if any(o in f[0] for o in OWNERS) and not any(p in f[0] for p in passed_over)
    ]
    other = [f for f in frames if "/jax/" not in f[0]]
    for filename, function in owned or other:
        short = (
            filename.split("site-packages/")[-1]
            .split("/src/")[-1]
            .split("/vendor/")[-1]
        )
        return f"{short}:{function}"
    return "?"


def is_reset_site(site: str) -> bool:
    """Whether a draw's site (file:function, as _call_site names it) is an episode's start."""
    return site.rpartition(":")[2] in RESET_FUNCTIONS


def _key_data(key: jax.Array) -> jax.Array:
    return (
        jax.random.key_data(key)
        if jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
        else key
    )


def _wrap(
    name: str, original: Callable, session: Session, mode: str, table: Table
) -> Callable:
    def draw(
        role: Role, key: jax.Array, sample: Callable[[jax.Array], jax.Array]
    ) -> jax.Array:
        kd = _key_data(key)
        if mode == "record":
            value = sample(key)

            def log(k: np.ndarray, v: np.ndarray) -> np.ndarray:
                session.log.append((role, np.asarray(k), np.asarray(v)))
                return np.asarray(v)

            return io_callback(
                log,
                jax.ShapeDtypeStruct(value.shape, value.dtype),
                kd,
                value,
                ordered=False,
            )
        entry = table.get(role, table.get((*role[:3], _NARROW.get(role[3], role[3]))))
        if entry is None:
            session.unreplayed.add(role)
            return sample(key)
        keys, values = entry
        hit = jnp.all(jnp.asarray(keys) == kd, axis=-1)
        jax.debug.callback(
            lambda h: None if np.asarray(h).any() else session.misses.append(role), hit
        )
        index = jnp.argmax(hit)
        if is_reset_site(role[1]):  # the start the episode kept (RESET_FUNCTIONS)
            index = hit.shape[0] - 1 - jnp.argmax(hit[::-1])
        return jnp.asarray(values, role[3])[index]

    def sampler(key: jax.Array, *args: Any, **kwargs: Any) -> jax.Array:
        site = _call_site()
        if (
            name == "permutation"
            and not isinstance(args[0], int)
            and jnp.ndim(args[0]) >= 1
        ):
            x, axis = jnp.asarray(args[0]), kwargs.get("axis", 0)
            n = x.shape[axis]
            idx = draw(
                ("permutation", site, (n,), "int32"), key, lambda k: original(k, n)
            )
            return jnp.take(x, idx, axis=axis)
        out = jax.eval_shape(lambda k: original(k, *args, **kwargs), key)
        role = (name, site, tuple(out.shape), str(out.dtype))
        return draw(role, key, lambda k: original(k, *args, **kwargs))

    return sampler


@contextlib.contextmanager
def _taped(mode: str, table: Table) -> Iterator[Session]:
    session, originals = (
        Session(),
        {n: getattr(_random, n) for n in SAMPLERS if hasattr(_random, n)},
    )
    jax.clear_caches()
    try:
        for name, original in originals.items():
            wrapped = _wrap(name, original, session, mode, table)
            setattr(_random, name, wrapped)
            if hasattr(jax.random, name):
                setattr(jax.random, name, wrapped)
        yield session
    finally:
        for name, original in originals.items():
            setattr(_random, name, original)
            if hasattr(jax.random, name):
                setattr(jax.random, name, original)
        jax.clear_caches()


def recording() -> contextlib.AbstractContextManager[Session]:
    return _taped("record", {})


def replaying(table: Table) -> contextlib.AbstractContextManager[Session]:
    return _taped("replay", table)


def by_role(log: list) -> dict[Role, list[tuple[np.ndarray, np.ndarray]]]:
    """Role -> [(key, value)] in program order; a vmapped draw gives one entry per batch element."""
    groups: dict[Role, list] = collections.defaultdict(list)
    for role, keys, values in log:
        for idx in np.ndindex(*keys.shape[:-1]):
            groups[role].append((keys[idx], values[idx]))
    return dict(groups)


def key_table(
    keys: Mapping[Role, list], values: Mapping[Role, list]
) -> dict[Role, tuple[np.ndarray, np.ndarray]]:
    """A replay table giving the n-th draw of each role (by its key in ``keys``) the n-th value."""
    return {r: (np.stack([k for k, _ in keys[r]]), np.stack(values[r])) for r in keys}


# --- key uses --------------------------------------------------------------------------------------

#: The indices a sampler's draw is taken to span: the counters of its values, far fewer than the tag
#: of a named key (ajax.random_keys), so a key a draw used may still be folded with a name.
DRAW_SPAN = 2**20
#: The uses that hand keys out rather than numbers.
DERIVATIONS = ("split", "fold_in")
#: The helpers that hand keys out, passed over when a use is named by its caller.
KEY_HELPERS = ("/src/ajax/state.py", "/src/ajax/random_keys.py")


@dataclasses.dataclass
class KeyUses:
    """Every use of a key while a program ran: (what, site, key, datum). A sampler's draw spans the
    indices 0 to DRAW_SPAN, a split into n keys 0 to n, and a fold_in its datum alone: under JAX's
    partitionable threefry the keys fold_in(key, i) and split(key, n)[i] and a draw's i-th value
    come from one counter, i. A key drawn on and also split or folded at an index the draw spans, or
    drawn on twice (but for an episode's start, RESET_FUNCTIONS), hands out related numbers. Two
    splits or folds of one key hand out the same keys, whose own uses show any reuse."""

    log: list = dataclasses.field(default_factory=list)

    def shared(self) -> set[tuple[str, ...]]:
        """The pairs of uses (what at site) of one key that hand out related numbers, each once."""
        uses: dict[bytes, list] = collections.defaultdict(list)
        for what, site, keys, data in self.log:
            for index in np.ndindex(*keys.shape[:-1]):
                datum = int(np.broadcast_to(data, keys.shape[:-1])[index])
                first_index = datum if what == "fold_in" else 0
                uses[keys[index].tobytes()].append(
                    (f"{what} {site}", what, first_index)
                )
        pairs = set()
        for key_uses in uses.values():
            draws: list = []
            for use in key_uses:
                # An episode's start redrawn on its key is one use (RESET_FUNCTIONS).
                repeated_start = use in draws and is_reset_site(use[0])
                if use[1] not in DERIVATIONS and not repeated_start:
                    draws.append(use)
            derived = [use for use in key_uses if use[1] in DERIVATIONS]
            for i, (drawn, _, _) in enumerate(draws):
                for other, _, _ in draws[i + 1 :]:
                    pairs.add(tuple(sorted((drawn, other))))
                for other, _, first_index in derived:
                    if first_index < DRAW_SPAN:  # an index the draw spans
                        pairs.add(tuple(sorted((drawn, other))))
        return pairs


@contextlib.contextmanager
def recording_key_uses() -> Iterator[KeyUses]:
    """Log every use of a key (each sampler's draw, split and fold_in) while a program traced inside
    runs, through a callback beside the key: a branch not taken logs nothing. A sampler's own calls
    to another (choice's to permutation) are its draw's, not uses of their own."""
    # depth: how many wrapped calls the trace is inside, so that a sampler's own calls log nothing.
    uses, depth = KeyUses(), [0]
    names = [name for name in (*SAMPLERS, *DERIVATIONS) if hasattr(_random, name)]
    originals = {name: getattr(_random, name) for name in names}

    def log_use(what: str, key: jax.Array, datum: Any) -> None:
        site = _call_site(KEY_HELPERS)

        def append(key_data: np.ndarray, data: np.ndarray) -> None:
            uses.log.append((what, site, np.asarray(key_data), np.asarray(data)))

        jax.debug.callback(append, _key_data(key), jnp.asarray(datum, jnp.uint32))

    def wrap(name: str, original: Callable) -> Callable:
        def use(key: jax.Array, *args: Any, **kwargs: Any) -> jax.Array:
            if depth[0]:
                return original(key, *args, **kwargs)
            depth[0] += 1
            try:
                out = original(key, *args, **kwargs)
            finally:
                depth[0] -= 1
            if name == "split":
                num = args[0] if args else kwargs.get("num", 2)
                log_use(name, key, int(np.prod(num)))
            elif name == "fold_in":
                log_use(name, key, args[0] if args else kwargs["data"])
            else:
                log_use(name, key, 0)
            return out

        return use

    jax.clear_caches()
    try:
        for name, original in originals.items():
            setattr(_random, name, wrap(name, original))
            if hasattr(jax.random, name):
                setattr(jax.random, name, getattr(_random, name))
        yield uses
    finally:
        for name, original in originals.items():
            setattr(_random, name, original)
            if hasattr(jax.random, name):
                setattr(jax.random, name, original)
        jax.clear_caches()


# --- steps -----------------------------------------------------------------------------------------


class _Captured(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Step:
    """An agent's training step: ``iteration(state, index, tick) -> (state, metrics)``, ``update``
    (the agent's update closure, when it has one), and ``fresh(key)``, a fresh run's state."""

    iteration: Callable
    update: Callable | None
    fresh: Callable[[jax.Array], Any]


def capture_step(agent: Any, n_timesteps: int, **train_options: Any) -> Step:
    """The agent's step as its train builds it for n_timesteps, with train_options (a logging
    configuration makes the iteration evaluate)."""
    import ajax.agents.loop as loop

    got: dict = {}
    original = loop.TrainLoop.train

    def grab(
        self: Any, init: Callable, iteration: Callable, num_updates: int, **kw: Any
    ) -> Any:
        got.update(loop=self, init=init, iteration=iteration, kw=kw)
        raise _Captured

    # The test seam replaces a method for the duration of one build (mypy's codes differ with how
    # much of the project it sees).
    loop.TrainLoop.train = grab  # type: ignore[method-assign, assignment]
    try:
        agent.train(seed=[0], n_timesteps=n_timesteps, **train_options)
    except _Captured:
        pass
    finally:
        loop.TrainLoop.train = original  # type: ignore[method-assign]
    lp, kw = got["loop"], got["kw"]
    context = {
        "env_args": lp.env_args,
        "update": kw.get("update"),
        "buffer": kw.get("buffer"),
    }
    one_update = kw.get("one_update")

    def fresh_state(key: jax.Array) -> Any:
        """The train function's fresh state, with the slot of the actor's
        last metrics where its update keeps them (TrainLoop.train)."""
        state = loop.fresh_state(
            got["init"], lp.extensions, lp.total_timesteps, key, **context
        )
        if one_update is None:
            return state
        return loop.with_actor_metrics_slot(state, one_update)

    return Step(got["iteration"], kw.get("update"), jax.jit(fresh_state))


def _body_name(f: Callable) -> str:
    return str(
        getattr(f, "__name__", None)
        or getattr(getattr(f, "func", None), "__name__", "")
    )


def capture_scan(name: str, run: Callable[[], Any]) -> tuple[Callable, Any]:
    """The body and initial carry of the first ``lax.scan`` with a body named ``name`` that ``run()``
    traces; tracing stops there."""
    original, captured = jax.lax.scan, []

    def scan(f: Callable, init: Any, *args: Any, **kwargs: Any) -> Any:
        if _body_name(f) == name:
            captured.append((f, init))
            raise _Captured
        return original(f, init, *args, **kwargs)

    jax.lax.scan = scan
    try:
        run()
    except _Captured:
        pass
    finally:
        jax.lax.scan = original
    if not captured:
        raise RuntimeError(f"no lax.scan with a body named {name!r} was traced")
    return captured[0]


@contextlib.contextmanager
def stashing(*names: str) -> Iterator[dict]:
    """While tracing a step, keep the outputs ``(carry, stacked)`` of the first scan of each named
    body, so the traced step can return them beside its own output."""
    original, stash = jax.lax.scan, {}

    def scan(f: Callable, init: Any, *args: Any, **kwargs: Any) -> Any:
        out = original(f, init, *args, **kwargs)
        if _body_name(f) in names and _body_name(f) not in stash:
            stash[_body_name(f)] = out
        return out

    jax.lax.scan = scan
    try:
        yield stash
    finally:
        jax.lax.scan = original


@contextlib.contextmanager
def substituting(outputs: Mapping[str, Any]) -> Iterator[None]:
    """While tracing a step, give each scan named in ``outputs`` those outputs ``(carry, stacked)``
    in place of running it: a step run again in float64 from a float32 run's collection, say. It
    decides at trace time, so it clears JAX's trace caches, as the tape does."""
    original = jax.lax.scan

    def scan(f: Callable, init: Any, *args: Any, **kwargs: Any) -> Any:
        if _body_name(f) in outputs:
            return jax.tree_util.tree_map(jnp.asarray, outputs[_body_name(f)])
        return original(f, init, *args, **kwargs)

    jax.lax.scan = scan
    jax.clear_caches()
    try:
        yield
    finally:
        jax.lax.scan = original
        jax.clear_caches()
