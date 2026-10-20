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
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
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


def _call_site() -> str:
    frames = [
        f for f in inspect.stack()[2:] if pathlib.Path(f.filename).resolve() != _HERE
    ]
    owned = [f for f in frames if any(o in f.filename for o in OWNERS)]
    other = [f for f in frames if "/jax/" not in f.filename]
    for f in owned or other:
        short = (
            f.filename.split("site-packages/")[-1]
            .split("/src/")[-1]
            .split("/vendor/")[-1]
        )
        return f"{short}:{f.function}"
    return "?"


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
        return jnp.asarray(values, role[3])[jnp.argmax(hit)]

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


def capture_step(agent: Any, n_timesteps: int) -> Step:
    from ajax.agents import loop

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
        agent.train(seed=[0], n_timesteps=n_timesteps)
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
    fresh = jax.jit(
        lambda key: loop.fresh_state(
            got["init"], lp.stack, lp.total_timesteps, key, **context
        )
    )
    return Step(got["iteration"], kw.get("update"), fresh)


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
