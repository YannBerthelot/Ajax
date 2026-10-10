"""One way to train for a probe: every seed in one compiled program,
optionally capturing what the agent logs or resuming a state; and one cache
(``readings``) so every test of a cell shares its training. Only numpy
readings are cached, never states. ``worker`` trains through the real logging
process; ``in_subprocess`` runs it in a fresh interpreter, where a hang
times out instead of stalling the suite.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import os
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence

import jax
import numpy as np

from ajax.agents import loop as shared_loop
from ajax.checkpoint import restore_into, save_checkpoint
from ajax.logging.wandb_logging import LoggingConfig, load_scalars_from_tfevents

from . import readouts

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROLLOUT_AGENTS = ("PPO", "APO", "PQN")
TRAIN_KEYS = (
    "Train/episodic mean reward",
    "Eval/episodic mean reward",
    "Eval/mean episodic length",
)


@dataclasses.dataclass
class Run:
    agent: Any
    state: Any
    events: list[tuple[int, dict[str, np.ndarray]]]
    seeds: tuple[int, ...]

    def logged(self, key: str) -> np.ndarray:
        """Per seed, ``key`` at the latest logged timestep (NaN if never)."""
        out, latest = (
            np.full(len(self.seeds), np.nan),
            np.full(len(self.seeds), -np.inf),
        )
        for i, m in self.events:
            t = float(np.asarray(m["timestep"]).reshape(-1)[0])
            if key in m and t >= latest[i]:
                latest[i], out[i] = t, float(np.asarray(m[key]).reshape(-1)[0])
        return out

    def timesteps(self) -> np.ndarray:
        """(seeds, events) logged timesteps in order, padded with -1."""
        per = [
            sorted(int(m["timestep"]) for i, m in self.events if i == s)
            for s in range(len(self.seeds))
        ]
        out = np.full((len(per), max(map(len, per), default=0)), -1)
        for s, ts in enumerate(per):
            out[s, : len(ts)] = ts
        return out

    def values(self, key: str) -> np.ndarray:
        """``key`` in every event, NaN where an event lacks it."""
        return np.array([float(m.get(key, np.nan)) for _, m in self.events])


def log_config(every: int, folder: str | None = None) -> LoggingConfig:
    return LoggingConfig(
        config={},
        project_name="probe",
        run_name="probe",
        log_frequency=every,
        folder=folder,
        use_tensorboard=folder is not None,
        use_wandb=False,
    )


def train_module(agent: Any) -> Any:
    """The module whose ``make_train`` the agent runs (it binds ``vmap_log``)."""
    for cls in type(agent).__mro__:
        make_train = getattr(sys.modules[cls.__module__], "make_train", None)
        if make_train is not None:
            return sys.modules[make_train.__module__]
    raise RuntimeError(f"no make_train for {type(agent).__name__}")


@contextlib.contextmanager
def captured_logs(agent: Any) -> Iterator[list]:
    """Replace ``vmap_log`` and the worker start in the agent's train module
    and the shared loop (where present: UDRL never logs); yield the (seed
    index, metrics) events."""
    modules, events = (train_module(agent), shared_loop), []

    def capture(metrics: Mapping[str, Any], index: Any, **_: Any) -> None:
        events.append(
            (int(np.asarray(index)), {k: np.asarray(v) for k, v in metrics.items()})
        )

    stubs = {"vmap_log": capture, "start_async_logging": lambda: None}
    saved = {(m, k): getattr(m, k) for m in modules for k in stubs if hasattr(m, k)}
    for m, k in saved:
        setattr(m, k, stubs[k])
    try:
        yield events
    finally:
        for (m, k), v in saved.items():
            setattr(m, k, v)


def train(
    agent: Any,
    seeds: Sequence[int],
    budget: int,
    log_every: int | None = None,
    **kw: Any,
) -> Run:
    """Train ``seeds`` for ``budget`` env steps (0: untrained); with
    ``log_every``, evaluate and capture the logs at that frequency."""
    config = None if log_every is None else log_config(log_every)
    with captured_logs(agent) as events:
        out = agent.train(
            seed=list(seeds), n_timesteps=budget, logging_config=config, **kw
        )
        state = out[0] if isinstance(out, tuple) else out
        jax.block_until_ready(state)
        jax.effects_barrier()
    return Run(agent, state, list(events), tuple(seeds))


def resume(
    run: Run, budget: int, folder: str = "", agent: Any = None, **kw: Any
) -> Run:
    """Continue ``run`` for ``budget`` more steps on ``agent`` (its own by
    default; the call may donate the state: read it first). With ``folder``,
    through a checkpoint restored into the agent's 0-step skeleton, as a new
    process would (ajax/checkpoint.py)."""
    agent, state = agent or run.agent, run.state
    if folder:
        path = os.path.join(folder, "checkpoint.pkl")
        save_checkpoint(state, path)
        state = restore_into(train(agent, run.seeds, 0, **kw).state, path)
    return train(agent, run.seeds, budget, initial_state=state, **kw)


def readings(
    build: Callable[[], Any], read: Callable[[Run], Mapping[str, Any]], **kw: Any
) -> Callable:
    """A case's ``(seeds, budget) -> readings``: build the agent, train it
    once per (seeds, budget) and apply ``read``; cached, copies returned."""

    @functools.cache
    def go(seeds: tuple[int, ...], budget: int) -> dict[str, np.ndarray]:
        return {
            k: np.asarray(v)
            for k, v in read(train(build(), seeds, budget, **kw)).items()
        }

    return lambda seeds, budget: {
        k: v.copy() for k, v in go(tuple(seeds), budget).items()
    }


def iterations(
    agent: str, n_envs: int, budget: int, rollout: int = 1, start: int = 0
) -> list[int]:
    """The post-collection timestep of every training iteration of
    ``rollout`` steps per env: rollout agents run one past the budget
    (train_PPO.py:1446), others at least one."""
    per = n_envs * rollout
    count = budget // per + 1 if agent in ROLLOUT_AGENTS else max(budget // per, 1)
    return [start + k * per for k in range(1, count + 1)]


def contract_readings(run: Run) -> dict[str, np.ndarray]:
    """The bookkeeping readings of a run: final timestep, returns (rolling
    mean, every env's window), logged timesteps and values, optimiser steps,
    parameter checksums and the env's record."""
    s, c = run.state, run.state.collector_state
    out = {
        "timestep": c.timestep,
        "mean return": c.episodic_mean_return,
        "returns": np.asarray(c.episodic_return_state.buffer).reshape(-1),
        "events": run.timesteps(),
        "checksums": readouts.checksums(s.actor_state.params, len(run.seeds)),
    }
    out |= {k: run.values(k) for k in TRAIN_KEYS}
    out |= {f"steps {k}": v for k, v in readouts.optimizer_steps(s).items()}
    for f in ("first_a", "first_u", "last_a", "last_u"):
        with contextlib.suppress(AttributeError):
            out[f] = readouts.record(s, f)
    return {k: np.asarray(v) for k, v in out.items()}


def worker(
    agent: Any, seeds: Sequence[int], budget: int, every: int, folder: str
) -> dict:
    """Train through the real logging process; each run's TensorBoard
    scalars of ``TRAIN_KEYS``."""
    agent.train(
        seed=list(seeds), n_timesteps=budget, logging_config=log_config(every, folder)
    )
    root, out = os.path.join(folder, "tensorboard"), {}
    for run in sorted(os.listdir(root)):
        scalars = load_scalars_from_tfevents(os.path.join(root, run))
        out[run] = {
            k: [v for _, v in scalars.get(k.replace(" ", "_"), [])] for k in TRAIN_KEYS
        }
    return out


def in_subprocess(function: str, args: list, timeout: float) -> Any:
    """Call ``module:function(*args)`` in a fresh interpreter; its JSON."""
    module, name = function.split(":")
    code = f"import json, sys, {module} as m; print(json.dumps(m.{name}(*json.loads(sys.argv[1]))))"
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    cmd = [sys.executable, "-c", code, json.dumps(args)]
    out = subprocess.run(
        cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout
    )
    if out.returncode:
        raise RuntimeError(out.stderr[-4000:])
    return json.loads(out.stdout.strip().splitlines()[-1])
