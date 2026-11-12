"""Evaluation in a thread of its own, on a device of its own, while training
goes on: ``LoggingConfig(evaluation_device=...)``.

A logged run evaluates between chunks of training
(:func:`ajax.perf_utils.build_chunked_train`). With an evaluation device, the
evaluation program still computes what it reads of training (the update's
metrics, the extensions' metrics, the evaluation count) and hands the host
the rest: the state without its replay, the key of its timestep
(:func:`ajax.log.evaluation_keys`) and those metrics. Training continues at
once. A thread scores every seed's policy of that timestep together on the
device, then sends each seed's row to the logging worker under its own
timestep. The train call waits for the last one and writes the scores into
the rows it returns. They are the inline evaluation's: the same policy at the
same step, on the same keys, up to floating-point rounding.

On a GPU, two programs of one process run one after the other, so the device
is another GPU or the CPU. On the CPU, a MuJoCo Warp run is evaluated on MJX's
pure-JAX physics (:func:`ajax.evaluate.evaluation_physics`).
"""

from __future__ import annotations

import functools
import queue
import threading
from typing import Any, Callable, Optional

import jax
import numpy as np

#: The fields holding a replay or stored rollouts: no evaluation reads them,
#: and copying them to the host at each evaluation would cost more than it.
REPLAY_FIELDS = ("buffer_state", "replay_state", "last_rollout", "rollout")


def without_replay(agent_state: Any) -> Any:
    """``agent_state`` with its replay and stored rollouts set to ``None``, on
    the state and on its collector: an evaluation reading one fails as it
    traces instead of scoring on a stand-in."""

    def drop(state: Any) -> Any:
        held = [
            name for name in REPLAY_FIELDS if getattr(state, name, None) is not None
        ]
        return state.replace(**dict.fromkeys(held)) if held else state

    agent_state = drop(agent_state)
    collector = getattr(agent_state, "collector_state", None)
    if collector is None:
        return agent_state
    return agent_state.replace(collector_state=drop(collector))


class AsyncEvaluator:
    """Scores the evaluations of one train call of ``n_seeds`` seeds on
    ``device``, in a thread, and logs each seed's row through ``log_fn(row,
    seed index)``. An evaluation is scored once every seed's has come."""

    def __init__(self, device: Any, n_seeds: int, log_fn: Callable) -> None:
        self.device = device
        self.n_seeds = n_seeds
        self.log_fn = log_fn
        self.lock = threading.Lock()
        self.waiting: dict = {}  # timestep -> {seed index: (state, key, metrics)}
        self.jobs: queue.Queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.scored: list = []  # each evaluation's scores, leaves (seeds,)
        self.error: Optional[Exception] = None

    def hand_over(
        self,
        flag: Any,
        scores: Callable,
        agent_state: Any,
        key: Any,
        metrics: dict,
        index: Any,
    ) -> None:
        """In the evaluation program, in place of logging ``metrics``: when
        ``flag`` is set, hand the host what ``scores(agent_state, key)``
        reads, and the metrics to log with its scores. Under the seed vmap,
        JAX calls the host once per seed."""
        receive = functools.partial(self.receive, scores)
        jax.debug.callback(
            receive, flag, without_replay(agent_state), key, metrics, index
        )

    def receive(
        self,
        scores: Callable,
        flag: Any,
        state: Any,
        key: Any,
        metrics: dict,
        index: Any,
    ) -> None:
        if not bool(flag):
            return
        # Copies: a callback's arrays may be the program's own buffers.
        handed = jax.tree.map(np.array, (state, key, metrics))
        timestep = int(metrics["timestep"])
        with self.lock:
            seeds = self.waiting.setdefault(timestep, {})
            seeds[int(index)] = handed
            if len(seeds) < self.n_seeds:
                return
            del self.waiting[timestep]
            if self.thread is None:
                self.thread = threading.Thread(target=self.run, daemon=True)
                self.thread.start()
        self.jobs.put((scores, [seeds[i] for i in range(self.n_seeds)]))

    def run(self) -> None:
        programs: dict = {}
        job = self.jobs.get()
        while job is not None:
            if self.error is None:
                try:
                    self.score(programs, *job)
                except Exception as error:  # re-raised by wait
                    self.error = error
            job = self.jobs.get()

    def score(self, programs: dict, scores: Callable, seeds: list) -> None:
        """Score one evaluation of every seed together, then log each row."""
        states, keys, rows = zip(*seeds, strict=True)
        stacked = (
            jax.tree.map(lambda *leaves: np.stack(leaves), *states),
            np.stack(keys),
        )
        with jax.default_device(self.device):
            if scores not in programs:
                programs[scores] = jax.jit(jax.vmap(scores))
            inputs = jax.device_put(stacked, self.device)
            result = jax.device_get(programs[scores](*inputs))
        self.scored.append(result)
        for index, row in enumerate(rows):
            row = dict(row)
            row.update({name: values[index] for name, values in result.items()})
            self.log_fn(row, index)

    def wait(self) -> dict:
        """Wait for every evaluation handed over (after the programs that
        hand them over have run: :func:`jax.effects_barrier`), then return
        their scores, each ``(seeds, evaluations)``, and start afresh."""
        if self.thread is not None:
            self.jobs.put(None)
            self.thread.join()
        scored, error, waiting = self.scored, self.error, self.waiting
        self.thread, self.scored, self.error, self.waiting = None, [], None, {}
        if error is not None:
            raise RuntimeError("The asynchronous evaluation failed.") from error
        if waiting:
            raise RuntimeError(
                f"Evaluations at timesteps {sorted(waiting)} came from"
                f" fewer than the {self.n_seeds} seeds."
            )
        if not scored:
            return {}
        return {
            name: np.stack([scores[name] for scores in scored], axis=1)
            for name in scored[0]
        }
