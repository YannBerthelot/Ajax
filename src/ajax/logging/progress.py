"""What a train call shows and returns without a logging configuration of
its own: a line per evaluation, printed in the training process
(:class:`ProgressPrinter`), and the evaluations by name
(:class:`Evaluations`)."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import jax
import numpy as np

from ajax.logging.wandb_logging import LoggingConfig

RETURN = "Eval/episodic mean reward"
TIMESTEP = "timestep"
# A train call without a logging configuration evaluates about this often.
DEFAULT_EVALUATIONS = 10


def default_log_frequency(n_timesteps: int) -> int:
    """Environment steps between evaluations for :data:`DEFAULT_EVALUATIONS`
    over ``n_timesteps``, rounded up, at least 1."""
    return max(1, -(-int(n_timesteps) // DEFAULT_EVALUATIONS))


class Evaluations(dict):
    """A train call's evaluations: the dict of every logged name to an array
    with a row per member (seed) and a column per evaluation, with the two
    most read by name."""

    @property
    def returns(self) -> Any:
        """The mean evaluation return: a row per member, a column per
        evaluation."""
        return self[RETURN]

    @property
    def timesteps(self) -> Any:
        """The environment step of each evaluation, the same for every
        member."""
        return self[TIMESTEP][0]


def _flatten(evaluations: Evaluations) -> tuple[list, tuple]:
    names = tuple(sorted(evaluations))
    return [evaluations[name] for name in names], names


def _unflatten(names: tuple, values: Sequence[Any]) -> Evaluations:
    return Evaluations(zip(names, values, strict=True))


# As a dict is: the leaves in the order of the sorted names.
jax.tree_util.register_pytree_node(Evaluations, _flatten, _unflatten)


class ProgressPrinter:
    """Prints a run's progress: a first line when training starts, one per
    evaluation, one at the end.

    It is called in the training process with each member's metrics at each
    evaluation (``LoggingConfig.on_evaluation``). ``together``: the members
    train side by side, so an evaluation is one line, their mean return and
    its range, printed once every member's has come; else (a Gymnasium
    environment, one seed after another) each member's evaluation is a line
    of its own. ``write`` and ``clock`` are the print and the timer.
    """

    def __init__(
        self,
        title: str,
        labels: Sequence[str],
        end_timestep: int,
        together: bool = True,
        write: Callable[[str], None] = print,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.title, self.labels = title, list(labels)
        self.end_timestep, self.together = int(end_timestep), together
        self.write, self.clock = write, clock
        self.started = clock()
        self.waiting: dict[int, dict[int, float]] = {}
        self.lock = threading.Lock()

    def start(self) -> None:
        self.started = self.clock()
        self.write(
            f"{self.title}: compiling, then a line per evaluation"
            " (pass logging_config=None for a silent run with none)."
        )

    def __call__(self, metrics: Mapping[str, Any], index: Any) -> None:
        step = int(np.asarray(metrics[TIMESTEP]))
        score = float(np.asarray(metrics.get(RETURN, np.nan)))
        member = int(index)
        with self.lock:
            if not self.together:
                self.line(step, [score], self.labels[member])
                return
            scores = self.waiting.setdefault(step, {})
            scores[member] = score
            if len(scores) == len(self.labels):
                del self.waiting[step]
                self.line(step, [scores[m] for m in sorted(scores)], None)

    def line(self, step: int, scores: list, label: Any) -> None:
        elapsed = self.clock() - self.started
        text = (
            f"{step:>11,} / {self.end_timestep:,} steps | return {np.mean(scores):.1f}"
        )
        if len(scores) > 1:
            text += (
                f" (min {min(scores):.1f}, max {max(scores):.1f}"
                f" over {len(scores)} seeds)"
            )
        if label is not None:
            text += f" | {label}"
        self.write(f"{text} | {elapsed:.1f} s")

    def close(self) -> None:
        self.write(f"Done in {self.clock() - self.started:.1f} s.")


def run_logging(
    logging_config: Any,
    agent_name: str,
    seeds: Sequence[int],
    n_timesteps: int,
    end_timestep: int,
    together: bool,
) -> tuple[Any, Any]:
    """train's ``logging_config`` as the call runs it, and the printer to
    close at its end (None unless the default prints): a LoggingConfig or
    None as given; ``"default"`` becomes about ten evaluations with a
    started :class:`ProgressPrinter` and no backend, or None for a call of
    no step (one that only builds the state)."""
    if not isinstance(logging_config, str):
        return logging_config, None
    if logging_config != "default":
        raise ValueError(
            f"logging_config={logging_config!r}: pass a LoggingConfig,"
            ' None (no evaluation) or leave it out ("default").'
        )
    if n_timesteps < 1:
        return None, None
    printer = ProgressPrinter(
        f"{agent_name}, {len(seeds)} run(s) of {n_timesteps:,} steps",
        [f"seed {seed}" for seed in seeds],
        end_timestep,
        together=together,
    )
    printer.start()
    config = LoggingConfig(
        log_frequency=default_log_frequency(n_timesteps),
        backends=(),
        on_evaluation=printer,
    )
    return config, printer


def finished(result: Any, printer: Any) -> Any:
    """train's ``(state, out)`` at its end: the printer closed, a dict of
    evaluations as :class:`Evaluations`."""
    if printer is not None:
        printer.close()
    state, out = result
    if isinstance(out, dict) and not isinstance(out, Evaluations):
        out = Evaluations(out)
    return state, out
