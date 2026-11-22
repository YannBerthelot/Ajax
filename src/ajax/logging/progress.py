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


def default_log_frequency(n_timesteps: int, iteration_steps: int = 1) -> int:
    """Environment steps between evaluations for about
    :data:`DEFAULT_EVALUATIONS` over ``n_timesteps``: a tenth of them,
    rounded up to a whole number of the agent's iterations of
    ``iteration_steps`` steps, so every evaluation comes after as many
    iterations as the last and the run compiles one length of them."""
    tenth = max(1, -(-int(n_timesteps) // DEFAULT_EVALUATIONS))
    return -(-tenth // iteration_steps) * iteration_steps


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

    def layout(self, rows: str = "seed") -> str:
        """How ``returns`` is laid out, on one line: a row per ``rows``
        (a single seed has a row too), a column per evaluation, with its
        shape."""
        shape = tuple(np.shape(self[RETURN]))
        return (
            f"evaluations.returns has a row per {rows} and a column per"
            f" evaluation: shape {shape}."
        )

    def __repr__(self) -> str:
        # The layout before the arrays: a dict's own repr opens on them,
        # where the row of a single seed reads as one more bracket.
        if RETURN not in self:
            return f"Evaluations({dict.__repr__(self)})"
        names = ", ".join(sorted(self))
        return (
            f"Evaluations: {self.layout()} Every logged name has that"
            f" layout ({names}); dict(evaluations) shows the arrays."
        )


def _flatten(evaluations: Evaluations) -> tuple[list, tuple]:
    names = tuple(sorted(evaluations))
    return [evaluations[name] for name in names], names


def _unflatten(names: tuple, values: Sequence[Any]) -> Evaluations:
    return Evaluations(zip(names, values, strict=True))


# As a dict is: the leaves in the order of the sorted names.
jax.tree_util.register_pytree_node(Evaluations, _flatten, _unflatten)


def summary(scores: Sequence[float], unit: str) -> str:
    """The mean return of ``scores``, with their range when several."""
    text = f"return {np.mean(scores):.1f}"
    if len(scores) > 1:
        text += (
            f" (min {min(scores):.1f}, max {max(scores):.1f} over {len(scores)} {unit})"
        )
    return text


class ProgressPrinter:
    """Prints a run's progress: a first line when training starts, one per
    evaluation, one at the end.

    It is called in the training process with each member's metrics at each
    evaluation (``LoggingConfig.on_evaluation``). ``together``: the members
    train side by side, so an evaluation is one line, their mean return and
    its range, printed once every member's has come; else (a Gymnasium
    environment, one seed after another) each member's evaluation is a line
    of its own. ``groups``: each member's group (a sweep's configuration);
    with more than one, an evaluation's line is followed by one per group.
    ``write`` and ``clock`` are the print and the timer.
    """

    def __init__(
        self,
        title: str,
        labels: Sequence[str],
        end_timestep: int,
        together: bool = True,
        write: Callable[[str], None] = print,
        clock: Callable[[], float] = time.perf_counter,
        groups: Sequence[str] | None = None,
    ) -> None:
        self.title, self.labels = title, list(labels)
        self.groups = list(groups) if groups and len(set(groups)) > 1 else []
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
        text = f"{step:>11,} / {self.end_timestep:,} steps | "
        text += summary(scores, "runs" if self.groups else "seeds")
        if label is not None:
            text += f" | {label}"
        self.write(f"{text} | {elapsed:.1f} s")
        if len(self.groups) != len(scores):
            return
        for group in dict.fromkeys(self.groups):
            own = [s for s, g in zip(scores, self.groups, strict=True) if g == group]
            self.write(f"{'':>14}{group}: {summary(own, 'seeds')}")

    def close(self, evaluations: Any = None) -> None:
        """The last line, with how the returned ``evaluations`` are laid
        out: every call prints it, since a single seed has its row too."""
        text = f"Done in {self.clock() - self.started:.1f} s."
        if isinstance(evaluations, Evaluations) and RETURN in evaluations:
            text += " " + evaluations.layout("run" if self.groups else "seed")
        self.write(text)


def run_logging(
    logging_config: Any,
    agent_name: str,
    seeds: Sequence[int],
    n_timesteps: int,
    end_timestep: int,
    together: bool,
    iteration_steps: int = 1,
    groups: Sequence[str] | None = None,
) -> tuple[Any, Any]:
    """train's ``logging_config`` as the call runs it, and the printer to
    close at its end (None unless the default prints): a LoggingConfig or
    None as given; ``"default"`` becomes about ten evaluations
    (:func:`default_log_frequency`) with a started :class:`ProgressPrinter`
    and no backend, or None for a call of
    no step (one that only builds the state). ``groups``: each member's
    configuration in a sweep, for the printer."""
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
        groups=groups,
    )
    printer.start()
    config = LoggingConfig(
        log_frequency=default_log_frequency(n_timesteps, iteration_steps),
        backends=(),
        on_evaluation=printer,
    )
    return config, printer


def finished(result: Any, printer: Any) -> Any:
    """train's ``(state, out)`` at its end: the printer closed, a dict of
    evaluations as :class:`Evaluations`."""
    state, out = result
    if isinstance(out, dict) and not isinstance(out, Evaluations):
        out = Evaluations(out)
    if printer is not None:
        printer.close(out)
    return state, out
