"""Schedules: hyperparameters that change over training."""

from collections.abc import Callable
from typing import Any, Union

import jax.numpy as jnp
import optax

#: A hyperparameter over training: a constant, or a function of the
#: timestep. The timestep is traced, so a function is written with ``jnp``
#: (``jnp.clip``, ``jnp.where``), never a Python ``if`` or ``min``.
Schedule = Union[float, Callable[[Any], Any]]


def progress(timestep: Any, horizon: Any) -> Any:
    """The share of a run of ``horizon`` steps (at least one) done at
    ``timestep``, unclipped: what every schedule over training reads."""
    return timestep / jnp.maximum(horizon, 1)


def at(schedule: Schedule, timestep: Any) -> Any:
    """``schedule``'s value at ``timestep``."""
    return schedule(timestep) if callable(schedule) else schedule


def linear(start: float, end: float, steps: float) -> Callable[[Any], Any]:
    """From ``start`` at timestep 0 to ``end`` at ``steps`` (at least one),
    then ``end``."""
    steps = max(steps, 1.0)
    return lambda timestep: start + jnp.clip(timestep / steps, 0.0, 1.0) * (end - start)


# ----------------------------
# Optimizer step schedules
# ----------------------------


def warmup_cosine_schedule(
    peak_value: float,
    warmup_steps: int,
    total_steps: int,
    end_value_fraction: float = 0.1,
    init_value_fraction: float = 0.0,
) -> Callable[[int], float]:
    """Linear warmup to ``peak_value`` then cosine decay to ``end_value``.

    An optax step schedule (``step -> learning_rate``) usable as
    ``OptimizerConfig.learning_rate``. The warmup-cosine profile is the
    one the in-context controller reference code trains with (Busetto et
    al. 2024, GPT-style: 5k warmup steps, decay to ``peak / 10``).

    ``warmup_steps=0`` gives pure cosine decay from ``peak_value``.
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if not 0 <= warmup_steps < total_steps:
        # optax needs at least one decay step after the warmup.
        raise ValueError(
            f"need 0 <= warmup_steps < total_steps, got {warmup_steps} and {total_steps}"
        )
    return optax.warmup_cosine_decay_schedule(
        init_value=peak_value * init_value_fraction,
        peak_value=peak_value,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=peak_value * end_value_fraction,
    )
