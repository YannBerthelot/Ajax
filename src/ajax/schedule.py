from collections.abc import Callable
from typing import Optional, Union

import jax.numpy as jnp
from flax import struct

ArrayLike = Union[float, jnp.ndarray]


# ----------------------------
# Schedules
# ----------------------------


from collections.abc import Callable
from typing import Optional

import jax.numpy as jnp

# ----------------------------
# Window helper
# ----------------------------


def _window_schedule(core_fn, start: float, end: float, min_value: float = 0.0):
    """
    Applies a schedule only inside [start, end].

    Before start -> 1
    After end -> min_value
    """

    length = end - start

    def schedule(t: float):
        # normalize to [0,1]
        u = (t - start) / length
        u = jnp.clip(u, 0.0, 1.0)

        val = core_fn(u)

        val = jnp.where(t < start, 1.0, val)
        val = jnp.where(t > end, min_value, val)

        return val

    return schedule


# ----------------------------
# Core schedules
# ----------------------------


def _linear_core(u, min_value=0.0):
    return min_value + (1 - min_value) * (1 - u)


def _exponential_core(u, p):
    return jnp.exp(-p * u)


def _polynomial_core(u, p):
    return (1.0 - u) ** p


# ----------------------------
# Schedule picker
# ----------------------------


def schedule_picker(name: Optional[str]) -> Callable[[float], float]:
    if name is None:
        return lambda _: 1.0

    parts = name.split("_")
    kind = parts[0]

    # ----------------------------
    # Map base names to full window 0 → 1
    if kind in ("linear", "exponential", "polynomial") and len(parts) == 1:
        parts = [kind, "0", "1"]  # default window 0 → 1

    # ----------------------------
    # Linear
    if kind == "linear":
        if len(parts) not in (3, 4):
            raise ValueError(f"Invalid linear schedule: {name}")

        start = float(parts[1])
        end = float(parts[2])
        min_value = 0.0
        if len(parts) == 4:
            min_value = float(parts[3]) / 100.0

        return _window_schedule(
            lambda u: _linear_core(u, min_value), start, end, min_value
        )

    # ----------------------------
    # Exponential
    if kind == "exponential":
        if len(parts) not in (3, 4):
            raise ValueError(f"Invalid exponential schedule: {name}")

        start = float(parts[1])
        end = float(parts[2])
        p = 5.0
        if len(parts) == 4:
            p = float(parts[3])

        return _window_schedule(lambda u: _exponential_core(u, p), start, end, 0.0)

    # ----------------------------
    # Polynomial
    if kind == "polynomial":
        if len(parts) not in (3, 4):
            raise ValueError(f"Invalid polynomial schedule: {name}")

        start = float(parts[1])
        end = float(parts[2])
        p = 2
        if len(parts) == 4:
            p = float(parts[3])

        return _window_schedule(lambda u: _polynomial_core(u, p), start, end, 0.0)

    raise ValueError(f"Unknown schedule: {name}")


# ----------------------------
# Scheduled parameter
# ----------------------------


@struct.dataclass(eq=False)
class Scheduled:
    value: ArrayLike
    schedule_fn: Callable = struct.field(pytree_node=False)

    # ----------------------------
    # Hash only static parts
    # ----------------------------

    def __hash__(self):
        return hash((Scheduled, self.schedule_fn))

    # ----------------------------
    # Evaluate schedule
    # ----------------------------

    def at(self, train_frac: float):
        return self.value * self.schedule_fn(train_frac)

    # ----------------------------
    # Arithmetic helpers
    # ----------------------------

    def mul(self, other, train_frac):
        return self.at(train_frac) * other

    def add(self, other, train_frac):
        return self.at(train_frac) + other


# ----------------------------
# Constructor
# ----------------------------


def make_scheduled(
    x: ArrayLike,
    schedule_name: Optional[str] = None,
) -> Scheduled:
    return Scheduled(
        value=x,
        schedule_fn=schedule_picker(schedule_name),
    )
