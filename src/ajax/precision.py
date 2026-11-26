"""Compute precision: float32, or mixed precision with bfloat16 layers.

``ActorCritic.train(precision=...)`` takes ``"float32"``, ``"bfloat16"`` or
``"auto"``; ``None`` is the agent's default, its original code's precision
(float32 for every agent so far).

Mixed precision keeps the parameters, gradients and optimiser state in
float32 and runs the layers in bfloat16: each layer reads its float32
parameters cast to bfloat16, its matrix products add up in float32 and
round their output to bfloat16, and flax computes normalisation statistics
in float32. Every network returns float32, so what follows it --
distributions, log-probabilities, losses, targets, returns, normalisers --
stays float32. This is the policy of DreamerV3's original code
(``danijar/dreamerv3@2411f7d``, ``dreamerv3/configs.yaml:24-25``:
``compute_dtype: bfloat16``, ``param_dtype: float32``); every other agent's
original computes in float32, so bfloat16 changes their numbers.

bfloat16 matrix products run on tensor cores from NVIDIA compute capability
8.0 (Ampere) on, and on TPUs. Older GPUs (Turing 7.5, Pascal 6.1) and CPUs
have no bfloat16 arithmetic: XLA computes it in float32 and converts around
it, no faster than float32 (on an RTX 2080 Ti about equal, on a CPU slower)
while the numbers still change. ``"auto"`` therefore gives bfloat16 only
where every device of the run has bfloat16 units, and float32 elsewhere.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

#: The compute precisions a run may ask for.
PRECISIONS = ("float32", "bfloat16", "auto")


def has_fast_bfloat16(device: Any) -> bool:
    """Whether ``device`` multiplies bfloat16 matrices in hardware: a TPU,
    or an NVIDIA GPU of compute capability 8.0 or more. A device that
    reports no capability (or another vendor's, not checked) counts as
    without."""
    if device.platform == "tpu":
        return True
    if device.platform in ("gpu", "cuda"):
        capability = getattr(device, "compute_capability", None)
        try:
            major = int(str(capability).split(".")[0])
        except ValueError:
            return False
        return major >= 8
    return False


def resolve_precision(precision: str, devices: Optional[Sequence[Any]] = None) -> str:
    """The compute dtype's name, ``"float32"`` or ``"bfloat16"``, for a run
    asking for ``precision`` on ``devices`` (None: JAX's default device).

    ``"auto"`` is bfloat16 when every device has bfloat16 units
    (:func:`has_fast_bfloat16`), so that the members of a run split over
    devices compute alike; ``"bfloat16"`` is taken even where the hardware
    emulates it, with a warning."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    devices = list(devices) if devices else [jax.devices()[0]]
    fast = all(has_fast_bfloat16(device) for device in devices)
    if precision == "auto":
        return "bfloat16" if fast else "float32"
    if precision == "bfloat16" and not fast:
        logger.warning(
            "precision='bfloat16' on %s, which has no bfloat16 units: XLA"
            " emulates it, no faster than float32.",
            ", ".join(sorted({str(device.device_kind) for device in devices})),
        )
    return precision


def cast_floating(tree: Any, dtype: Any) -> Any:
    """``tree`` with its float32 leaves cast to ``dtype``: parameters read in
    the compute dtype, whose gradient reaches the float32 ones."""

    def cast(leaf: Any) -> Any:
        if hasattr(leaf, "dtype") and leaf.dtype == jnp.float32:
            return leaf.astype(dtype)
        return leaf

    return jax.tree.map(cast, tree)
