"""Compute precision: float32, or mixed precision with bfloat16 compute.

PROTOTYPE (bf16 study, 2026-11-20): not wired into the public surface yet.

Mixed precision keeps the parameters, gradients and optimiser state in
float32 and runs the network layers in bfloat16: each layer casts its input
and its float32 parameters to bfloat16, the matrix products accumulate in
float32 and round their output to bfloat16, and normalisation statistics are
computed in float32 (flax's ``force_float32_reductions``). Every network
returns float32, so what follows it -- distributions, log-probabilities,
losses, targets, returns, normalisers -- stays float32. This is the policy
of DreamerV3's reference code (``danijar/dreamerv3``, ``compute_dtype:
bfloat16``, ``param_dtype: float32``).

bfloat16 matrix products run on tensor cores from NVIDIA compute capability
8.0 (Ampere) on, and on TPUs. Older GPUs (Turing 7.5, Pascal 6.1) and CPUs
have no bfloat16 arithmetic: XLA computes it in float32 and converts around
it, so ``"auto"`` falls back to float32 there.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

#: The compute precisions a run may ask for.
PRECISIONS = ("float32", "bfloat16", "auto")


def has_fast_bfloat16(device: Optional[Any] = None) -> bool:
    """Whether ``device`` (default: JAX's first device) multiplies bfloat16
    matrices in hardware: a TPU, or an NVIDIA GPU of compute capability 8.0
    or more."""
    device = jax.devices()[0] if device is None else device
    if device.platform == "tpu":
        return True
    if device.platform in ("gpu", "cuda"):
        capability = getattr(device, "compute_capability", None)
        if capability is None:
            return False
        major = int(str(capability).split(".")[0])
        return major >= 8
    return False


def compute_dtype(precision: str, device: Optional[Any] = None) -> jnp.dtype:
    """The layers' compute dtype for ``precision``: ``"float32"``,
    ``"bfloat16"`` (even where the hardware emulates it), or ``"auto"``
    (bfloat16 where :func:`has_fast_bfloat16`, else float32)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    if precision == "auto":
        precision = "bfloat16" if has_fast_bfloat16(device) else "float32"
    return jnp.dtype(precision)


def cast_floating(tree: Any, dtype: Any) -> Any:
    """``tree`` with its float32 leaves cast to ``dtype`` (parameters read
    in the compute dtype; the cast's gradient reaches the float32 ones)."""

    def cast(leaf: Any) -> Any:
        if hasattr(leaf, "dtype") and leaf.dtype == jnp.float32:
            return leaf.astype(dtype)
        return leaf

    return jax.tree.map(cast, tree)


def to_float32(tree: Any) -> Any:
    """``tree`` with its floating leaves in float32 (a network's outputs)."""

    def cast(leaf: Any) -> Any:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf.astype(jnp.float32)
        return leaf

    return jax.tree.map(cast, tree)
