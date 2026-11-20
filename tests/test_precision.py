"""Compute precision (PROTOTYPE, bf16 study 2026-11-20).

Contract, from the study's design (not from the implementation):
* ``compute_dtype("auto", device)`` is bfloat16 exactly on hardware with
  bfloat16 matrix units -- a TPU, or an NVIDIA GPU of compute capability 8.0
  or more -- and float32 elsewhere (older GPUs, CPUs, devices reporting no
  capability). ``"float32"`` and ``"bfloat16"`` are taken as asked; any other
  value is refused.
* A network in mixed precision starts from the same float32 parameters as in
  float32, returns float32, and its gradients are float32; its output agrees
  with the float32 network's to bfloat16 accuracy.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.networks.networks import Encoder
from ajax.precision import compute_dtype


def device(platform, capability=None):
    return SimpleNamespace(platform=platform, compute_capability=capability)


@pytest.mark.parametrize(
    "fake, expected",
    [
        (device("cuda", "6.1"), "float32"),  # GTX 1080 Ti (Pascal)
        (device("cuda", "7.5"), "float32"),  # RTX 2080 Ti (Turing)
        (device("cuda", "8.0"), "bfloat16"),  # A100: the boundary itself
        (device("cuda", "8.9"), "bfloat16"),  # L40S
        (device("cuda", "10.0"), "bfloat16"),  # "10.0" < "8.0" as text
        (device("gpu", "9.0"), "bfloat16"),
        (device("tpu"), "bfloat16"),
        (device("cpu"), "float32"),
        (device("cuda", None), "float32"),  # no capability reported
        (device("rocm", "gfx90a"), "float32"),  # unchecked vendors fall back
    ],
)
def test_auto_picks_bfloat16_only_on_hardware_with_bfloat16_units(fake, expected):
    assert compute_dtype("auto", fake) == jnp.dtype(expected)


@pytest.mark.parametrize("asked", ["float32", "bfloat16"])
def test_an_explicit_precision_overrides_the_hardware(asked):
    assert compute_dtype(asked, device("cuda", "6.1")) == jnp.dtype(asked)
    assert compute_dtype(asked, device("cuda", "8.9")) == jnp.dtype(asked)


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError):
        compute_dtype("float16", device("cuda", "8.9"))


ARCHITECTURE = ("64", "relu", "64", "relu")


def test_mixed_precision_encoder_starts_from_the_float32_weights():
    """Catches parameters stored in bfloat16, or rounded through it at init."""
    x = jnp.zeros((1, 5))
    key = jax.random.PRNGKey(0)
    full = Encoder(input_architecture=ARCHITECTURE).init(key, x)
    mixed = Encoder(input_architecture=ARCHITECTURE, dtype="bfloat16").init(key, x)
    assert jax.tree.structure(full) == jax.tree.structure(mixed)
    for ours, theirs in zip(jax.tree.leaves(full), jax.tree.leaves(mixed), strict=True):
        assert theirs.dtype == jnp.float32
        np.testing.assert_array_equal(ours, theirs)


def test_mixed_precision_encoder_returns_float32_close_to_float32():
    """Catches features left in bfloat16 (heads and losses would follow them)
    and a cast that corrupts values instead of rounding them."""
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (8, 5))
    variables = Encoder(input_architecture=ARCHITECTURE).init(key, x)
    full = Encoder(input_architecture=ARCHITECTURE).apply(variables, x)
    mixed = Encoder(input_architecture=ARCHITECTURE, dtype="bfloat16").apply(
        variables, x
    )
    assert mixed.dtype == jnp.float32
    # bfloat16 keeps 8 significant bits: ~0.4% per rounding, a few roundings
    # deep; 3% of the features' scale bounds it with room.
    scale = float(jnp.max(jnp.abs(full)))
    np.testing.assert_allclose(mixed, full, atol=0.03 * scale)


def test_mixed_precision_gradients_reach_float32_parameters():
    """Catches gradients taken in bfloat16 (the optimiser would then update
    float32 weights from bfloat16 moments) or cut at the cast."""
    key = jax.random.PRNGKey(2)
    x = jax.random.normal(key, (8, 5))
    encoder = Encoder(input_architecture=ARCHITECTURE, dtype="bfloat16")
    variables = Encoder(input_architecture=ARCHITECTURE).init(key, x)
    gradients = jax.grad(lambda v: jnp.sum(encoder.apply(v, x) ** 2))(variables)
    for leaf in jax.tree.leaves(gradients):
        assert leaf.dtype == jnp.float32
        assert bool(jnp.any(leaf != 0))
