"""What a layer's activations on a fixed set of states say about it.

Pure functions of ``(n_states, n_units)`` arrays, compiled inside the
training loop at each probe.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import entr

#: The spread under which a unit counts as constant: a few float32 ulps at 1,
#: the largest value a saturated tanh unit reaches.
DEAD_UNIT_SPREAD = 1e-6


def effective_rank(activations: jax.Array) -> jax.Array:
    """``exp(H(p))``, with ``p`` the singular values normalised to sum to 1
    (Roy & Vetterli, EUSIPCO 2007): k for k equal singular values, between 1
    and k when some dominate, 0 for a layer that is zero on every state."""
    singular_values = jnp.linalg.svd(activations, compute_uv=False)
    total = singular_values.sum()
    return jnp.where(total > 0, jnp.exp(entr(singular_values / total).sum()), 0.0)


def linear_cka(activations: jax.Array, other: jax.Array) -> jax.Array:
    """Linear centred kernel alignment (Kornblith et al., ICML 2019) of two
    layers on the same states, the units' order and scale aside: 1 when one
    is a rotation and scaling of the other. NaN when either is constant
    across the states, which it cannot compare."""

    def centred(features: jax.Array) -> jax.Array:
        features = features - features.mean(axis=0)
        # Its value does not depend on the scale; the squares below overflow
        # float32 on large activations without this.
        return features / jnp.linalg.norm(features)

    first, second = centred(activations), centred(other)
    alignment = jnp.sum(jnp.square(second.T @ first))
    return alignment / (
        jnp.linalg.norm(first.T @ first) * jnp.linalg.norm(second.T @ second)
    )


def mean_activation_norm(activations: jax.Array) -> jax.Array:
    """The L2 norm of a state's activations, averaged over the states."""
    return jnp.linalg.norm(activations, axis=-1).mean()


def dead_unit_fraction(
    activations: jax.Array, spread: float = DEAD_UNIT_SPREAD
) -> jax.Array:
    """The fraction of units whose output no state changes: a ReLU stuck at
    zero (Lyle et al., ICML 2023) or a tanh pinned at +-1. Either carries no
    information about the state, whatever its value."""
    unit_spread = activations.max(axis=0) - activations.min(axis=0)
    return jnp.mean(unit_spread <= spread)
