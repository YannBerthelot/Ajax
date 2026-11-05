"""The representation metrics on matrices whose answers are known.

Contract: effective rank is exp(entropy) of the singular values normalised to
sum to 1 (Roy & Vetterli 2007), so k equal singular values give k. Linear CKA
(Kornblith et al. 2019) is 1 on identical inputs, unchanged by a rotation, a
scale or a shift of either input. A unit is dead when no state changes its
output. Expected values are worked out by hand in the docstrings.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.representation.metrics import (
    dead_unit_fraction,
    effective_rank,
    linear_cka,
    mean_activation_norm,
)


def orthonormal_columns(n_rows: int, n_columns: int) -> np.ndarray:
    random = np.random.default_rng(0).normal(size=(n_rows, n_columns))
    return np.linalg.qr(random)[0]


@pytest.mark.parametrize("rank", [1, 2, 5])
def test_equal_singular_values_give_their_count(rank):
    """Catches a rank without its exponential (the entropy log k) or counted
    on the wrong axis: three times k orthonormal columns has k singular values
    of 3, whatever the matrix's width."""
    k_columns = 3.0 * orthonormal_columns(32, rank)
    matrix = k_columns @ orthonormal_columns(8, rank).T

    assert effective_rank(jnp.asarray(matrix, jnp.float32)) == pytest.approx(
        rank, rel=1e-4
    )


def test_unequal_singular_values_follow_the_entropy_of_their_shares():
    """Catches normalising the squared singular values, the other convention
    (which gives exp(H(.9, .1)) = 1.38). Singular values 3 and 1 have the
    shares (.75, .25), H = .5623 and exp(H) = 1.7548."""
    columns = orthonormal_columns(16, 2)
    matrix = columns @ np.diag([3.0, 1.0])

    assert effective_rank(jnp.asarray(matrix, jnp.float32)) == pytest.approx(
        1.7548, rel=1e-3
    )


def test_a_layer_of_zeros_has_rank_zero():
    """Catches a NaN in the logs for the layer it matters most for: 0/0 in
    the shares of a layer that has died entirely."""
    assert effective_rank(jnp.zeros((8, 4))) == 0.0


def test_cka_of_a_layer_with_itself_is_one():
    """Catches a kernel product missing its normalisation."""
    layer = jnp.asarray(np.random.default_rng(1).normal(size=(64, 16)), jnp.float32)

    assert linear_cka(layer, layer) == pytest.approx(1.0, abs=1e-5)


def test_cka_ignores_the_units_order_and_scale():
    """Catches a comparison of units one by one: two layers holding the same
    representation in other coordinates must read as the same."""
    layer = np.random.default_rng(2).normal(size=(64, 8))
    rotated = 7.0 * layer @ orthonormal_columns(8, 8)

    first, second = (jnp.asarray(x, jnp.float32) for x in (layer, rotated))
    assert linear_cka(first, second) == pytest.approx(1.0, abs=1e-5)


def test_cka_ignores_a_shift_of_the_layer():
    """Catches kernels left uncentred: a ReLU layer's positive mean would
    otherwise read as similarity. Over these rows, YtX = (2, 0), the Gram
    norms are sqrt(8) and 2, so CKA = 4 / (sqrt(8) * 2) = 1 / sqrt(2)."""
    layer = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    part = layer[:, :1]

    expected = 1 / np.sqrt(2)
    first, second = jnp.asarray(layer + 5.0), jnp.asarray(part - 3.0)
    assert linear_cka(first, second) == pytest.approx(expected, rel=1e-5)


def test_cka_of_uncorrelated_layers_is_zero():
    """Catches an alignment that cannot fall: these columns are orthogonal
    across the states, so YtX is all zeros."""
    first = jnp.array([[1.0], [-1.0], [0.0], [0.0]])
    second = jnp.array([[0.0], [0.0], [1.0], [-1.0]])

    assert linear_cka(first, second) == 0.0


def test_cka_of_large_activations_stays_finite():
    """Catches the fourth powers overflowing float32: a critic's features can
    reach 1e6 on an unscaled reward, and CKA does not depend on the scale."""
    layer = jnp.asarray(np.random.default_rng(3).normal(size=(64, 8)), jnp.float32)

    assert linear_cka(1e6 * layer, 1e6 * layer) == pytest.approx(1.0, abs=1e-5)


def test_cka_against_a_constant_layer_is_undefined():
    """Catches a made-up similarity for the first probe, which compares with
    nothing yet: NaN, which the logger drops, not 0 or 1."""
    layer = jnp.arange(12.0).reshape(4, 3)

    assert jnp.isnan(linear_cka(layer, jnp.zeros((4, 3))))


def test_activation_norm_averages_the_states_l2_norms():
    """Catches a norm over the whole matrix. Here (5 + 0) / 2."""
    assert mean_activation_norm(jnp.array([[3.0, 4.0], [0.0, 0.0]])) == 2.5


def test_a_unit_is_dead_when_no_state_changes_it():
    """Catches counting zeros alone, which misses a saturated tanh unit, or
    reading a quiet but live unit as dead. Units: stuck at 0, varying, pinned
    at 1, varying a little: two of four dead."""
    activations = jnp.array(
        [[0.0, 0.5, 1.0, 0.1], [0.0, -0.5, 1.0, 0.1001], [0.0, 2.0, 1.0, 0.1002]]
    )

    assert dead_unit_fraction(activations) == 0.5
