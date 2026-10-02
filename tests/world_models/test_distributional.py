"""symlog / symexp and TwoHot against the DreamerV3 and TD-MPC2 reference code.

Pinned numbers are from ``docs/world_models/shared_blocks.md`` (B1, B2),
computed with literal translations of the reference functions on the
references' float32 bins; the same translations (``reference_impls``) are
the oracles for the randomised comparisons. Ajax's bins are the float32
roundings of the float64 values (deviation D22 / T24), so the oracles are
fed Ajax's bins wherever an algorithm is compared bit for bit, and pins
that depend on the bins allow for the bin difference.

What is tested exactly, and to which bound (``eps`` = float32 epsilon):
DreamerV3 encode and loss are bit-identical to the oracles given the same
bins; decode is within ``4 eps E_p|b|`` of the oracle eagerly and under
``jax.jit`` (the oracle's own pair sum under jit is no closer). TD-MPC2's
floor-formula encoder agrees with the generic interpolation to 2e-5
(measured 1.1e-5: the generic one is within 2e-6 of the exact float64
weights, the float32 floor formula up to 8.6e-6 from them because its
positions near 100 have an ulp of 7.6e-6), not the 1e-6 of
``shared_blocks.md`` B2 test 3.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.distributional import TwoHot, symexp, symlog

from . import reference_impls as ref

D3 = TwoHot.dreamerv3()
T2 = TwoHot.tdmpc2()
SEEDS = (0, 1, 2)
EPS = float(np.finfo(np.float32).eps)


def _abs_expectation(two_hot: TwoHot, logits) -> np.ndarray:
    """``E_p|b|``: the condition scale of the expectation ``E_p[b]``.

    A float32 sum of ``p_k b_k`` is accurate to a few ``eps E_p|b|``,
    whatever the order; this is the scale of every decode tolerance.
    """
    probs = np.asarray(jax.nn.softmax(jnp.asarray(logits, jnp.float32)), np.float64)
    return (probs * np.abs(np.asarray(two_hot.bins(), np.float64))).sum(-1)


def _float64_bins(two_hot: TwoHot) -> np.ndarray:
    """The bins in float64 from the full range (not the mirrored half)."""
    grid = np.linspace(-two_hot.limit, two_hot.limit, two_hot.num_bins)
    if two_hot.transform == "identity":
        return np.sign(grid) * np.expm1(np.abs(grid))
    return grid


def _targets(seed: int, n: int = 2000) -> np.ndarray:
    """Targets over the whole range both encoders see.

    Log-uniform magnitudes from 1e-6 to 1e10 with random signs (beyond
    TD-MPC2's range of ~22025 and up to DreamerV3's ~4.85e8 and past it),
    uniform values in [-30, 30], [-1, 1] and [-3e4, 3e4], zeros of both
    signs and far out-of-range values.
    """
    rng = np.random.default_rng(seed)
    magnitudes = 10.0 ** rng.uniform(-6, 10, n)
    signs = rng.choice([-1.0, 1.0], n)
    return np.concatenate(
        [
            magnitudes * signs,
            rng.uniform(-30, 30, n),
            rng.uniform(-1, 1, n),
            rng.uniform(-3e4, 3e4, n),
            [0.0, -0.0, 1e30, -1e30, 4.9e8, -4.9e8],
        ]
    ).astype(np.float32)


def _exact_tdmpc2_weights(y: np.ndarray) -> np.ndarray:
    """TD-MPC2's two-hot weights in float64 on the exact grid -10 + 0.2 k.

    Uses the float32 ``symlog(y)`` so that only the interpolation, not the
    transform, is compared.
    """
    x = np.clip(np.asarray(symlog(jnp.asarray(y))).astype(np.float64), -10.0, 10.0)
    position = (x + 10.0) / 0.2
    k = np.clip(np.floor(position), 0, 99).astype(int)
    offset = position - k
    weights = np.zeros((len(y), 101))
    weights[np.arange(len(y)), k] = 1.0 - offset
    weights[np.arange(len(y)), k + 1] += offset
    return weights


# ------------------------------------------------------------------ B1 symlog


def test_symlog_pinned_values():
    x = jnp.array([-1000.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1000.0], jnp.float32)
    expected = [-6.908755, -0.6931472, -0.4054651, 0.0, 0.4054651, 0.6931472, 6.908755]
    np.testing.assert_allclose(symlog(x), expected, rtol=1e-6)
    np.testing.assert_allclose(symexp(symlog(x)), x, rtol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_symlog_symexp_match_both_references(seed):
    x = _targets(seed)
    x = x[np.abs(x) < 1e30]
    # DreamerV3 uses log1p / expm1 like Ajax: identical.
    np.testing.assert_array_equal(symlog(x), ref.d_symlog(jnp.asarray(x)))
    y = np.asarray(symlog(x))
    np.testing.assert_array_equal(symexp(y), ref.d_symexp(jnp.asarray(y)))
    # TD-MPC2 writes log(1 + |x|) and exp(|x|) - 1: equal up to float32
    # rounding of 1 + |x| (absolute 6e-8 near 0).
    np.testing.assert_allclose(symlog(x), ref.t_symlog(x), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(symexp(y), ref.t_symexp(y), rtol=1e-5, atol=1e-7)


def test_symlog_is_odd_and_inverts_symexp():
    x = _targets(0)
    x = x[np.abs(x) < 1e6]
    np.testing.assert_array_equal(symlog(-x), -symlog(x))
    np.testing.assert_array_equal(symexp(-x[:100]), -symexp(x[:100]))
    np.testing.assert_allclose(symexp(symlog(x)), x, rtol=2e-6, atol=1e-7)


def test_symlog_gradient_matches_reference():
    """Slope 1 next to 0; exactly 0 at 0, as in both references.

    ``sign(x) * log1p(|x|)`` has ``sign(0) = 0``, so the derivative at
    exactly 0 is 0 in the references too (torch's sign and abs give the
    same). ``shared_blocks.md`` B1 test 4 expects 1.0 there, which no
    literal implementation produces.
    """
    points = jnp.array([-2.0, -1e-6, 0.0, 1e-6, 0.5, 3.0], jnp.float32)
    for ours, theirs in ((symlog, ref.d_symlog), (symexp, ref.d_symexp)):
        grad_ours = jax.vmap(jax.grad(ours))(points)
        grad_theirs = jax.vmap(jax.grad(theirs))(points)
        np.testing.assert_array_equal(grad_ours, grad_theirs)
        np.testing.assert_allclose(grad_ours[jnp.array([1, 3])], 1.0, rtol=1e-5)
        assert grad_ours[2] == 0.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_symlog_preserves_dtype(dtype):
    x = jnp.array([-3.0, 0.0, 2.0], dtype)
    assert symlog(x).dtype == dtype
    assert symexp(x).dtype == dtype


# ------------------------------------------------------- B2 TwoHot: config


def test_paper_constructors():
    assert TwoHot.dreamerv3() == TwoHot(num_bins=255, limit=20.0, transform="identity")
    assert TwoHot.tdmpc2() == TwoHot(num_bins=101, limit=10.0, transform="symlog")


def test_twohot_is_a_hashable_jit_static_argument():
    assert hash(TwoHot.dreamerv3()) == hash(TwoHot.dreamerv3())
    assert TwoHot.dreamerv3() != TwoHot.tdmpc2()
    traces = []

    @functools.partial(jax.jit, static_argnums=0)
    def decode(two_hot, logits):
        traces.append(two_hot)
        return two_hot.decode(logits)

    logits = jax.random.normal(jax.random.PRNGKey(0), (300, 255))
    # XLA fuses and reorders the jitted sum: equal to float32 rounding.
    assert np.all(
        np.abs(decode(TwoHot.dreamerv3(), logits) - D3.decode(logits))
        <= 4 * EPS * _abs_expectation(D3, logits)
    )
    decode(TwoHot.dreamerv3(), logits)
    assert len(traces) == 1  # equal instances share the compiled function
    decode(T2, logits[:, :101])
    assert len(traces) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_bins": 254, "limit": 20.0, "transform": "identity"},
        {"num_bins": 1, "limit": 20.0, "transform": "identity"},
        {"num_bins": 101, "limit": 0.0, "transform": "symlog"},
        {"num_bins": 101, "limit": 10.0, "transform": "log"},
    ],
)
def test_invalid_configurations_raise(kwargs):
    with pytest.raises(ValueError):
        TwoHot(**kwargs)


# --------------------------------------------------------- B2 TwoHot: bins


@pytest.mark.parametrize("two_hot", [D3, T2], ids=["dreamerv3", "tdmpc2"])
def test_bins_are_rounded_float64_constants(two_hot):
    """The float32 rounding of the float64 bins, exactly antisymmetric with
    a +0 middle bin, and the same constants under ``jax.jit``."""
    bins = two_hot.bins()
    n = two_hot.num_bins
    assert bins.shape == (n,) and bins.dtype == jnp.float32
    np.testing.assert_array_equal(bins, _float64_bins(two_hot).astype(np.float32))
    assert bins[n // 2] == 0.0 and not np.signbit(bins[n // 2])
    np.testing.assert_array_equal(bins, -bins[::-1])
    np.testing.assert_array_equal(jax.jit(two_hot.bins)(), bins)


def test_dreamerv3_bins_match_reference():
    """The reference evaluates ``symexp(jnp.linspace(-20, 0, 128))`` in
    float32 (nets.py:464-466): up to 1.3e-6 relative from the rounded
    float64 values, mostly from float32 ``linspace`` rounding near 0."""
    bins = D3.bins()
    np.testing.assert_allclose(bins, ref.d_symexp_twohot_bins(255), rtol=1.3e-6)
    np.testing.assert_allclose(bins[128], 0.17055759, rtol=1e-6)
    np.testing.assert_allclose(bins[0], -4.8516518e8, rtol=1e-6)


def test_tdmpc2_bins_match_reference():
    """``torch.linspace(-10, 10, 101)`` (math.py:92) is within 4.8e-7 of the
    rounded float64 grid (measured with torch 2.2.2; its middle bin is
    -1.5e-7); numpy's float32 ``linspace`` is that grid exactly."""
    np.testing.assert_array_equal(
        T2.bins(), np.linspace(-10, 10, 101, dtype=np.float32)
    )


# ------------------------------------------------------- B2 TwoHot: encode


@pytest.mark.parametrize(
    "y, indices, weights",
    [
        (1.0, [131, 132], [0.61732966, 0.38267030]),
        (-3.7, [117, 118], [0.81556690, 0.18443313]),
        (0.0, [127], [1.0]),
        (1e10, [254], [1.0]),
        (-1e10, [0], [1.0]),
        (float(D3.bins()[130]), [130], [1.0]),  # exactly a bin
    ],
)
def test_dreamerv3_encode_pinned(y, indices, weights):
    """Pins computed on the reference's float32 bins, which differ from
    Ajax's by up to 1.3e-6 relative: the weights move by up to 1e-6."""
    w = np.asarray(D3.encode(jnp.float32(y)))
    np.testing.assert_array_equal(np.nonzero(w)[0], indices)
    np.testing.assert_allclose(w[indices], weights, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "y, indices, weights",
    [
        (1.0, [53, 54], [0.5342636, 0.4657364]),
        (-3.7, [42, 43], [0.73781204, 0.26218796]),
        (0.0, [50], [1.0]),
        (1e10, [100], [1.0]),
        (-1e10, [0], [1.0]),
        (22025.0, [99, 100], [1.0681152e-4, 0.99989319]),
    ],
)
def test_tdmpc2_encode_pinned(y, indices, weights):
    """Pins from TD-MPC2's floor formula (``reference_impls.t_two_hot``).

    Agreement within 3e-6; against the exact weights within 3e-6 too (the
    floor formula is 1.9e-6 off at 22025, where its position is ~99.9999).
    """
    w = np.asarray(T2.encode(jnp.float32(y)))
    np.testing.assert_array_equal(np.nonzero(w)[0], indices)
    np.testing.assert_allclose(w[indices], weights, atol=3e-6)
    np.testing.assert_allclose(
        w, _exact_tdmpc2_weights(np.array([y], np.float32))[0], atol=3e-6
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_dreamerv3_encode_matches_reference(seed):
    """Bit-identical to the reference algorithm given the same bins."""
    y = _targets(seed)
    expected = ref.d_twohot_target(jnp.asarray(y), D3.bins())
    np.testing.assert_array_equal(D3.encode(y), expected)


@pytest.mark.parametrize("two_hot", [D3, T2], ids=["dreamerv3", "tdmpc2"])
@pytest.mark.parametrize("seed", SEEDS)
def test_encode_is_the_same_under_jit(two_hot, seed):
    y = _targets(seed)
    np.testing.assert_array_equal(jax.jit(two_hot.encode)(y), two_hot.encode(y))


@pytest.mark.parametrize("seed", SEEDS)
def test_tdmpc2_encode_matches_reference(seed):
    y = _targets(seed)
    ours = np.asarray(T2.encode(y))
    np.testing.assert_allclose(ours, ref.t_two_hot(y[:, None]), rtol=0, atol=2e-5)
    np.testing.assert_allclose(ours, _exact_tdmpc2_weights(y), rtol=0, atol=4e-6)


@pytest.mark.parametrize("two_hot", [D3, T2], ids=["dreamerv3", "tdmpc2"])
@pytest.mark.parametrize("seed", SEEDS)
def test_encode_is_a_two_hot_distribution(two_hot, seed):
    w = np.asarray(two_hot.encode(_targets(seed)))
    assert np.all(w >= 0.0)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-6)
    support = w > 0
    assert np.all(support.sum(-1) <= 2)
    for row in support[support.sum(-1) == 2]:
        lo, hi = np.nonzero(row)[0]
        assert hi == lo + 1


def test_encode_exact_bin_hits_and_out_of_range():
    bins = D3.bins()
    np.testing.assert_array_equal(D3.encode(bins), np.eye(255, dtype=np.float32))
    # TD-MPC2's grid is in symlog space: symexp(bin) maps back up to rounding.
    w = np.asarray(T2.encode(symexp(T2.bins())))
    np.testing.assert_array_equal(np.argmax(w, -1), np.arange(101))
    assert np.all(np.max(w, -1) > 1.0 - 1e-5)
    for two_hot, big in ((D3, 5e8), (T2, 2.3e4)):
        w = np.asarray(two_hot.encode(jnp.array([big, 1e30, -big, -1e30])))
        n = two_hot.num_bins
        np.testing.assert_array_equal(w[:, n - 1], [1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(w[:, 0], [0.0, 0.0, 1.0, 1.0])


def test_interpolation_space_false_friend():
    """Same bins, same target, different weights: raw vs symlog space."""
    raw = TwoHot(num_bins=255, limit=20.0, transform="identity").encode(1.0)
    log = TwoHot(num_bins=255, limit=20.0, transform="symlog").encode(1.0)
    np.testing.assert_array_equal(np.nonzero(raw)[0], [131, 132])
    np.testing.assert_array_equal(np.nonzero(log)[0], [131, 132])
    np.testing.assert_allclose(raw[132], 0.3826703, rtol=1e-5)
    np.testing.assert_allclose(log[132], 0.4015, atol=1e-4)


# ------------------------------------------------------- B2 TwoHot: decode


@pytest.mark.parametrize("two_hot", [D3, T2], ids=["dreamerv3", "tdmpc2"])
@pytest.mark.parametrize("batch", [(), (4,), (1024,)])
@pytest.mark.parametrize("value", [0.0, 1.0, -3.3, 50.0])
def test_decode_of_uniform_logits_is_exactly_zero(two_hot, batch, value):
    """Eagerly and under ``jax.jit``, where XLA fuses multiply-adds; the
    reference's ``p_i b_i + p_j b_j`` pairs give 0.07 to 0.16 there (CPU)."""
    logits = jnp.full((*batch, two_hot.num_bins), value)
    assert np.all(np.asarray(two_hot.decode(logits)) == 0.0)
    assert np.all(np.asarray(jax.jit(two_hot.decode)(logits)) == 0.0)


def test_naive_decode_of_uniform_logits_is_not_zero():
    """The guard the exact-zero test needs: naive sums over DreamerV3's
    bins are far from 0 (paper p.18, "summation order matters")."""
    weighted = np.asarray(jax.nn.softmax(jnp.zeros(255)) * D3.bins())
    naive = functools.reduce(lambda a, b: np.float32(a + b), weighted, np.float32(0))
    assert abs(naive) > 0.1  # left to right
    assert abs(float(jnp.sum(weighted))) > 0.1  # XLA's reduction order


def test_decode_pinned():
    np.testing.assert_allclose(
        D3.decode(jnp.linspace(-1, 1, 255, dtype=jnp.float32)), 2.4558884e7, rtol=1e-5
    )
    np.testing.assert_allclose(
        T2.decode(jnp.linspace(-1, 1, 101, dtype=jnp.float32)), 23.267637, rtol=1e-5
    )


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_dreamerv3_decode_matches_reference(seed, scale, jit):
    """Within ``4 eps E_p|b|`` of the reference run op by op (measured: 1.3
    eagerly, 2.9 under jit, as for the reference's own sum under jit)."""
    logits = scale * jax.random.normal(jax.random.PRNGKey(seed), (300, 255))
    expected = np.asarray(ref.d_twohot_mean(logits, D3.bins()))
    decode = jax.jit(D3.decode) if jit else D3.decode
    error = np.abs(np.asarray(decode(logits), np.float64) - expected)
    assert np.all(error <= 4 * EPS * _abs_expectation(D3, logits))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_tdmpc2_decode_matches_reference(seed, scale, jit):
    """Symmetric vs naive sum, compared in symlog space: within
    ``eps (1 + 8 E_p|b|)`` (measured ``eps (1 + 5.3 E_p|b|)``, 4.8e-6 at
    most); the ``eps`` term is TD-MPC2's ``exp(|x|) - 1``, which rounds
    expectations below ``eps / 2`` to 0."""
    logits = np.asarray(scale * jax.random.normal(jax.random.PRNGKey(seed), (300, 101)))
    expected = np.asarray(symlog(jnp.asarray(ref.t_two_hot_inv(logits)[:, 0])))
    decode = jax.jit(T2.decode) if jit else T2.decode
    error = np.abs(np.asarray(symlog(decode(logits)), np.float64) - expected)
    assert np.all(error <= EPS * (1 + 8 * _abs_expectation(T2, logits)))


@pytest.mark.parametrize("two_hot", [D3, T2], ids=["dreamerv3", "tdmpc2"])
def test_decode_inverts_encode(two_hot):
    y = jnp.array([1.0, -3.7, 123.4, 0.0], jnp.float32)
    logits = jnp.log(jnp.maximum(two_hot.encode(y), 1e-30))
    np.testing.assert_allclose(two_hot.decode(logits), y, rtol=1e-6)


# --------------------------------------------------------- B2 TwoHot: loss


@pytest.mark.parametrize("y", [-1e6, -3.7, 0.0, 1.0, 1e9])
def test_loss_of_uniform_logits_is_log_num_bins(y):
    np.testing.assert_allclose(D3.loss(jnp.zeros(255), y), 5.5412636, rtol=1e-6)
    np.testing.assert_allclose(T2.loss(jnp.zeros(101), y), 4.6151205, rtol=1e-6)


def test_loss_pinned():
    logits = jnp.linspace(-1, 1, 255, dtype=jnp.float32)
    np.testing.assert_allclose(D3.loss(logits, 1.0), 5.669425, rtol=1e-6)
    logits = jnp.linspace(-1, 1, 101, dtype=jnp.float32)
    np.testing.assert_allclose(T2.loss(logits, 1.0), 4.710373, rtol=1e-6)


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_dreamerv3_loss_matches_reference(seed, scale):
    """Bit-identical to the reference given the same bins."""
    y = _targets(seed)[::8]
    logits = scale * jax.random.normal(jax.random.PRNGKey(seed), (len(y), 255))
    expected = -ref.d_twohot_log_prob(logits, jnp.asarray(y), D3.bins())
    np.testing.assert_array_equal(D3.loss(logits, y), expected)


@pytest.mark.parametrize("scale", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_tdmpc2_loss_matches_reference(seed, scale):
    """Equal up to the encoder difference and the log-softmax form.

    ``|loss - ref| <= |w - w_ref|_1 max_k |log p_k| + 3 eps (max|logits| +
    loss)``: the 2e-5 weight tolerance on each of the two bins, and
    ``logits - logsumexp(logits)`` against ``F.log_softmax``, which
    subtracts the max first.
    """
    y = _targets(seed)[::8]
    logits = np.asarray(
        scale * jax.random.normal(jax.random.PRNGKey(seed), (len(y), 101))
    )
    expected = ref.t_soft_ce(logits, y[:, None])[:, 0]
    bound = 4e-5 * np.max(np.abs(jax.nn.log_softmax(logits)), -1) + 3 * EPS * (
        np.max(np.abs(logits), -1) + np.abs(expected)
    )
    assert np.all(np.abs(T2.loss(logits, y) - expected) <= bound)


@pytest.mark.parametrize("two_hot", [D3, T2], ids=["dreamerv3", "tdmpc2"])
def test_loss_gradient_is_softmax_minus_target_and_stops_at_y(two_hot):
    k = two_hot.num_bins
    logits = jax.random.normal(jax.random.PRNGKey(0), (5, k))
    y = jnp.array([1.0, -3.7, 0.0, 250.0, -1e4], jnp.float32)
    grad_logits = jax.grad(lambda lg: two_hot.loss(lg, y).sum())(logits)
    expected = jax.nn.softmax(logits) - two_hot.encode(y)
    np.testing.assert_allclose(grad_logits, expected, atol=1e-6)
    np.testing.assert_allclose(grad_logits.sum(-1), 0.0, atol=1e-6)
    grad_y = jax.grad(lambda t: two_hot.loss(logits, t).sum())(y)
    np.testing.assert_array_equal(grad_y, 0.0)


@pytest.mark.parametrize("two_hot", [D3, T2], ids=["dreamerv3", "tdmpc2"])
def test_shapes_and_dtypes_without_batch_reduction(two_hot):
    k = two_hot.num_bins
    logits = jax.random.normal(jax.random.PRNGKey(1), (4, 7, k)).astype(jnp.bfloat16)
    y = jnp.ones((4, 7), jnp.bfloat16)
    assert two_hot.encode(y).shape == (4, 7, k)
    assert two_hot.encode(y).dtype == jnp.float32
    assert two_hot.decode(logits).shape == (4, 7)
    assert two_hot.decode(logits).dtype == jnp.float32
    loss = two_hot.loss(logits, y)
    assert loss.shape == (4, 7) and loss.dtype == jnp.float32
    # bf16 inputs are cast to float32 first: same values as float32 inputs.
    upcast = logits[2, 3].astype(jnp.float32)
    np.testing.assert_allclose(loss[2, 3], two_hot.loss(upcast, 1.0), rtol=1e-6)
    np.testing.assert_allclose(
        two_hot.decode(logits)[2, 3],
        two_hot.decode(upcast),
        rtol=0,
        atol=4 * EPS * _abs_expectation(two_hot, upcast),
    )
