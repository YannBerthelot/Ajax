"""ReturnNormalizer and RunningScale against the DreamerV3 and TD-MPC2 code.

Pinned numbers are from ``docs/world_models/shared_blocks.md`` (B3),
computed with the literal translations in ``reference_impls``
(``DMoments``: DreamerV3 ``Moments(impl='perc')``; ``TRunningScale``:
TD-MPC2 ``RunningScale``), which are also the oracles of the randomised
comparisons.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.normalizers import ReturnNormalizer, RunningScale

from . import reference_impls as ref

SEEDS = (0, 1, 2)


def _ramp(scale: float) -> np.ndarray:
    """``arange(256) * scale`` as a (256, 1) batch: 5-95 range 229.5 * scale."""
    return np.arange(256, dtype=np.float32)[:, None] * np.float32(scale)


def test_initial_states():
    norm = ReturnNormalizer.create()
    assert (norm.lo, norm.hi, norm.rate, norm.limit) == (0.0, 0.0, 0.01, 1.0)
    assert norm.lo.dtype == norm.hi.dtype == jnp.float32
    assert norm.scale() == 1.0
    scale = RunningScale.create()
    assert (scale.value, scale.rate) == (1.0, 0.01)
    assert scale.value.dtype == jnp.float32


def test_initial_states_can_be_donated():
    """No two leaves of a fresh state alias one buffer, so a jitted update
    can take it by donation (XLA refuses to donate one buffer twice)."""
    x = _ramp(1.0)
    for create in (ReturnNormalizer.create, RunningScale.create):
        update = jax.jit(lambda norm, x: norm.update(x), donate_argnums=0)
        jax.block_until_ready(update(create(), x))


def test_pinned_ramp_sequence():
    """Ranges 0.2295, 2.295, 229.5, 2295 in turn (shared_blocks.md B3)."""
    norm, scale = ReturnNormalizer.create(), RunningScale.create()
    d3_expected = [1.0, 1.0, 2.3199697, 25.246771]
    t2_expected = [1.0, 1.01295, 3.2978203, 26.214842]
    for s, d3, t2 in zip((0.001, 0.01, 1.0, 10.0), d3_expected, t2_expected):
        norm, scale = norm.update(_ramp(s)), scale.update(_ramp(s))
        np.testing.assert_allclose(norm.scale(), d3, rtol=1e-6)
        np.testing.assert_allclose(scale.scale(), t2, rtol=1e-6)
    np.testing.assert_allclose(norm.lo, 1.4025983, rtol=1e-6)
    np.testing.assert_allclose(norm.hi, 26.649368, rtol=1e-6)


def _run(xs):
    """Apply one update of both normalisers per batch of ``xs``, in a jitted scan."""

    def step(carry, x):
        norm, scale = carry
        return (norm.update(x), scale.update(x)), None

    init = (ReturnNormalizer.create(), RunningScale.create())
    (norm, scale), _ = jax.jit(lambda c, x: jax.lax.scan(step, c, x))(init, xs)
    return norm, scale


def test_pinned_constant_range():
    x = (np.linspace(0, 1, 256, dtype=np.float32) * 50 / 0.9)[:, None]
    norm, scale = _run(np.broadcast_to(x, (100, *x.shape)))
    np.testing.assert_allclose(norm.scale(), 31.698427, rtol=1e-5)
    np.testing.assert_allclose(scale.scale(), 32.064415, rtol=1e-5)


def test_pinned_alternating_ranges_separate_clamp_order():
    """Ranges alternating 0 and 3: DreamerV3 clamps after the EMA (EMA of
    the range ~1.5, read as max(1, .)), TD-MPC2 before it (EMA of
    max(1, 0) = 1 and 3, i.e. ~2)."""
    zero = np.zeros((256, 1), np.float32)
    ramp = (np.linspace(0, 1, 256, dtype=np.float32) * 3 / 0.9)[:, None]
    xs = np.stack([zero, ramp] * 1000)
    norm, scale = _run(xs)
    np.testing.assert_allclose(norm.scale(), 1.50753, rtol=1e-5)
    np.testing.assert_allclose(scale.scale(), 2.00502, rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_return_normalizer_matches_reference(seed):
    """Imagined-return-shaped batches (B*K, H) with drifting location/scale:
    bit-identical to the reference run op by op."""
    rng = np.random.default_rng(seed)
    norm, oracle = ReturnNormalizer.create(), ref.DMoments()
    for step in range(30):
        x = rng.normal(size=(64, 15)) * 10.0 ** rng.uniform(-2, 3) + step
        x = x.astype(np.float32)
        norm = norm.update(x)
        offset, span = oracle(jnp.asarray(x))
        np.testing.assert_array_equal(norm.lo, offset)
        np.testing.assert_array_equal(norm.hi, oracle.high)
        np.testing.assert_array_equal(norm.scale(), span)


@pytest.mark.parametrize("seed", SEEDS)
def test_running_scale_matches_reference(seed):
    """(B, 1) batches of t = 0 Q values, as TD-MPC2 passes them."""
    rng = np.random.default_rng(seed)
    scale, oracle = RunningScale.create(), ref.TRunningScale()
    for step in range(30):
        q = rng.normal(size=(256, 1)) * 10.0 ** rng.uniform(-2, 3) - 5.0 * step
        q = q.astype(np.float32)
        scale = scale.update(q)
        oracle.update(q)
        np.testing.assert_allclose(scale.scale(), oracle.value, rtol=1e-6)
        # Same division the reference applies after updating (scale.py:45).
        np.testing.assert_allclose(q / scale.scale(), oracle(q), rtol=1e-6)


def test_custom_rate_and_limit_match_reference():
    rng = np.random.default_rng(0)
    norm = ReturnNormalizer.create(rate=0.3, limit=2.0)
    oracle = ref.DMoments(rate=0.3, limit=2.0)
    scale, scale_oracle = RunningScale.create(rate=0.3), ref.TRunningScale(tau=0.3)
    for _ in range(5):
        x = (rng.normal(size=(128, 1)) * 3.0).astype(np.float32)
        norm, scale = norm.update(x), scale.update(x)
        scale_oracle.update(x)
        np.testing.assert_array_equal(norm.scale(), oracle(jnp.asarray(x))[1])
        np.testing.assert_allclose(scale.scale(), scale_oracle.value, rtol=1e-6)


def test_custom_limit_binds():
    """``limit`` is the floor: at creation (range 0) and after an update
    whose EMA'd range (0.3 * 1) is below it."""
    assert ReturnNormalizer.create(limit=2.0).scale() == 2.0
    x = _ramp(1.0 / 229.5)  # 5-95 range 1
    norm = ReturnNormalizer.create(rate=0.3, limit=2.0).update(x)
    oracle = ref.DMoments(rate=0.3, limit=2.0)
    assert norm.scale() == 2.0 == oracle(jnp.asarray(x))[1]
    assert ReturnNormalizer.create(rate=0.3, limit=0.1).update(x).scale() < 2.0


def test_update_then_read():
    """The current batch already counts in the scale read in the same step."""
    x = _ramp(1000.0 / 229.5)  # 5-95 range 1000
    np.testing.assert_allclose(
        ReturnNormalizer.create().update(x).scale(), 10.0, rtol=1e-5
    )
    np.testing.assert_allclose(
        RunningScale.create().update(x).scale(), 10.99, rtol=1e-5
    )


def test_percentiles_over_all_elements():
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 64, 15)) * 7.0
    flat = x.reshape(-1)
    for create in (ReturnNormalizer.create, RunningScale.create):
        np.testing.assert_array_equal(
            create().update(x).scale(), create().update(flat).scale()
        )
    np.testing.assert_array_equal(
        RunningScale.create().update(x[0, :, :1]).scale(),
        RunningScale.create().update(x[0, :, 0]).scale(),
    )


def test_no_gradient_flows_through_the_scale_or_the_state():
    x = jax.random.normal(jax.random.PRNGKey(0), (256, 1)) * 30.0
    for create in (ReturnNormalizer.create, RunningScale.create):
        np.testing.assert_array_equal(
            jax.grad(lambda v, c=create: c().update(v).scale())(x), 0.0
        )
        state = create().update(x)
        grad = jax.grad(lambda v, c=create: (v / c().update(v).scale()).sum())(x)
        np.testing.assert_allclose(grad, 1.0 / state.scale(), rtol=1e-6)
    # The state itself (read without scale()) carries no gradient either.
    for leaf in ("lo", "hi"):
        grad = jax.grad(
            lambda v, k=leaf: getattr(ReturnNormalizer.create().update(v), k)
        )
        np.testing.assert_array_equal(grad(x), 0.0)
    grad = jax.grad(lambda v: RunningScale.create().update(v).value)
    np.testing.assert_array_equal(grad(x), 0.0)


@pytest.mark.parametrize("create", [ReturnNormalizer.create, RunningScale.create])
def test_percentiles_in_float32_for_low_precision_inputs(create):
    """bf16 returns are upcast before the percentiles, as DreamerV3 does
    (``x.astype(f32)``): the state equals the float32 computation."""
    x = (100.0 * jax.random.normal(jax.random.PRNGKey(0), (256, 1)) + 37.0).astype(
        jnp.bfloat16
    )
    ours = create().update(x)
    upcast = create().update(x.astype(jnp.float32))
    for a, b in zip(jax.tree_util.tree_leaves(ours), jax.tree_util.tree_leaves(upcast)):
        assert a.dtype == jnp.float32
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ours.scale(), upcast.scale())


def test_hyperparameters_are_static_and_state_is_scalar_leaves():
    norm = ReturnNormalizer.create(rate=0.02)
    assert jax.tree_util.tree_leaves(norm) == [norm.lo, norm.hi]
    assert jax.tree_util.tree_leaves(RunningScale.create()) == [1.0]
    traces = []

    @jax.jit
    def update(state, x):
        traces.append(None)
        return state.update(x)

    x = jnp.arange(10.0)
    update(ReturnNormalizer.create(), x)
    update(ReturnNormalizer.create(), x + 1.0)
    assert len(traces) == 1
    update(ReturnNormalizer.create(rate=0.5), x)
    assert len(traces) == 2
