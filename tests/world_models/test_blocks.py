"""linear, NormedMLP and their initializers against the DreamerV3 / TD-MPC2 code.

Pinned numbers are from ``docs/world_models/shared_blocks.md`` (B4, B5).
The forward passes are compared with literal translations of the
reference MLPs (``reference_impls.d_mlp``: DreamerV3 MLP of Linear ->
RMSNorm -> SiLU; ``reference_impls.t_mlp_hidden``: TD-MPC2 NormedLinear
stack with dropout in the first layer), fed the same parameters, randomised
so that norm scales and shifts are exercised.
"""

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.distributional import TwoHot
from ajax.networks.blocks import NormedMLP, linear
from ajax.networks.utils import parse_initialization

from . import reference_impls as ref

SEEDS = (0, 1, 2)
FLAX_TRUNC_FACTOR = 1.0 / 0.87962566103423978  # flax variance_scaling


def _perturbed(params, seed):
    """Every leaf replaced by a random array of the same shape (std 0.5, mean 1
    for norm scales), so a forward comparison sees non-trivial values."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    new = []
    for (path, leaf), key in zip(leaves, keys):
        noise = 0.5 * jax.random.normal(key, leaf.shape, leaf.dtype)
        is_scale = jax.tree_util.keystr(path).endswith("['scale']")
        new.append(1.0 + noise if is_scale else noise / math.sqrt(leaf.shape[0]))
    return jax.tree_util.tree_unflatten(treedef, new)


def _d3_layers(params, n):
    p = params["params"]
    return [
        (
            p[f"Dense_{i}"]["kernel"],
            p[f"Dense_{i}"]["bias"],
            p[f"RMSNorm_{i}"]["scale"],
        )
        for i in range(n)
    ]


def _t2_layers(params, n):
    p = params["params"]
    return [
        tuple(
            np.asarray(a)
            for a in (
                p[f"Dense_{i}"]["kernel"],
                p[f"Dense_{i}"]["bias"],
                p[f"LayerNorm_{i}"]["scale"],
                p[f"LayerNorm_{i}"]["bias"],
            )
        )
        for i in range(n)
    ]


# ------------------------------------------------------------- B5 initializers


@pytest.mark.parametrize("outscale", [1.0, 0.1, 0.01])
@pytest.mark.parametrize("shape", [(256, 512), (8, 1024, 256), (64, 3)])
def test_trunc_normal_fan_in_is_dreamerv3_init_up_to_its_constant(shape, outscale):
    """Same key: flax's 1/0.87962566 vs DreamerV3's literal 1.1368.

    The kernels differ by the constant factor 1.13684723 / 1.1368, a
    relative difference of 4.16e-5 (also for the rank-3 BlockLinear kernel,
    whose fan-in is the full input width).
    """
    key = jax.random.PRNGKey(0)
    ours = parse_initialization(f"trunc_normal_fan_in({outscale})")(key, shape)
    theirs = ref.d_init_normal(key, shape, scale=outscale)
    relative = np.abs(ours - theirs) / np.abs(theirs)
    assert 4.1e-5 < relative.min() and relative.max() < 4.2e-5
    np.testing.assert_allclose(ours * (1.1368 / FLAX_TRUNC_FACTOR), theirs, rtol=1e-6)
    # linear's outscale on the unscaled initializer is the same distribution.
    layer = linear(shape[-1], "trunc_normal_fan_in", outscale=outscale)
    assert ref.d_fans(shape)[0] == shape[-2] * int(np.prod(shape[:-2]))
    if len(shape) == 2:
        kernel = layer.init(key, jnp.zeros((1, shape[0])))["params"]["kernel"]
        assert np.allclose(np.std(kernel), outscale / math.sqrt(shape[0]), rtol=0.05)


@pytest.mark.parametrize(
    "shape, expected_std", [((256, 512), 0.0625), ((8, 1024, 256), 0.011048543)]
)
def test_trunc_normal_fan_in_statistics(shape, expected_std):
    w = parse_initialization("trunc_normal_fan_in")(jax.random.PRNGKey(1), shape)
    np.testing.assert_allclose(np.std(w), expected_std, rtol=0.01)
    fan_in = shape[-2] * int(np.prod(shape[:-2]))
    assert np.max(np.abs(w)) <= 2.0 * FLAX_TRUNC_FACTOR / math.sqrt(fan_in) * (1 + 1e-6)


def test_tdmpc2_init_is_an_untruncated_normal():
    """torch ``trunc_normal_(std=0.02)`` truncates at the absolute +-2, i.e.
    100 standard deviations: N(0, 0.02^2). jax's ``truncated_normal(0.02)``
    cuts at +-0.04 (2 sigma) and has std 0.0176: the false friend."""
    shape = (256, 512)
    w = np.asarray(parse_initialization("normal(0.02)")(jax.random.PRNGKey(2), shape))
    np.testing.assert_allclose(w.std(), 0.02, rtol=0.01)
    np.testing.assert_allclose(w.mean(), 0.0, atol=2e-4)
    tail = np.mean(np.abs(w) > 0.04)
    np.testing.assert_allclose(tail, 2 * (1 - 0.977249868), atol=3e-3)  # 2(1 - Phi(2))
    assert np.max(np.abs(w)) > 0.08  # 4 sigma: present among 131072 draws
    false_friend = jax.nn.initializers.truncated_normal(0.02)(
        jax.random.PRNGKey(2), shape
    )
    assert np.max(np.abs(false_friend)) <= 0.04
    np.testing.assert_allclose(np.std(false_friend), 0.0176, rtol=0.01)


# ---------------------------------------------------------------- linear


def test_linear_parameters_and_forward():
    layer = linear(7, "normal(0.02)")
    assert isinstance(layer, nn.Dense)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 5))
    params = layer.init(jax.random.PRNGKey(1), x)
    p = params["params"]
    assert set(p) == {"kernel", "bias"}
    assert p["kernel"].shape == (5, 7) and p["kernel"].dtype == jnp.float32
    assert p["bias"].shape == (7,) and p["bias"].dtype == jnp.float32
    np.testing.assert_array_equal(p["bias"], 0.0)
    p = _perturbed(params, 0)
    y = layer.apply(p, x)
    assert y.shape == (3, 4, 7)
    expected = np.asarray(x) @ np.asarray(p["params"]["kernel"]) + np.asarray(
        p["params"]["bias"]
    )
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kernel_init", ["trunc_normal_fan_in", "normal(0.02)"])
def test_linear_outscale_scales_the_kernel_only(kernel_init):
    x = jnp.ones((2, 16))
    key = jax.random.PRNGKey(3)
    base = linear(8, kernel_init).init(key, x)["params"]
    scaled = linear(8, kernel_init, outscale=0.01).init(key, x)["params"]
    np.testing.assert_allclose(scaled["kernel"], 0.01 * base["kernel"], rtol=1e-6)
    zero = linear(8, kernel_init, outscale=0.0).init(key, x)["params"]
    np.testing.assert_array_equal(zero["kernel"], 0.0)
    for p in (base, scaled, zero):
        np.testing.assert_array_equal(p["bias"], 0.0)


def test_linear_accepts_initializer_callables():
    init = jax.nn.initializers.constant(0.5)
    p = linear(3, init, bias_init=jax.nn.initializers.ones).init(
        jax.random.PRNGKey(0), jnp.ones((1, 2))
    )["params"]
    np.testing.assert_array_equal(p["kernel"], 0.5)
    np.testing.assert_array_equal(p["bias"], 1.0)


def test_linear_outscale_never_scales_the_bias():
    p = linear(3, "ones_init", bias_init="ones_init", outscale=0.5).init(
        jax.random.PRNGKey(0), jnp.ones((1, 2))
    )["params"]
    np.testing.assert_array_equal(p["kernel"], 0.5)
    np.testing.assert_array_equal(p["bias"], 1.0)


def test_linear_name_in_a_compact_module():
    class Head(nn.Module):
        @nn.compact
        def __call__(self, x):
            return linear(4, "zeros", name="reward")(linear(3, "zeros")(x))

    params = Head().init(jax.random.PRNGKey(0), jnp.ones((1, 2)))["params"]
    assert set(params) == {"Dense_0", "reward"}


class _ZeroHeadModel(nn.Module):
    """A paper trunk followed by a two-hot head with ``outscale = 0``."""

    two_hot: TwoHot

    @nn.compact
    def __call__(self, x):
        if self.two_hot.transform == "identity":
            h = NormedMLP.dreamerv3(1, 32)(x)
            init = "trunc_normal_fan_in"
        else:
            h = NormedMLP.tdmpc2(2, 32, dropout=0.01)(x, deterministic=True)
            init = "normal(0.02)"
        logits = linear(self.two_hot.num_bins, init, outscale=0.0)(h)
        return self.two_hot.decode(logits)


@pytest.mark.parametrize("two_hot", [TwoHot.dreamerv3(), TwoHot.tdmpc2()])
def test_zero_initialised_two_hot_head_predicts_exactly_zero(two_hot):
    """Reward / value heads at init: outscale 0 -> uniform softmax -> 0,
    also when the trunk, head and decode are compiled together (XLA then
    fuses the decode's multiply-adds)."""
    model = _ZeroHeadModel(two_hot)
    x = 10.0 * jax.random.normal(jax.random.PRNGKey(0), (6, 9))
    params = model.init(jax.random.PRNGKey(1), x)
    np.testing.assert_array_equal(model.apply(params, x), 0.0)
    np.testing.assert_array_equal(jax.jit(model.apply)(params, x), 0.0)


# ------------------------------------------------------------- NormedMLP


def test_paper_presets():
    d3 = NormedMLP.dreamerv3(3, 256)
    assert (d3.act, d3.norm, d3.norm_eps, d3.kernel_init, d3.dropout) == (
        "silu",
        "rms",
        1e-4,
        "trunc_normal_fan_in",
        0.0,
    )
    t2 = NormedMLP.tdmpc2(2, 512, dropout=0.01)
    assert (t2.act, t2.norm, t2.norm_eps, t2.kernel_init, t2.dropout) == (
        "mish",
        "layer",
        1e-5,
        "normal(0.02)",
        0.01,
    )


@pytest.mark.parametrize(
    "mlp, expected_norm, expected_act",
    [
        (
            NormedMLP.dreamerv3(1, 4),
            [0.36514595, 0.7302919, 1.0954379, 1.4605838],
            [0.21554038, 0.4928516, 0.8209259, 1.1854419],  # silu of the above
        ),
        (
            NormedMLP.tdmpc2(1, 4),
            [-1.3416355, -0.44721183, 0.44721183, 1.3416355],
            [-0.30609322, -0.2046668, 0.32911763, 1.2311375],  # mish of the above
        ),
    ],
    ids=["dreamerv3-rms-1e-4-silu", "tdmpc2-layer-1e-5-mish"],
)
def test_norm_epsilon_and_activation_pinned(mlp, expected_norm, expected_act):
    """Identity kernel: the norm sees [1, 2, 3, 4]. flax's default epsilon
    1e-6 would give -1.3416403 as the first LayerNorm entry."""
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]])
    params = mlp.init(jax.random.PRNGKey(0), x)
    params["params"]["Dense_0"]["kernel"] = jnp.eye(4)
    y, state = mlp.apply(
        params, x, capture_intermediates=True, mutable=["intermediates"]
    )
    norm_name = "RMSNorm_0" if mlp.norm == "rms" else "LayerNorm_0"
    normed = state["intermediates"][norm_name]["__call__"][0]
    np.testing.assert_allclose(normed[0], expected_norm, rtol=1e-6)
    np.testing.assert_allclose(y[0], expected_act, rtol=1e-6)


def test_layer_norm_uses_two_pass_variance():
    """Mean-1000 inputs: the fast variance E[x^2] - E[x]^2 (flax's default)
    is off by ~0.2; the two-pass variance of TD-MPC2's torch LayerNorm is
    within float32 rounding of the mean (~7e-5) of the exact result."""
    x = 1000.0 + jax.random.normal(jax.random.PRNGKey(0), (32, 64))
    mlp = NormedMLP.tdmpc2(1, 64)
    params = mlp.init(jax.random.PRNGKey(1), x)
    params["params"]["Dense_0"]["kernel"] = jnp.eye(64)
    _, state = mlp.apply(
        params, x, capture_intermediates=True, mutable=["intermediates"]
    )
    ours = state["intermediates"]["LayerNorm_0"]["__call__"][0]
    exact = ref.t_layer_norm(np.asarray(x, np.float64), np.ones(64), np.zeros(64))
    np.testing.assert_allclose(ours, exact, atol=2e-4)
    torch_port = ref.t_layer_norm(
        np.asarray(x), np.ones(64, np.float32), np.zeros(64, np.float32)
    )
    np.testing.assert_allclose(ours, torch_port, atol=3e-4)
    fast = nn.LayerNorm(epsilon=1e-5).apply(
        {"params": params["params"]["LayerNorm_0"]}, x
    )
    assert np.max(np.abs(fast - exact)) > 1e-2  # the test discriminates


@pytest.mark.parametrize("seed", SEEDS)
def test_dreamerv3_mlp_matches_reference(seed):
    """Dense -> RMSNorm -> SiLU per layer on (B, T, F), DreamerV3's MLP."""
    mlp = NormedMLP.dreamerv3(3, 32)
    x = 3.0 * jax.random.normal(jax.random.PRNGKey(seed), (4, 5, 12))
    params = _perturbed(mlp.init(jax.random.PRNGKey(seed + 10), x), seed)
    expected = ref.d_mlp(x, _d3_layers(params, 3))
    np.testing.assert_allclose(mlp.apply(params, x), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_tdmpc2_mlp_matches_reference_without_dropout(seed):
    mlp = NormedMLP.tdmpc2(2, 32, dropout=0.01)
    x = 3.0 * jax.random.normal(jax.random.PRNGKey(seed), (16, 12))
    params = _perturbed(
        mlp.init(jax.random.PRNGKey(seed + 10), x, deterministic=True), seed
    )
    expected = ref.t_mlp_hidden(np.asarray(x), _t2_layers(params, 2))
    ours = mlp.apply(params, x, deterministic=True)
    np.testing.assert_allclose(ours, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_tdmpc2_mlp_matches_reference_with_dropout(seed):
    """Dropout sits between the first Dense and its LayerNorm (layers.py:96-100)."""
    p = 0.3  # TD-MPC2 uses 0.01; a larger rate makes the mask visible
    mlp = NormedMLP.tdmpc2(2, 32, dropout=p)
    x = 3.0 * jax.random.normal(jax.random.PRNGKey(seed), (64, 12))
    params = _perturbed(
        mlp.init(jax.random.PRNGKey(seed + 10), x, deterministic=True), seed
    )
    rngs = {"dropout": jax.random.PRNGKey(seed + 20)}
    ours, state = mlp.apply(
        params,
        x,
        deterministic=False,
        rngs=rngs,
        capture_intermediates=True,
        mutable=["intermediates"],
    )
    mask = np.asarray(state["intermediates"]["Dropout_0"]["__call__"][0] != 0)
    np.testing.assert_allclose(mask.mean(), 1 - p, atol=0.03)
    expected = ref.t_mlp_hidden(np.asarray(x), _t2_layers(params, 2), p, mask)
    np.testing.assert_allclose(ours, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("preset", ["dreamerv3", "tdmpc2"])
def test_layer_order_from_intermediates(preset):
    """Dense -> [Dropout] -> Norm -> Act, layer by layer."""
    if preset == "dreamerv3":
        mlp, norm_name, act = NormedMLP.dreamerv3(2, 16), "RMSNorm", jax.nn.silu
    else:
        mlp, norm_name, act = (
            NormedMLP.tdmpc2(2, 16, dropout=0.2),
            "LayerNorm",
            jax.nn.mish,
        )
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 6))
    params = _perturbed(mlp.init(jax.random.PRNGKey(1), x, deterministic=True), 0)
    y, state = mlp.apply(
        params,
        x,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(2)},
        capture_intermediates=True,
        mutable=["intermediates"],
    )
    inter = {
        k: v["__call__"][0]
        for k, v in state["intermediates"].items()
        if k != "__call__"
    }
    p = params["params"]
    expected_modules = {"Dense_0", "Dense_1", f"{norm_name}_0", f"{norm_name}_1"}
    if preset == "tdmpc2":
        expected_modules.add("Dropout_0")  # first layer only
    assert set(inter) == expected_modules
    h = x
    for i in range(2):
        np.testing.assert_allclose(
            inter[f"Dense_{i}"],
            h @ p[f"Dense_{i}"]["kernel"] + p[f"Dense_{i}"]["bias"],
            rtol=1e-5,
            atol=1e-6,
        )
        norm_in = (
            inter["Dropout_0"]
            if (i == 0 and preset == "tdmpc2")
            else inter[f"Dense_{i}"]
        )
        norm = (
            nn.RMSNorm(epsilon=1e-4)
            if preset == "dreamerv3"
            else nn.LayerNorm(1e-5, use_fast_variance=False)
        )
        np.testing.assert_allclose(
            inter[f"{norm_name}_{i}"],
            norm.apply({"params": p[f"{norm_name}_{i}"]}, norm_in),
            rtol=1e-6,
            atol=1e-6,
        )
        h = act(inter[f"{norm_name}_{i}"])
    np.testing.assert_allclose(y, h, rtol=1e-6)


def test_dropout_only_when_not_deterministic_and_uses_the_dropout_rng():
    mlp = NormedMLP.tdmpc2(2, 32, dropout=0.01)
    no_dropout = NormedMLP.tdmpc2(2, 32)
    x = jax.random.normal(jax.random.PRNGKey(0), (128, 12))
    params = mlp.init(jax.random.PRNGKey(1), x, deterministic=True)
    # Same parameter tree with or without dropout (Dropout has no parameters).
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
        no_dropout.init(jax.random.PRNGKey(1), x)
    )
    reference = no_dropout.apply(params, x)
    np.testing.assert_array_equal(mlp.apply(params, x, deterministic=True), reference)
    k1, k2 = jax.random.PRNGKey(2), jax.random.PRNGKey(3)
    train1 = mlp.apply(params, x, deterministic=False, rngs={"dropout": k1})
    assert not np.allclose(train1, reference)
    np.testing.assert_array_equal(
        train1, mlp.apply(params, x, deterministic=False, rngs={"dropout": k1})
    )
    assert not np.allclose(
        train1, mlp.apply(params, x, deterministic=False, rngs={"dropout": k2})
    )
    with pytest.raises(Exception, match="dropout"):
        mlp.apply(params, x, deterministic=False)  # no 'dropout' rng
    with pytest.raises(ValueError, match="deterministic"):
        mlp.apply(params, x)  # dropout > 0 needs an explicit mode


def test_parameter_shapes_and_counts():
    d3 = NormedMLP.dreamerv3(3, 256)
    d3_params = d3.init(jax.random.PRNGKey(0), jnp.zeros((1, 100)))["params"]
    assert d3_params["RMSNorm_0"]["scale"].shape == (256,)
    assert set(d3_params["RMSNorm_0"]) == {"scale"}  # no shift
    n_d3 = sum(x.size for x in jax.tree_util.tree_leaves(d3_params))
    assert n_d3 == (100 * 256 + 2 * 256) + 2 * (256 * 256 + 2 * 256)
    t2 = NormedMLP.tdmpc2(2, 512, dropout=0.01)
    t2_params = t2.init(jax.random.PRNGKey(0), jnp.zeros((1, 518)), deterministic=True)
    assert set(t2_params["params"]["LayerNorm_0"]) == {"scale", "bias"}
    head = linear(101, "normal(0.02)", outscale=0.0)
    head_params = head.init(jax.random.PRNGKey(1), jnp.zeros((1, 512)))
    count = sum(x.size for x in jax.tree_util.tree_leaves((t2_params, head_params)))
    # TD-MPC2 walker (latent 512, action 6) reward head / one Q member.
    assert count == 582_245
    pi = NormedMLP.tdmpc2(2, 512).init(jax.random.PRNGKey(0), jnp.zeros((1, 512)))
    pi_head = linear(12, "normal(0.02)").init(
        jax.random.PRNGKey(1), jnp.zeros((1, 512))
    )
    assert sum(x.size for x in jax.tree_util.tree_leaves((pi, pi_head))) == 533_516


def test_statistics_in_float32():
    """bf16 inputs are promoted: the norm statistics and outputs are float32
    and equal the float32 computation on the upcast input."""
    mlp = NormedMLP.dreamerv3(2, 16)
    x = (100.0 * jax.random.normal(jax.random.PRNGKey(0), (4, 8))).astype(jnp.bfloat16)
    params = _perturbed(mlp.init(jax.random.PRNGKey(1), x), 0)
    y = mlp.apply(params, x)
    assert y.dtype == jnp.float32
    for leaf in jax.tree_util.tree_leaves(params):
        assert leaf.dtype == jnp.float32
    expected = ref.d_mlp(x.astype(jnp.float32)[None], _d3_layers(params, 2))[0]
    np.testing.assert_allclose(y, expected, rtol=1e-5, atol=1e-6)


def test_leading_dimensions_are_batch_dimensions():
    mlp = NormedMLP.dreamerv3(2, 16)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 5, 7))
    params = mlp.init(jax.random.PRNGKey(1), x)
    flat = mlp.apply(params, x.reshape(15, 7)).reshape(3, 5, 16)
    np.testing.assert_allclose(mlp.apply(params, x), flat, rtol=1e-6)


def test_callable_activation_and_invalid_norm():
    mlp = NormedMLP(
        1, 4, act=jax.nn.relu, norm="rms", norm_eps=1e-4, kernel_init="zeros"
    )
    params = mlp.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))
    params["params"]["Dense_0"]["bias"] = jnp.array([-1.0, 1.0, -2.0, 2.0])
    np.testing.assert_allclose(
        mlp.apply(params, jnp.ones((1, 3)))[0],
        jax.nn.relu(jnp.array([-1.0, 1.0, -2.0, 2.0]) / math.sqrt(2.5 + 1e-4)),
        rtol=1e-6,
    )
    bad = NormedMLP(1, 4, act="silu", norm="batch", norm_eps=1e-4, kernel_init="zeros")
    with pytest.raises(ValueError, match="norm"):
        bad.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))
