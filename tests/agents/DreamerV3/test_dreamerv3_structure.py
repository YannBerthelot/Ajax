"""Structure of the DreamerV3 world model: BlockLinear, presets, shapes, init.

The 12M shapes are pinned against the real reference
(``fixtures/dreamerv3_wm_shapes_12m.npz``, written by
``docs/world_models/parity/dreamerv3_world_model_fixtures.py`` from
``danijar/dreamerv3@29eb964`` with the ``size12m`` preset, vector obs 24 and
a continuous action of dimension 6) and against the kernel shapes given in
dreamerv3_spec 2.4.
"""

from __future__ import annotations

import functools
import json
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DreamerV3.networks import (
    BlockLinear,
    WorldModel,
    features,
    init_world_model,
    initial_state,
)
from ajax.agents.DreamerV3.state import MODEL_SIZES, DreamerV3Config
from ajax.distributional import TwoHot

from .common import OBS_DIM, TINY, reference_to_ajax

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
TRUNC_STD = 0.87962566103423978  # std of the unit normal truncated at +-2


@functools.lru_cache(maxsize=1)
def _params_12m() -> dict:
    """Initial 12M world model, vector obs 24, continuous action 6."""
    config = DreamerV3Config.from_model_size("12m")
    return init_world_model(jax.random.PRNGKey(1), config, 24, 6)


def _block_diagonal(kernel: np.ndarray) -> np.ndarray:
    g, i, o = kernel.shape
    dense = np.zeros((g * i, g * o), kernel.dtype)
    for k in range(g):
        dense[k * i : (k + 1) * i, k * o : (k + 1) * o] = kernel[k]
    return dense


def test_block_linear_is_a_block_diagonal_dense_layer():
    """Output block k depends on input block k only: forward and gradients
    equal a dense layer whose kernel is block-diagonal (dreamerv3_spec 1.2)."""
    layer = BlockLinear(features=12, blocks=4)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 5, 8))
    params = layer.init(jax.random.PRNGKey(1), x)["params"]
    params = {**params, "bias": jax.random.normal(jax.random.PRNGKey(2), (12,))}
    assert params["kernel"].shape == (4, 2, 3)
    dense = _block_diagonal(np.asarray(params["kernel"]))
    np.testing.assert_allclose(
        layer.apply({"params": params}, x), x @ dense + params["bias"], atol=1e-6
    )
    cotangent = jax.random.normal(jax.random.PRNGKey(3), (3, 5, 12))
    grads = jax.grad(
        lambda p, x: jnp.sum(cotangent * layer.apply({"params": p}, x)), (0, 1)
    )(params, x)
    dense_grads = jax.grad(
        lambda w, x: jnp.sum(cotangent * (x @ w + params["bias"])), (0, 1)
    )(jnp.asarray(dense), x)
    np.testing.assert_allclose(grads[1], dense_grads[1], atol=1e-5)
    np.testing.assert_allclose(
        _block_diagonal(np.asarray(grads[0]["kernel"])),
        np.asarray(dense_grads[0]) * (dense != 0),
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="divisible"):
        layer.init(jax.random.PRNGKey(0), jnp.zeros((1, 6)))


def test_block_linear_init_uses_the_full_input_width_as_fan_in():
    """``Std[W] = 1 / sqrt(I)`` with ``I`` the total input width, ``sqrt(g)``
    smaller than a per-block fan-in would give (dreamerv3_spec 1.2, 1.5)."""
    width, blocks = 2048, 8
    layer = BlockLinear(features=256, blocks=blocks)
    kernel = layer.init(jax.random.PRNGKey(0), jnp.zeros((1, width)))["params"][
        "kernel"
    ]
    assert kernel.shape == (blocks, width // blocks, 256 // blocks)
    expected = 1 / math.sqrt(width)
    np.testing.assert_allclose(np.std(kernel), expected, rtol=0.02)
    assert np.max(np.abs(kernel)) <= 2 * expected / TRUNC_STD * (1 + 1e-6)
    assert np.std(kernel) < 0.5 / math.sqrt(width / blocks)


@pytest.mark.parametrize("size", sorted(MODEL_SIZES))
def test_presets(size):
    """``d = units = hidden``, ``deter = 8 d``, ``classes = d / 16``; 32
    latents and 8 blocks at every size; feature width ``10 d`` (spec 2.1)."""
    d = MODEL_SIZES[size]
    config = DreamerV3Config.from_model_size(size)
    assert (config.units, config.hidden, config.deter, config.classes) == (
        d,
        d,
        8 * d,
        d // 16,
    )
    assert (config.stoch, config.blocks, config.bins) == (32, 8, 255)
    assert config.feat_dim == 10 * d
    assert hash(config) == hash(DreamerV3Config.from_model_size(size))


def test_preset_values_and_overrides():
    """The model dimension of every preset (dreamerv3_spec 2.1; 2411f7d
    ``configs.yaml`` ``size12m`` ... ``size400m``, ``size1m`` from the later
    code) and explicit widths overriding each preset value."""
    assert MODEL_SIZES == {
        "1m": 64,
        "12m": 256,
        "25m": 384,
        "50m": 512,
        "100m": 768,
        "200m": 1024,
        "400m": 1536,
    }
    assert DreamerV3Config() == DreamerV3Config.from_model_size("12m")
    config = DreamerV3Config.from_model_size("12m", deter=512, classes=8, rep_scale=0.2)
    assert (config.units, config.hidden, config.deter, config.classes) == (
        256,
        256,
        512,
        8,
    )
    assert config.rep_scale == 0.2
    config = DreamerV3Config.from_model_size("12m", units=128, hidden=96)
    assert (config.units, config.hidden, config.deter, config.classes) == (
        128,
        96,
        2048,
        16,
    )
    assert config.gamma == pytest.approx(0.996997, abs=1e-6)
    with pytest.raises(ValueError, match="model_size"):
        DreamerV3Config.from_model_size("13m")
    with pytest.raises(ValueError, match="divisible"):
        DreamerV3Config(deter=100, blocks=8)
    with pytest.raises(ValueError, match="unimix"):
        DreamerV3Config(unimix=1.0)
    with pytest.raises(ValueError, match="classes must be positive"):
        DreamerV3Config.from_model_size("1m", classes=0)
    with pytest.raises(ValueError, match="return_horizon"):
        DreamerV3Config(return_horizon=1.0)


def _shapes(params) -> dict[tuple[str, ...], tuple[int, ...]]:
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    return {tuple(k.key for k in path): tuple(leaf.shape) for path, leaf in flat}


def test_12m_world_model_has_the_reference_shapes():
    """Every parameter of the 12M world model has the shape of its
    counterpart in the real reference, except the reward logits: the
    reference emits 256 and drops the last (``nets.py:437-443``; deviations.md
    section 1, "Two-hot output width")."""
    with np.load(FIXTURES / "dreamerv3_wm_shapes_12m.npz") as data:
        meta = json.loads(str(data["meta"]))
    config = DreamerV3Config.from_model_size("12m")
    ours = _shapes(
        jax.eval_shape(
            lambda: init_world_model(
                jax.random.PRNGKey(0), config, meta["obs_dim"], meta["action_dim"]
            )
        )
    )
    theirs = {
        reference_to_ajax(name): tuple(shape) for name, shape in meta["shapes"].items()
    }
    assert set(ours) == set(theirs)
    differing = {k for k in ours if ours[k] != theirs[k]}
    assert differing == {("rew", "out", "kernel"), ("rew", "out", "bias")}
    assert ours[("rew", "out", "kernel")] == (256, 255)
    assert theirs[("rew", "out", "kernel")] == (256, 256)
    count = lambda shapes: sum(math.prod(s) for s in shapes.values())  # noqa: E731
    assert count(theirs) - count(ours) == config.units + 1
    rssm = {k: v for k, v in ours.items() if k[0] == "rssm"}
    assert count(rssm) == 5_783_040  # the reference optimizer's agent/dyn count


def test_12m_rssm_core_matches_the_spec():
    """dreamerv3_spec 2.4 at 12M: per-block input 256 + 768 = 1024;
    ``dynhid`` kernel (8, 1024, 256) with fan-in 8192; ``dyngru`` kernel
    (8, 256, 768) with fan-in 2048; feature width 2560."""
    config = DreamerV3Config.from_model_size("12m")
    shapes = _shapes(
        jax.eval_shape(lambda: init_world_model(jax.random.PRNGKey(0), config, 24, 6))
    )
    assert shapes[("rssm", "dynhid0", "kernel")] == (8, 1024, 256)
    assert shapes[("rssm", "dyngru", "kernel")] == (8, 256, 768)
    assert shapes[("rssm", "dynhid0_norm", "scale")] == (2048,)
    assert config.feat_dim == 2560
    core = sum(
        math.prod(v)
        for k, v in shapes.items()
        if k[0] == "rssm"
        and k[1] in ("dynin0", "dynin1", "dynin2", "dynhid0", "dynhid0_norm", "dyngru")
    )
    assert core == (
        (2048 + 2) * 256 + (512 + 2) * 256 + (6 + 2) * 256  # dynin0/1/2
        + 8 * 1024 * 256 + 2 * 2048  # dynhid0 + its norm
        + 8 * 256 * 768 + 3 * 2048  # dyngru
    )  # fmt: skip
    for name, fan_in in (("dynhid0", 8192), ("dyngru", 2048)):
        kernel = _params_12m()["rssm"][name]["kernel"]
        np.testing.assert_allclose(np.std(kernel), 1 / math.sqrt(fan_in), rtol=0.02)


def test_initial_state_is_zero():
    """The initial ``(h, z)`` is zeros, not learned (dreamerv3_spec 2.2,
    ``nets.py:39-45``): ``deter [*, D]``, ``stoch [*, S, C]``."""
    state = initial_state(TINY, (2, 3))
    assert state.deter.shape == (2, 3, TINY.deter)
    assert state.stoch.shape == (2, 3, TINY.stoch, TINY.classes)
    assert not np.any(state.deter) and not np.any(state.stoch)


def test_zero_initialised_reward_head_predicts_exactly_zero():
    """Reward output kernel and bias are 0 (outscale 0), so the logits are
    uniform and the two-hot prediction is exactly 0, also under jit
    (dreamerv3_spec 1.5, 1.9; paper p.6)."""
    params = init_world_model(jax.random.PRNGKey(0), TINY, OBS_DIM, 2)
    assert not np.any(params["rew"]["out"]["kernel"])
    assert not np.any(params["rew"]["out"]["bias"])
    model = WorldModel(TINY, OBS_DIM)
    deter = 10 * jax.random.normal(jax.random.PRNGKey(1), (7, 3, TINY.deter))
    stoch = jax.nn.one_hot(
        jax.random.randint(jax.random.PRNGKey(2), (7, 3, TINY.stoch), 0, TINY.classes),
        TINY.classes,
    )

    def predict(params):
        logits = model.apply(
            {"params": params},
            features(deter, stoch),
            method=WorldModel.reward_logits,
        )
        return TwoHot.dreamerv3(TINY.bins).decode(logits)

    assert np.all(np.asarray(predict(params)) == 0.0)
    assert np.all(np.asarray(jax.jit(predict)(params)) == 0.0)


def test_output_layer_scales():
    """Decoder output 0.1 (2411f7d), continue 1.0, latent logits 1.0; hidden
    layers 1.0; every bias 0 and every norm scale 1 (dreamerv3_spec 1.5).
    The kernels are truncated normals (deviation D23): ``|W| <= 2 sigma_0``
    with ``sigma_0 = std / 0.8796``, the bound reached by the largest entry
    (an untruncated normal exceeds it, a uniform of that std stays below
    ``0.76`` of it)."""
    config = DreamerV3Config.from_model_size("12m")
    params = _params_12m()
    d = config.units
    stds = {
        ("dec", "out"): 0.1 / math.sqrt(d),
        ("con", "out"): 1 / math.sqrt(d),
        ("rssm", "obslogit"): 1 / math.sqrt(config.hidden),
        ("rssm", "priorlogit"): 1 / math.sqrt(config.hidden),
        ("dec", "mlp", "Dense_1"): 1 / math.sqrt(d),
        ("enc", "Dense_0"): 1 / math.sqrt(24),
    }
    for path, expected in stds.items():
        kernel = params
        for key in path:
            kernel = kernel[key]
        kernel = np.asarray(kernel["kernel"])
        np.testing.assert_allclose(np.std(kernel), expected, rtol=0.05)
        bound = 2 * expected / TRUNC_STD
        assert 0.8 * bound < np.max(np.abs(kernel)) <= bound * (1 + 1e-6), path
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        if path[-1].key == "bias":
            assert not np.any(leaf), path
        if path[-1].key == "scale":
            assert np.all(np.asarray(leaf) == 1.0), path
