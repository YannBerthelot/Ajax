"""TD-MPC2 network structure: parameter counts, initialisation, dropout, sizes.

Pinned numbers are from ``docs/world_models/tdmpc2_spec.md`` 1.20 and 1.24
(paper App. H; ``nicklashansen/tdmpc2@5f6fade:tdmpc2/common/world_model.py``,
``layers.py``, ``init.py``).
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.TDMPC2.core import create_update_state
from ajax.agents.TDMPC2.networks import (
    MODEL_SIZE,
    make_policy_prior,
    make_world_model,
    simnorm,
    simnorm_head,
)
from ajax.agents.TDMPC2.state import TDMPC2Config

TINY = TDMPC2Config(latent_dim=16, enc_dim=32, mlp_dim=32, num_q=5, dropout=0.01)


def _count(tree) -> int:
    return sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(tree))


@pytest.mark.parametrize(
    "obs_dim, action_dim, task_dim, expected",
    [
        # Single-task walker (S = 24, A = 6), spec 1.24.
        (
            24,
            6,
            0,
            {
                "encoder": 139_520,
                "dynamics": 794_112,
                "reward": 582_245,
                "q": 2_911_225,
                "pi": 533_516,
                "total": 4_960_618,
            },
        ),
        # Multi-task inputs (S = 39 padded, A = 6, task_dim 96), spec 1.24; the
        # 7,680-parameter embedding table is not part of these modules (M8).
        (
            39,
            6,
            96,
            {
                "encoder": 167_936,
                "dynamics": 843_264,
                "reward": 631_397,
                "q": 3_156_985,
                "pi": 582_668,
                "total": 5_389_930 - 7_680,
            },
        ),
    ],
)
def test_parameter_counts_match_the_paper_at_5m(
    obs_dim, action_dim, task_dim, expected
):
    config = TDMPC2Config.from_model_size(5)
    state = create_update_state(
        jax.random.PRNGKey(0), config, obs_dim, action_dim, task_dim=task_dim
    )
    wm = state.world_model_state.params
    counts = {name: _count(wm[name]) for name in ("encoder", "dynamics", "reward", "q")}
    counts["pi"] = _count(state.actor_state.params)
    counts["total"] = sum(counts.values())
    assert counts == expected
    # Target Q is excluded from the counts (spec 1.24) but mirrors Q.
    assert _count(state.world_model_state.target_params) == expected["q"]


def _widths(model_size, **kwargs):
    config = TDMPC2Config.from_model_size(model_size, **kwargs)
    return {k: getattr(config, k) for k in MODEL_SIZE[model_size]}


def test_model_size_presets_and_explicit_overrides():
    assert _widths(5) == {
        "enc_dim": 256,
        "mlp_dim": 512,
        "latent_dim": 512,
        "num_enc_layers": 2,
        "num_q": 5,
    }
    assert _widths(1)["num_q"] == 2
    assert _widths(317) == {
        "enc_dim": 4096,
        "mlp_dim": 4096,
        "latent_dim": 1376,
        "num_enc_layers": 5,
        "num_q": 8,
    }
    assert [MODEL_SIZE[s]["num_enc_layers"] for s in (1, 5, 19, 48, 317)] == [
        2,
        2,
        3,
        4,
        5,
    ]
    assert _widths(19, mlp_dim=64, num_q=3, latent_dim=None) == {
        "enc_dim": 1024,
        "mlp_dim": 64,
        "latent_dim": 768,
        "num_enc_layers": 3,
        "num_q": 3,
    }
    config = TDMPC2Config.from_model_size(48, latent_dim=64, horizon=5)
    assert (config.enc_dim, config.latent_dim, config.horizon) == (1792, 64, 5)
    with pytest.raises(ValueError, match="model_size"):
        TDMPC2Config.from_model_size(7)


def test_config_is_hashable_and_validated():
    assert hash(TDMPC2Config()) == hash(TDMPC2Config())
    assert TDMPC2Config().replace(num_q=2).num_q == 2
    with pytest.raises(ValueError, match="num_q"):
        TDMPC2Config(num_q=1)
    with pytest.raises(ValueError, match="simnorm_dim"):
        TDMPC2Config(latent_dim=12)


def test_simnorm_softmaxes_groups_of_eight():
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 32)) * 5.0
    y = simnorm(x, 8)
    assert y.shape == x.shape
    groups = y.reshape(3, 4, 4, 8)
    np.testing.assert_allclose(groups.sum(-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(groups, jax.nn.softmax(x.reshape(3, 4, 4, 8), -1))
    with pytest.raises(ValueError, match="divisible"):
        simnorm(jnp.zeros((2, 12)), 8)


@pytest.fixture(scope="module")
def tiny_state():
    return create_update_state(jax.random.PRNGKey(0), TINY, 5, 2)


def test_encoder_and_dynamics_outputs_are_on_the_simplex(tiny_state):
    wm = tiny_state.world_model_state
    obs = jax.random.normal(jax.random.PRNGKey(1), (3, 7, 5))  # arbitrary leading axes
    z = wm.apply_fn({"params": wm.params}, obs, method="encode")
    assert z.shape == (3, 7, 16)
    np.testing.assert_allclose(z.reshape(3, 7, 2, 8).sum(-1), 1.0, rtol=1e-6)
    z2 = wm.apply_fn({"params": wm.params}, z, jnp.zeros((3, 7, 2)), method="next")
    np.testing.assert_allclose(z2.reshape(3, 7, 2, 8).sum(-1), 1.0, rtol=1e-6)


def test_zero_initialised_heads_predict_exactly_zero(tiny_state):
    """Zero final weights and biases: every reward / Q logit is 0 and the
    decoded prediction is exactly 0 (spec 1.19; symmetric decode, T24)."""
    wm = tiny_state.world_model_state
    params = wm.params
    for head in (params["reward"]["out"], params["q"]["members"]["out"]):
        assert not np.any(head["kernel"]) and not np.any(head["bias"])
    z = jax.random.uniform(jax.random.PRNGKey(2), (4, 16))
    a = jax.random.uniform(jax.random.PRNGKey(3), (4, 2), minval=-1, maxval=1)
    reward = wm.apply_fn({"params": params}, z, a, method="reward_logits")
    q = wm.apply_fn({"params": params}, z, a, method="q_logits")
    assert q.shape == (5, 4, 101) and reward.shape == (4, 101)
    two_hot = TINY.two_hot
    assert not np.any(two_hot.decode(reward)) and not np.any(two_hot.decode(q))
    assert not np.any(jax.jit(two_hot.decode)(q))


def test_initialisation_follows_the_reference(tiny_state):
    """N(0, 0.02^2) kernels, zero biases, unit LayerNorm scales (the SimNorm
    heads included); independent Q members; the target Q is a copy; the
    policy's last layer is not zeroed (spec 1.13)."""
    wm = tiny_state.world_model_state
    members = wm.params["q"]["members"]["trunk"]["Dense_0"]["kernel"]
    assert members.shape == (5, 18, 32)
    for i in range(5):
        for j in range(i + 1, 5):
            assert not np.allclose(members[i], members[j])
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, wm.target_params, wm.params["q"]
    )
    big = create_update_state(
        jax.random.PRNGKey(1), TDMPC2Config.from_model_size(1), 24, 6
    )
    kernel = big.world_model_state.params["dynamics"]["trunk"]["Dense_0"]["kernel"]
    assert abs(float(np.std(kernel)) - 0.02) < 2e-4
    assert float(np.abs(kernel).max()) > 0.07  # untruncated: > 3.5 std occurs
    pi_out = big.actor_state.params["out"]["kernel"]
    assert abs(float(np.std(pi_out)) - 0.02) < 2e-3
    assert not np.any(big.actor_state.params["out"]["bias"])
    for part in ("encoder", "dynamics"):
        head = big.world_model_state.params[part]["head"]
        assert abs(float(np.std(head["Dense_0"]["kernel"])) - 0.02) < 1e-3, part
        assert not np.any(head["Dense_0"]["bias"]), part
        norm = head["LayerNorm_0"]
        assert np.all(norm["scale"] == 1.0) and not np.any(norm["bias"]), part


@pytest.mark.parametrize("offset, spread", [(1000.0, 1.0), (0.0, 3e-3)])
def test_simnorm_head_is_linear_torch_layer_norm_and_simnorm(offset, spread):
    """``Linear -> LayerNorm -> SimNorm`` against float64 numpy with torch's
    two-pass variance and epsilon 1e-5 (``layers.py:85-100``): mean-1000
    inputs expose a fast ``E[x^2] - E[x]^2`` variance, a 3e-3 spread the
    epsilon."""
    head = simnorm_head(16, 8)
    x = jnp.eye(4, 6)
    params = head.init(jax.random.PRNGKey(0), x)["params"]
    kernel = (spread * np.random.default_rng(0).normal(size=(6, 16))).astype(np.float32)
    params["Dense_0"] = {
        "kernel": jnp.asarray(kernel),
        "bias": jnp.full((16,), offset, jnp.float32),
    }
    h = np.eye(4, 6) @ kernel.astype(np.float64) + offset
    h = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-5)
    g = np.exp(h.reshape(4, 2, 8))
    expected = (g / g.sum(-1, keepdims=True)).reshape(4, 16)
    np.testing.assert_allclose(head.apply({"params": params}, x), expected, atol=2e-4)


@pytest.mark.parametrize("num_enc_layers, hidden", [(1, 1), (2, 1), (3, 2)])
def test_encoder_has_max_of_layers_minus_one_and_one_hidden_layers(
    num_enc_layers, hidden
):
    """``max(num_enc_layers - 1, 1)`` hidden layers (``layers.py:148``)."""
    config = TINY.replace(num_enc_layers=num_enc_layers)
    state = create_update_state(jax.random.PRNGKey(0), config, 5, 2)
    trunk = state.world_model_state.params["encoder"]["trunk"]
    assert sorted(trunk) == sorted(
        f"{layer}_{i}" for i in range(hidden) for layer in ("Dense", "LayerNorm")
    )
    assert trunk["Dense_0"]["kernel"].shape == (5, 32)


def _q_dropout(state, deterministic: bool, key):
    wm = state.world_model_state
    z = jax.random.normal(jax.random.PRNGKey(7), (64, 16))
    a = jnp.zeros((64, 2))
    rngs = None if deterministic else {"dropout": key}
    out, inter = wm.apply_fn(
        {"params": wm.params},
        z,
        a,
        None,
        deterministic,
        method="q_logits",
        rngs=rngs,
        capture_intermediates=lambda mdl, _: isinstance(mdl, (nn.Dense, nn.Dropout)),
        mutable=["intermediates"],
    )
    return out, inter["intermediates"]["q"]["members"]


@pytest.fixture(scope="module")
def dropout_state():
    return create_update_state(jax.random.PRNGKey(0), TINY.replace(dropout=0.5), 5, 2)


def test_dropout_only_in_the_first_layer_of_each_member_and_independent(dropout_state):
    _, inter = _q_dropout(dropout_state, False, jax.random.PRNGKey(0))
    trunk = inter["trunk"]
    assert [k for k in trunk if k.startswith("Dropout")] == ["Dropout_0"]
    assert "Dropout_0" not in inter.get("out", {})
    pre = trunk["Dense_0"]["__call__"][0]
    post = trunk["Dropout_0"]["__call__"][0]
    kept = post != 0
    np.testing.assert_allclose(post[kept], pre[kept] / 0.5, rtol=1e-6)
    assert 0.4 < float(kept.mean()) < 0.6
    for i in range(5):  # independent masks per member
        for j in range(i + 1, 5):
            assert np.mean(kept[i] == kept[j]) < 0.65
    _, again = _q_dropout(dropout_state, False, jax.random.PRNGKey(1))
    kept_again = again["trunk"]["Dropout_0"]["__call__"][0] != 0
    assert np.mean(kept == kept_again) < 0.65  # a new key, a new mask


def test_dropout_off_when_deterministic(dropout_state):
    """(The Q logits are 0 at initialisation; compare the hidden layers.)"""
    _, inter = _q_dropout(dropout_state, True, None)
    trunk = inter["trunk"]
    np.testing.assert_array_equal(
        trunk["Dropout_0"]["__call__"][0], trunk["Dense_0"]["__call__"][0]
    )
    _, again = _q_dropout(dropout_state, True, None)
    _, stochastic = _q_dropout(dropout_state, False, jax.random.PRNGKey(0))
    hidden = trunk["Dense_1"]["__call__"][0]
    np.testing.assert_array_equal(hidden, again["trunk"]["Dense_1"]["__call__"][0])
    assert not np.allclose(hidden, stochastic["trunk"]["Dense_1"]["__call__"][0])


def _zero_rows(params, path, rows):
    """A copy of ``params`` with ``rows`` of the input axis of the kernel at
    ``path`` set to 0 (stacked Q kernels included)."""
    params = jax.tree_util.tree_map(lambda x: x, params)
    node = params
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = node[path[-1]].at[..., rows, :].set(0.0)
    return params


def test_every_module_takes_its_inputs_in_the_reference_order():
    """``[obs, e]`` (encoder), ``[z, e, a]`` (dynamics, reward, Q) and
    ``[z, e]`` (policy) (``world_model.py:78-132``), so that multi-task (M8)
    can load reference-ordered weights: zeroing a block of first-kernel rows
    removes exactly that input. ``e`` broadcasts over the leading axes."""
    obs_dim, latent, task_dim, act = 5, 16, 3, 2
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    e = jax.random.normal(keys[0], (4, task_dim))
    obs = jax.random.normal(keys[1], (6, 4, obs_dim))
    z = jax.random.uniform(keys[2], (6, 4, latent))
    a1 = jax.random.uniform(keys[3], (6, 4, act), minval=-1, maxval=1)
    a2 = jax.random.uniform(keys[4], (6, 4, act), minval=-1, maxval=1)

    world_model = make_world_model(TINY)
    wm = world_model.init(keys[5], jnp.zeros((4, obs_dim)), jnp.zeros((4, act)), e)
    wm_params = jax.tree_util.tree_map(lambda x: x, wm["params"])
    for head in (wm_params["reward"]["out"], wm_params["q"]["members"]["out"]):
        head["kernel"] = jax.random.normal(keys[6], head["kernel"].shape)
    policy = make_policy_prior(TINY, act)
    pi_params = policy.init(keys[7], z[0], e)["params"]

    def apply_wm(method, params, *inputs):
        return world_model.apply({"params": params}, *inputs, method=method)

    first = ("trunk", "Dense_0", "kernel")
    cases = {
        # name: (output given (params, e, a), params, first-kernel path,
        #        width of the first input, number of action inputs)
        "encoder": (
            lambda p, emb, a: apply_wm("encode", p, obs, emb),
            wm_params,
            ("encoder", *first),
            obs_dim,
            0,
        ),
        "dynamics": (
            lambda p, emb, a: apply_wm("next", p, z, a, emb),
            wm_params,
            ("dynamics", *first),
            latent,
            1,
        ),
        "reward": (
            lambda p, emb, a: apply_wm("reward_logits", p, z, a, emb),
            wm_params,
            ("reward", *first),
            latent,
            1,
        ),
        "q": (
            lambda p, emb, a: apply_wm("q_logits", p, z, a, emb),
            wm_params,
            ("q", "members", *first),
            latent,
            1,
        ),
        "pi": (
            lambda p, emb, a: policy.apply({"params": p}, z, emb)[0],
            pi_params,
            first,
            latent,
            0,
        ),
    }
    for name, (apply, params, path, width, has_action) in cases.items():
        kernel = params
        for key in path:
            kernel = kernel[key]
        assert kernel.shape[-2] == width + task_dim + has_action * act, name
        assert not np.allclose(apply(params, e, a1), apply(params, 2 * e, a1)), name
        no_emb = _zero_rows(params, path, slice(width, width + task_dim))
        np.testing.assert_allclose(
            apply(no_emb, e, a1), apply(no_emb, 2 * e, a1), rtol=1e-6, err_msg=name
        )
        if has_action:
            assert not np.allclose(apply(params, e, a1), apply(params, e, a2)), name
            no_action = _zero_rows(params, path, slice(width + task_dim, None))
            np.testing.assert_allclose(
                apply(no_action, e, a1),
                apply(no_action, e, a2),
                rtol=1e-6,
                err_msg=name,
            )
