"""Behaviour of the DreamerV3 world model (dreamerv3_spec sections 1-2).

Each property is pinned on the tiny world model of ``common.py`` (its
parameters perturbed so that no layer is degenerate): the observe scan, the
``is_first`` resets, the action bound, the latent distribution (unimix,
straight-through samples, KL, free bits), the output losses, the replay
context alignment and the gradient routing of dreamerv3_spec 2.16.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DreamerV3.distributions import (
    SYMLOG_MSE_TOLERANCE,
    OneHot,
    bernoulli_loss,
    draw_onehot_noise,
    symlog_mse,
)
from ajax.agents.DreamerV3.networks import (
    RSSM,
    WorldModel,
    encode_action,
    features,
    initial_state,
    observe,
)
from ajax.agents.DreamerV3.world_model import (
    LOSS_TERMS,
    draw_posterior_noise,
    kl_losses,
    world_model_loss,
)
from ajax.distributional import symlog

from .common import TINY, B, T, replay_batch, tiny_model

S, C = TINY.stoch, TINY.classes


def _apply(model, params, *args, method):
    return model.apply({"params": params}, *args, method=method)


def _rssm(params, *args, method, config=TINY):
    """Apply an :class:`RSSM` method on the world model's ``rssm`` subtree."""
    return RSSM(config).apply({"params": params["rssm"]}, *args, method=method)


def _noise(seed: int) -> jax.Array:
    return draw_posterior_noise(jax.random.PRNGKey(seed), TINY, (B, T))


@functools.lru_cache(maxsize=None)
def _loss_fn(config=TINY):
    """The model of ``config`` and its jitted ``loss(params, batch, noise)``."""
    model = WorldModel(config, 5)
    return model, jax.jit(functools.partial(world_model_loss, model))


# --------------------------------------------------------------- the RSSM


def test_observe_scan_equals_step_by_step():
    model, params, _ = tiny_model()
    batch = replay_batch(0)
    noise = _noise(1)
    carry = initial_state(TINY, (B,))
    tokens = _apply(model, params, batch.obs[:, 1:], method=WorldModel.encode)
    actions, is_first = batch.action[:, :-1], batch.is_first[:, 1:]
    final, feats = jax.jit(functools.partial(observe, RSSM(TINY)))(
        params["rssm"], carry, tokens, actions, is_first, noise
    )
    step = jax.jit(lambda c, *x: _rssm(params, c, *x, method=RSSM.observe_step))
    for t in range(T):
        carry, feat = step(
            carry, tokens[:, t], actions[:, t], is_first[:, t], noise[:, t]
        )
        for ours, theirs in zip(feat, feats):
            np.testing.assert_allclose(ours, theirs[:, t], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(carry.deter, final.deter, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(carry.stoch, final.stoch)


@pytest.mark.parametrize("discrete", [False, True])
def test_is_first_resets_state_stoch_and_previous_action(discrete):
    """At ``is_first`` the core runs on zeros: ``h = core(0, 0, 0)``, whatever
    the carry and the previous action -- including a discrete action 0,
    whose one-hot ``[1, 0, 0]`` is masked after the encoding (spec 2.2)."""
    _, params, action_dim = tiny_model(discrete)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    carry = initial_state(TINY, (3,))._replace(
        deter=jax.random.normal(keys[0], (3, TINY.deter)),
        stoch=jax.nn.one_hot(jax.random.randint(keys[1], (3, S), 0, C), C),
    )
    if discrete:
        action = encode_action(jnp.zeros((3,), jnp.int32), 3)
        assert np.all(np.asarray(action[:, 0]) == 1.0)
    else:
        action = jax.random.normal(keys[2], (3, action_dim))
    token = jax.random.normal(keys[3], (3, TINY.units))
    noise = draw_onehot_noise(jax.random.PRNGKey(5), (3, S, C))
    zeros = initial_state(TINY, (3,))
    is_first = jnp.array([True, False, True])

    state, feat = _rssm(
        params, carry, token, action, is_first, noise, method=RSSM.observe_step
    )
    h0 = _rssm(
        params, zeros.deter, zeros.stoch, jnp.zeros_like(action), method=RSSM.core
    )
    h_carry = _rssm(params, carry.deter, carry.stoch, action, method=RSSM.core)
    np.testing.assert_allclose(state.deter[0], h0[0], atol=1e-6)
    np.testing.assert_allclose(state.deter[2], h0[2], atol=1e-6)
    np.testing.assert_allclose(state.deter[1], h_carry[1], atol=1e-6)
    assert np.max(np.abs(np.asarray(h_carry[0] - h0[0]))) > 1e-3
    # Masking the action matters: core(0, 0, one_hot(0)) is not core(0, 0, 0).
    h_unmasked = _rssm(params, zeros.deter, zeros.stoch, action, method=RSSM.core)
    assert np.max(np.abs(np.asarray(h_unmasked[0] - h0[0]))) > 1e-3
    # The posterior is then sampled from (h_t, e_t) as usual.
    post = _rssm(params, state.deter, token, method=RSSM.posterior_logits)
    np.testing.assert_allclose(feat.logits, post, atol=1e-6)
    np.testing.assert_array_equal(
        np.argmax(state.stoch, -1),
        np.argmax(OneHot.from_logits(post, TINY.unimix).logits + noise, -1),
    )


def test_core_bounds_the_action_input():
    """``a / sg(max(1, |a|))``: components beyond +-1 act as their sign, with
    gradient ``1 / |a|`` through the division (spec 2.3)."""
    _, params, _ = tiny_model()
    state = initial_state(TINY, (1,))

    def core(action):
        return _rssm(params, state.deter + 0.3, state.stoch, action, method=RSSM.core)

    big, clipped = jnp.array([[3.0, -0.5]]), jnp.array([[1.0, -0.5]])
    np.testing.assert_allclose(core(big), core(clipped), atol=1e-7)
    grad_big = jax.grad(lambda a: core(a).sum())(big)
    grad_clipped = jax.grad(lambda a: core(a).sum())(clipped)
    np.testing.assert_allclose(grad_big[0, 0], grad_clipped[0, 0] / 3.0, rtol=1e-5)
    np.testing.assert_allclose(grad_big[0, 1], grad_clipped[0, 1], rtol=1e-5)


def test_imagine_step_samples_the_prior_without_reset():
    _, params, _ = tiny_model()
    key = jax.random.PRNGKey(3)
    carry = initial_state(TINY, (4,))._replace(
        deter=jax.random.normal(key, (4, TINY.deter))
    )
    action = jax.random.normal(jax.random.PRNGKey(4), (4, 2))
    noise = draw_onehot_noise(jax.random.PRNGKey(5), (4, S, C))
    state, feat = _rssm(params, carry, action, noise, method=RSSM.imagine_step)
    deter = _rssm(params, carry.deter, carry.stoch, action, method=RSSM.core)
    prior = _rssm(params, deter, method=RSSM.prior_logits)
    np.testing.assert_allclose(state.deter, deter, atol=1e-6)
    np.testing.assert_allclose(feat.logits, prior, atol=1e-6)
    expected = OneHot.from_logits(prior, TINY.unimix).sample(noise)
    np.testing.assert_array_equal(state.stoch, expected)


def test_imagine_step_samples_the_mixed_prior():
    """Imagination samples ``0.99 softmax + 0.01 / C`` (Algorithm C, spec
    2.8): with a prior so peaked that the last class has raw probability
    ``~e^-40``, noise that lifts that class above the others under the
    mixed probabilities (floor ``0.01 / C``) but not under the raw ones
    selects it."""
    _, params, _ = tiny_model()
    peaked = jnp.tile(jnp.linspace(20.0, -20.0, C), S)  # last class least likely
    params = {
        **params,
        "rssm": {
            **params["rssm"],
            "priorlogit": {
                "kernel": jnp.zeros_like(params["rssm"]["priorlogit"]["kernel"]),
                "bias": peaked,
            },
        },
    }
    carry = initial_state(TINY, (1,))
    action = jnp.zeros((1, 2))
    raw = peaked.reshape(1, S, C)
    mixed = np.asarray(OneHot.from_logits(raw, TINY.unimix).logits)
    bonus = mixed.max(-1) - mixed[..., -1] + 0.5
    noise = jnp.zeros((1, S, C)).at[..., -1].set(bonus)
    unmixed = np.asarray(jax.nn.log_softmax(raw, -1) + noise)
    assert np.all(unmixed.argmax(-1) == 0)  # without unimix: the likeliest class
    state, feat = _rssm(params, carry, action, noise, method=RSSM.imagine_step)
    np.testing.assert_array_equal(feat.logits, raw)
    np.testing.assert_array_equal(np.argmax(state.stoch, -1), C - 1)


# ----------------------------------------------------- latent distribution


def test_unimix_floor():
    """``p = 0.99 softmax(l) + 0.01 / C``: every class keeps at least
    ``0.01 / C`` however peaked the logits (spec 1.8)."""
    logits = jnp.array([[[50.0, -50.0, -50.0, -50.0], [0.0, 0.0, 0.0, 0.0]]])
    probs = np.asarray(OneHot.from_logits(logits, 0.01).probs, np.float64)
    np.testing.assert_allclose(probs.sum(-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(probs[0, 0, 1:], 0.01 / 4, rtol=1e-5)
    np.testing.assert_allclose(probs[0, 0, 0], 0.99 + 0.01 / 4, rtol=1e-6)
    np.testing.assert_allclose(probs[0, 1], 0.25, rtol=1e-6)
    assert probs.min() >= 0.01 / 4 * (1 - 1e-5)
    plain = OneHot.from_logits(logits, 0.0).probs
    np.testing.assert_allclose(plain, jax.nn.softmax(logits, -1), atol=1e-7)


def test_straight_through_sample():
    """Forward value exactly one-hot at ``argmax(log p + noise)``; gradient
    equal to that of the **mixed** probabilities (spec 1.8)."""
    logits = jax.random.normal(jax.random.PRNGKey(0), (6, S, C))
    noise = draw_onehot_noise(jax.random.PRNGKey(1), (6, S, C))
    weights = jax.random.normal(jax.random.PRNGKey(2), (6, S, C))
    dist = OneHot.from_logits(logits, 0.01)
    sample = np.asarray(dist.sample(noise))
    assert set(np.unique(sample)) <= {0.0, 1.0}
    np.testing.assert_array_equal(sample.sum(-1), 1.0)
    np.testing.assert_array_equal(
        sample.argmax(-1), np.argmax(np.asarray(dist.logits + noise), -1)
    )
    st = jax.grad(
        lambda x: jnp.sum(weights * OneHot.from_logits(x, 0.01).sample(noise))
    )(logits)
    mixed = jax.grad(lambda x: jnp.sum(weights * OneHot.from_logits(x, 0.01).probs))(
        logits
    )
    unmixed = jax.grad(lambda x: jnp.sum(weights * jax.nn.softmax(x, -1)))(logits)
    np.testing.assert_allclose(st, mixed, atol=1e-7)
    np.testing.assert_allclose(st, 0.99 * unmixed, atol=1e-6)


def test_gumbel_max_samples_follow_the_mixed_probabilities():
    logits = jnp.array([[2.0, 0.0, -1.0, -30.0]])
    dist = OneHot.from_logits(logits, 0.01)
    noise = draw_onehot_noise(jax.random.PRNGKey(0), (40_000, 1, 4))
    frequencies = np.asarray(dist.sample(noise)).mean(0)
    np.testing.assert_allclose(frequencies, dist.probs, atol=0.01)
    assert frequencies[0, 3] > 0.0  # the unimix floor 0.0025 is sampled


def test_kl_is_posterior_to_prior_summed_over_latents():
    """``KL(p || q) = sum_S sum_C p (log p - log q)`` of the mixed
    distributions, in that direction (spec 1.8, 2.14)."""
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    post, prior = (3 * jax.random.normal(k, (5, S, C)) for k in keys)
    p = np.asarray(OneHot.from_logits(post, 0.01).probs, np.float64)
    q = np.asarray(OneHot.from_logits(prior, 0.01).probs, np.float64)
    expected = np.sum(p * (np.log(p) - np.log(q)), (-2, -1))
    reverse = np.sum(q * (np.log(q) - np.log(p)), (-2, -1))
    kl = OneHot.from_logits(post, 0.01).kl(OneHot.from_logits(prior, 0.01))
    np.testing.assert_allclose(kl, expected, rtol=1e-5)
    assert np.max(np.abs(expected - reverse)) > 0.1
    entropy = OneHot.from_logits(post, 0.01).entropy()
    np.testing.assert_allclose(entropy, -np.sum(p * np.log(p), (-2, -1)), rtol=1e-5)


def test_kl_losses_stop_gradients():
    """``dyn = KL(sg(post) || prior)`` trains only the prior logits, ``rep =
    KL(post || sg(prior))`` only the posterior logits; same forward value
    (spec 2.14)."""
    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    post, prior = (3 * jax.random.normal(k, (5, S, C)) for k in keys)
    dyn, rep, kl = kl_losses(post, prior, 0.01, 0.0)
    np.testing.assert_array_equal(dyn, rep)
    np.testing.assert_array_equal(dyn, kl)
    for index, term in ((0, "dyn"), (1, "rep")):
        grads = jax.grad(
            lambda p, q, i=index: kl_losses(p, q, 0.01, 0.0)[i].sum(), (0, 1)
        )(post, prior)
        trained = {"dyn": 1, "rep": 0}[term]
        assert not np.any(grads[1 - trained]), term
        assert np.all(np.abs(np.asarray(grads[trained])).sum(-1) > 0), term
    assert not np.any(jax.grad(lambda p: kl_losses(p, prior, 0.01, 0.0)[2].sum())(post))


def test_free_bits_clamp_the_kl_summed_over_latents_at_one_nat():
    """``max(sum_S KL_s, 1)``: no gradient below 1 nat in total, the KL's
    gradient above -- even when every single latent is below 1 nat, which a
    per-latent clamp would cut (spec 2.14; Table 4 free nats 1)."""
    prior = jnp.zeros((2, S, C))
    shift = jnp.zeros(C).at[-1].set(1.0)
    post = jnp.stack([0.6 * shift, 2.0 * shift])[:, None].repeat(S, 1)
    kl = np.asarray(OneHot.from_logits(post, 0.01).kl(OneHot.from_logits(prior, 0.01)))
    per_latent = kl / S
    assert kl[0] < 1.0 < kl[1] and per_latent[1] < 1.0
    for index, loss in enumerate(kl_losses(post, prior, 0.01, 1.0)[:2]):
        np.testing.assert_allclose(loss, np.maximum(kl, 1.0), rtol=1e-6)
        grads = jax.grad(
            lambda p, q, i=index: kl_losses(p, q, 0.01, 1.0)[i].sum(), (0, 1)
        )(post, prior)
        trained = np.asarray(grads[1 - index])  # dyn -> prior, rep -> post
        assert not np.any(trained[0]) and np.any(trained[1])


def test_zero_free_nats_skip_the_clamp():
    """``free_nats = 0`` applies no clamp at all (``if free:``,
    ``29eb964:dreamerv3/nets.py:102-104``): a KL that float32 rounds below 0
    keeps its value and its gradient instead of being cut at 0."""
    keys = jax.random.split(jax.random.PRNGKey(7), 2)
    post = jax.random.normal(keys[0], (256, S, C))
    prior = post + 1e-7 * jax.random.normal(keys[1], post.shape)
    kl = OneHot.from_logits(post, 0.01).kl(OneHot.from_logits(prior, 0.01))
    negative = np.asarray(kl) < 0
    assert negative.any()
    dyn, rep, _ = kl_losses(post, prior, 0.01, 0.0)
    np.testing.assert_array_equal(dyn, kl)
    np.testing.assert_array_equal(rep, kl)
    grad = jax.grad(lambda q: kl_losses(post, q, 0.01, 0.0)[0].sum())(prior)
    raw = jax.grad(
        lambda q: OneHot.from_logits(post, 0.01).kl(OneHot.from_logits(q, 0.01)).sum()
    )(prior)
    np.testing.assert_array_equal(grad, raw)
    assert np.all(np.abs(np.asarray(grad))[negative].sum((-2, -1)) > 0)


# ------------------------------------------------------------ output losses


def test_symlog_mse_tolerance_no_half_and_sum():
    """Squared symlog errors below 1e-8 count 0, there is no 1/2 factor and
    the errors are summed over the feature axis (2411f7d ``TransformedMseDist``
    ``tol=1e-8``; spec 1.11). A squared error of 4e-8 is kept, 2.5e-9 is
    dropped: a tolerance of 1e-7 or more would drop both."""
    assert SYMLOG_MSE_TOLERANCE == 1e-8
    target = jnp.array([[0.0, 0.0, -50.0, 1e4]])  # symlog(0) = 0 exactly
    exact = symlog(target)
    offsets = jnp.array([[5e-5, 2e-4, 1.0, -2.0]])  # squared: 2.5e-9 .. 4
    small = symlog_mse(exact[:, :2] + offsets[:, :2], target[:, :2])
    np.testing.assert_allclose(small, [4e-8], rtol=1e-5)
    loss = symlog_mse(exact + offsets, target)
    np.testing.assert_allclose(loss, [4e-8 + 1.0 + 4.0], rtol=1e-6)
    grads = jax.grad(lambda p: symlog_mse(p, target).sum())(exact + offsets)
    np.testing.assert_allclose(grads[0, 0], 0.0)
    np.testing.assert_allclose(grads[0, 1:], 2 * np.asarray(offsets)[0, 1:], rtol=1e-5)
    target_grad = jax.grad(lambda y: symlog_mse(exact + offsets, y).sum())(target)
    assert not np.any(target_grad)


def test_bernoulli_soft_labels():
    logits = jnp.array([-3.0, -0.5, 0.0, 2.0, 8.0])
    target = jnp.array([0.0, 0.997, 0.5, 0.997, 1.0])
    loss = bernoulli_loss(logits, target)
    x, c = np.asarray(logits, np.float64), np.asarray(target, np.float64)
    sig = 1 / (1 + np.exp(-x))
    expected = -(c * np.log(sig) + (1 - c) * np.log(1 - sig))
    np.testing.assert_allclose(loss, expected, rtol=1e-6)
    grads = jax.grad(lambda x: bernoulli_loss(x, target).sum())(logits)
    np.testing.assert_allclose(grads, sig - c, atol=1e-6)
    assert not np.any(jax.grad(lambda c: bernoulli_loss(logits, c).sum())(target))


# --------------------------------------------------------- world-model loss


@pytest.mark.parametrize("discrete", [False, True])
def test_world_model_loss_terms(discrete):
    """Shapes, the free-bits clamp after summing the KL over latents, the
    soft continue label (truncations keep 0.997) and the weighted sum. The
    parameters are perturbed lightly so that some steps have a KL below 1
    nat (clamped) and some above."""
    model, params, _ = tiny_model(discrete, scale=0.25)
    batch = replay_batch(1, discrete)
    _, loss_fn = _loss_fn()
    out = loss_fn(params, batch, _noise(2))
    for term in LOSS_TERMS:
        assert out.losses[term].shape == (B, T)
    assert out.entries.deter.shape == (B, T, TINY.deter)
    assert out.entries.stoch.shape == (B, T, S) and out.entries.stoch.dtype == jnp.int32
    assert out.posterior.stoch.shape == (B, T, S, C)
    assert out.tokens.shape == (B, T, TINY.units)
    kl = OneHot.from_logits(out.posterior.logits, TINY.unimix).kl(
        OneHot.from_logits(out.prior_logits, TINY.unimix)
    )
    np.testing.assert_allclose(out.kl, kl, rtol=1e-6)
    assert np.any(kl < 1.0) and np.any(kl > 1.0)
    for term in ("dyn", "rep"):
        np.testing.assert_allclose(out.losses[term], np.maximum(kl, 1.0), rtol=1e-6)
    feat = features(out.posterior.deter, out.posterior.stoch)
    cont_logit = _apply(model, params, feat, method=WorldModel.cont_logit)
    target = np.full((B, T), 1 - 1 / 333, np.float32)
    target[np.asarray(batch.is_terminal[:, 1:])] = 0.0
    assert np.any(np.asarray(batch.is_last[:, 1:] & ~batch.is_terminal[:, 1:]))
    np.testing.assert_allclose(
        out.losses["con"], bernoulli_loss(cont_logit, target), rtol=1e-6
    )
    scales = (1.0, 1.0, 1.0, 1.0, 0.1)
    weighted = sum(s * np.mean(out.losses[k]) for s, k in zip(scales, LOSS_TERMS))
    np.testing.assert_allclose(out.weighted, weighted, rtol=1e-6)


def test_free_bits_act_per_step_in_the_loss():
    """Inside the world-model loss the clamp is per ``(b, t)``: the steps
    below the threshold give no gradient, those above do (spec 2.14). The
    threshold is set to the median KL so that both kinds of step exist."""
    model, params, _ = tiny_model()
    batch, noise = replay_batch(2), _noise(3)
    kl = np.asarray(_loss_fn()[1](params, batch, noise).kl)
    free = float(np.median(kl))
    config = dataclasses.replace(TINY, free_nats=free)
    _, loss_fn = _loss_fn(config)
    low = np.unravel_index(np.argmin(kl), kl.shape)
    high = np.unravel_index(np.argmax(kl), kl.shape)
    assert kl[low] < free < kl[high]
    for term in ("dyn", "rep"):
        grad = jax.jit(
            jax.grad(lambda p, i, t=term: loss_fn(p, batch, noise).losses[t][i])
        )
        assert not any(np.any(g) for g in jax.tree.leaves(grad(params, low)))
        assert any(np.any(g) for g in jax.tree.leaves(grad(params, high)))


_GROUPS = {
    "enc": lambda path: path[0] == "enc",
    "core": lambda path: path[0] == "rssm" and path[1].startswith("dyn"),
    "post": lambda path: path[0] == "rssm" and path[1] in ("obs", "obslogit"),
    "prior": lambda path: path[0] == "rssm" and path[1] in ("prior", "priorlogit"),
    "dec": lambda path: path[0] == "dec",
    "rew": lambda path: path[0] == "rew",
    "con": lambda path: path[0] == "con",
}
#: Which parameter groups each term trains (dreamerv3_spec 2.14, 2.16). The
#: heads read the posterior without a stop-gradient (reward_grad), so they
#: train the encoder, the posterior and the core; dyn trains the prior and,
#: through the non-stop-gradiented h_t, the core and everything upstream;
#: rep never reaches the prior.
ROUTING = {
    "rec": {"enc", "core", "post", "dec"},
    "rew": {"enc", "core", "post", "rew"},
    "con": {"enc", "core", "post", "con"},
    "dyn": {"enc", "core", "post", "prior"},
    "rep": {"enc", "core", "post"},
}


def test_gradient_routing():
    model, params, _ = tiny_model()
    config = dataclasses.replace(TINY, free_nats=0.0)  # every KL carries gradient
    _, loss_fn = _loss_fn(config)
    batch, noise = replay_batch(3), _noise(4)

    @jax.jit
    def term_grads(params):
        return {
            term: jax.grad(lambda p, t=term: loss_fn(p, batch, noise).losses[t].mean())(
                params
            )
            for term in LOSS_TERMS
        }

    for term, grads in term_grads(params).items():
        assert _reached(grads) == ROUTING[term], term


def _reached(grads) -> set[str]:
    """The parameter groups of :data:`_GROUPS` with a non-zero gradient."""
    reached = set()
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads)[0]:
        keys = tuple(k.key for k in path)
        group = next(g for g, match in _GROUPS.items() if match(keys))
        if np.any(np.asarray(leaf)):
            reached.add(group)
    return reached


def test_posterior_output_carries_gradient():
    """``WorldModelOutput.posterior`` is not stop-gradiented: the replay
    critic of the actor-critic reads ``concat(h_t, z_t)`` of these steps and
    trains the encoder, the core and the posterior through them
    (``replay_critic_grad: True``, ``29eb964:dreamerv3/agent.py:331-333``;
    dreamerv3_spec 2.16)."""
    _, params, _ = tiny_model()
    _, loss_fn = _loss_fn()
    batch, noise = replay_batch(3), _noise(4)
    weights = jax.random.normal(jax.random.PRNGKey(9), (B, T, TINY.feat_dim))

    @jax.jit
    def critic_like(params):
        post = loss_fn(params, batch, noise).posterior
        return jnp.sum(weights * features(post.deter, post.stoch))

    assert _reached(jax.grad(critic_like)(params)) == {"enc", "core", "post"}


def test_replay_context_alignment():
    """Algorithm J: the carry is the latent stored with row 0, the loss sees
    observations 1..T and previous actions 0..T-1 (29eb964). Nothing else of
    the context row, and no last action, enters; an ``is_first`` at index 1
    (batch row 1) also discards the context carry and action."""
    model, params, _ = tiny_model()
    batch, noise = replay_batch(4), _noise(5)
    _, loss_fn = _loss_fn()
    base = loss_fn(params, batch, noise)

    def changed(**fields):
        out = loss_fn(params, batch._replace(**fields), noise)
        diffs = [np.abs(np.asarray(a) - np.asarray(b)).max(-1) for a, b in (
            (out.losses["rec"][..., None], base.losses["rec"][..., None]),
            (out.entries.deter, base.entries.deter),
        )]  # fmt: skip
        return np.max(np.stack(diffs), 0) > 0  # [B, T]: which steps changed

    unused = {
        "obs": batch.obs.at[:, 0].add(5.0),
        "reward": batch.reward.at[:, 0].add(5.0),
        "is_first": batch.is_first.at[:, 0].set(False),
        "is_terminal": batch.is_terminal.at[:, 0].set(True),
        "action": batch.action.at[:, -1].add(5.0),
    }
    for name, value in unused.items():
        assert not np.any(changed(**{name: value})), name
    assert bool(batch.is_first[1, 1]) and not bool(batch.is_first[0, 1])
    for name, value in {
        "context_deter": batch.context_deter + 1.0,
        "context_stoch": (batch.context_stoch + 1) % C,
        "action": batch.action.at[:, 0].add(0.7),
    }.items():
        steps = changed(**{name: value})
        assert steps[0, 0] and not np.any(steps[1]), name
    # The context carry is data: no gradient reaches it.
    grads = jax.grad(
        lambda d: loss_fn(params, batch._replace(context_deter=d), noise).weighted
    )(batch.context_deter)
    assert not np.any(grads)
