"""Behaviour of DreamerV3's actor-critic, optimizer and training step.

dreamerv3_spec sections 3-4 (Algorithms F-J), each property pinned on small
synthetic inputs or on the tiny learner of ``common.py`` with perturbed
parameters: the actor's distributions and heads, the lambda-return (with
fractional continuations, next-state indexing), the imagination losses (the
weight with the start continuation, the return normaliser updated then
read, the advantage without offset, REINFORCE on the stop-gradiented
action, the entropy coefficient), the replay critic (mask, discount,
bootstrap), the imagination rollout, the gradient routing of the joint
loss, LaProp with per-tensor AGC and warmup, the three optimizers fed one
joint gradient (equal to one optimizer; a sequential variant differs) and
the slow critic. Parity with the reference code itself is in
``test_dreamerv3_train_parity.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ajax.agents.DreamerV3 import learner
from ajax.agents.DreamerV3.actor_critic import (
    imagination_loss,
    imagine,
    lambda_return,
    replay_critic_loss,
)
from ajax.agents.DreamerV3.distributions import (
    BoundedNormal,
    OneHotPolicy,
    Policy,
    draw_onehot_noise,
)
from ajax.agents.DreamerV3.networks import (
    RSSM,
    Actor,
    RSSMState,
    WorldModel,
    features,
    init_actor,
    init_critic,
    make_critic,
)
from ajax.agents.DreamerV3.optim import laprop, make_optimizer, scale_by_agc
from ajax.agents.DreamerV3.state import DreamerV3Config
from ajax.agents.DreamerV3.world_model import world_model_loss
from ajax.distributional import TwoHot
from ajax.normalizers import ReturnNormalizer

from .common import (
    ACTION_DIM,
    NUM_ACTIONS,
    OBS_DIM,
    TINY,
    B,
    T,
    perturbed,
    replay_batch,
    tiny_model,
)

#: The tiny learner, with a short imagination (the horizon only sets the
#: scan length; parity at the default 15 is in the parity tests) whose
#: length 10 is distinct from every other size (``H + 1 = 11`` too), and the
#: two lambdas at distinct non-default values, so that the tests can tell
#: which one a return used (the expected values read them from here).
CONFIG = dataclasses.replace(TINY, imag_horizon=10, lam=0.7, repval_lam=0.85)
H = CONFIG.imag_horizon
TWOHOT = TwoHot.dreamerv3(CONFIG.bins)
EPS = float(np.finfo(np.float32).eps)


def _key(seed: int) -> jax.Array:
    return jax.random.PRNGKey(seed)


def _concentrated(out: dict) -> dict:
    """A two-hot output layer whose bias favours the middle bins, so that
    its predictions are of order 1 (as a trained head's)."""
    j = np.arange(out["bias"].shape[-1])
    bias = -0.5 * np.abs(j - j.size // 2)
    return {**out, "bias": jnp.asarray(bias, jnp.float32)}


@functools.lru_cache(maxsize=None)
def tiny_learner(
    discrete: bool = False, config: DreamerV3Config = CONFIG
) -> learner.LearnerState:
    """The tiny learner with perturbed parameters (no degenerate layer) and
    fresh optimizers; the slow critic differs from the critic."""
    _, wm_params, action_dim = tiny_model(discrete, config=config)
    state = learner.init_learner(_key(0), config, OBS_DIM, action_dim, discrete)
    actor = perturbed(state.actor_state.params, 11)
    critic = perturbed(state.critic_state.params, 12)
    slow = perturbed(state.critic_state.params, 13)
    wm_params["rew"]["out"] = _concentrated(wm_params["rew"]["out"])
    critic["out"] = _concentrated(critic["out"])
    slow["out"] = _concentrated(slow["out"])

    def fresh(train_state, params, **extra):
        opt_state = train_state.tx.init(params)
        return train_state.replace(params=params, opt_state=opt_state, **extra)

    return state.replace(
        world_model_state=fresh(state.world_model_state, wm_params),
        actor_state=fresh(state.actor_state, actor),
        critic_state=fresh(state.critic_state, critic, target_params=slow),
    )


def _noise(seed: int, discrete: bool = False, config: DreamerV3Config = CONFIG):
    action_dim = NUM_ACTIONS if discrete else ACTION_DIM
    return learner.draw_train_noise(_key(seed), config, B, T, action_dim, discrete)


# ------------------------------------------------------ distributions, heads


def test_bounded_normal():
    """``Normal(tanh(m), 0.9 sigmoid(s + 2) + 0.1)`` per dimension, not
    squashed: samples leave ``[-1, 1]`` and the log-probability is the plain
    normal one, summed over dimensions (dreamerv3_spec 3.13)."""
    m = jnp.array([[0.0, 0.0], [2.0, -1.0]])
    s = jnp.array([[0.0, 0.0], [-5.0, 3.0]])
    dist = BoundedNormal.from_outputs(m, s, minstd=0.1, maxstd=1.0)
    np.testing.assert_allclose(dist.mean, np.tanh(m), rtol=1e-6)
    sigmoid = 1 / (1 + np.exp(-(np.asarray(s, np.float64) + 2)))
    np.testing.assert_allclose(dist.std, 0.9 * sigmoid + 0.1, rtol=1e-6)
    np.testing.assert_allclose(dist.std[0], 0.8927, atol=1e-4)  # at init
    noise = jnp.array([[3.0, -4.0], [0.5, 2.0]])
    sample = dist.sample(noise)
    np.testing.assert_allclose(sample, noise * dist.std + dist.mean, rtol=1e-6)
    assert np.max(np.abs(np.asarray(sample))) > 1.0
    mean, std = np.asarray(dist.mean, np.float64), np.asarray(dist.std, np.float64)
    x = np.asarray(sample, np.float64)
    logp = -0.5 * ((x - mean) / std) ** 2 - np.log(std) - 0.5 * math.log(2 * math.pi)
    np.testing.assert_allclose(dist.log_prob(sample), logp.sum(-1), rtol=1e-5)
    entropy = 0.5 + 0.5 * math.log(2 * math.pi) + np.log(std)
    np.testing.assert_allclose(dist.entropy(), entropy.sum(-1), rtol=1e-6)
    bounds = BoundedNormal.entropy_range(2, 0.1, 1.0)
    per_dim = 0.5 + 0.5 * math.log(2 * math.pi) + np.log([0.1, 1.0])
    np.testing.assert_allclose(bounds, 2 * per_dim, rtol=1e-12)


def test_one_hot_policy_uses_the_mixed_distribution():
    """2411f7d's discrete actor: probabilities ``0.99 softmax + 0.01 / A``;
    sampling, log-probability and entropy all use them (dreamerv3_spec
    3.14); the sample is a straight-through one-hot."""
    raw = jnp.array([[4.0, -2.0, 0.5], [0.0, 0.0, 0.0]])
    dist = OneHotPolicy.from_logits(raw, 0.01)
    mixed = 0.99 * np.asarray(jax.nn.softmax(raw)) + 0.01 / 3
    np.testing.assert_allclose(np.exp(dist.logits), mixed, rtol=1e-6)
    for k in range(3):
        one_hot = jax.nn.one_hot(jnp.array([k, k]), 3)
        log_prob = np.log(mixed[:, k])
        np.testing.assert_allclose(dist.log_prob(one_hot), log_prob, atol=1e-6)
    entropy = -(mixed * np.log(mixed)).sum(-1)
    np.testing.assert_allclose(dist.entropy(), entropy, atol=1e-6)
    assert OneHotPolicy.entropy_range(3) == (0.0, math.log(3))
    noise = draw_onehot_noise(_key(0), (2, 3))
    sample = dist.sample(noise)
    np.testing.assert_array_equal(
        sample, jax.nn.one_hot(jnp.argmax(dist.logits + noise, -1), 3)
    )
    weights = jnp.array([1.0, -2.0, 3.0])

    def through_sample(raw):
        return jnp.sum(OneHotPolicy.from_logits(raw, 0.01).sample(noise) * weights)

    def through_probs(raw):
        return jnp.sum(jnp.exp(OneHotPolicy.from_logits(raw, 0.01).logits) * weights)

    np.testing.assert_allclose(
        jax.grad(through_sample)(raw),
        jax.grad(through_probs)(raw),
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize("discrete", [False, True])
def test_actor_and_critic_heads(discrete):
    """3 hidden layers ``Dense -> RMSNorm -> SiLU`` each; the actor's output
    layers (separate ``mean`` and ``std``, or ``logits``) at outscale 0.01,
    so it starts near ``mean 0, std 0.89`` (or uniform); the critic's output
    is zero, so it predicts exactly 0 (dreamerv3_spec 1.5, 3.13-3.16)."""
    config = DreamerV3Config.from_model_size("12m")
    action_dim = 6
    actor = init_actor(_key(0), config, action_dim, discrete)
    critic = init_critic(_key(1), config)
    outputs = ("logits",) if discrete else ("mean", "std")
    assert set(actor) == {"mlp", *outputs}
    assert set(critic) == {"mlp", "out"}
    for mlp in (actor["mlp"], critic["mlp"]):
        assert set(mlp) == {f"{n}_{i}" for n in ("Dense", "RMSNorm") for i in range(3)}
    for name in outputs:
        kernel = np.asarray(actor[name]["kernel"])
        assert kernel.shape == (config.units, action_dim)
        np.testing.assert_allclose(
            np.std(kernel), 0.01 / math.sqrt(config.units), rtol=0.1
        )
    assert critic["out"]["kernel"].shape == (config.units, config.bins)
    assert not np.any(critic["out"]["kernel"]) and not np.any(critic["out"]["bias"])
    feat = 3 * jax.random.normal(_key(2), (64, config.feat_dim))
    value = TWOHOT.decode(make_critic(config).apply({"params": critic}, feat))
    assert np.all(np.asarray(value) == 0.0)
    policy = Actor(config, action_dim, discrete).apply({"params": actor}, feat)
    if discrete:
        np.testing.assert_allclose(np.exp(policy.logits), 1 / action_dim, rtol=0.05)
    else:
        assert np.max(np.abs(np.asarray(policy.mean))) < 0.1
        np.testing.assert_allclose(policy.std, 0.8927, atol=0.02)


def test_discrete_actor_mixes_with_actor_unimix():
    """The discrete actor mixes ``actor_unimix`` of the uniform distribution
    in (2411f7d ``actor.unimix``), not the latents' ``unimix``: with logits
    favouring one action, every other action keeps ``actor_unimix / A``."""
    config = dataclasses.replace(CONFIG, actor_unimix=0.3, unimix=0.01)
    params = init_actor(_key(0), config, NUM_ACTIONS, True)
    params["logits"]["bias"] = jnp.zeros(NUM_ACTIONS).at[0].set(50.0)
    feat = jnp.zeros((1, config.feat_dim))
    policy = Actor(config, NUM_ACTIONS, True).apply({"params": params}, feat)
    probs = np.exp(np.asarray(policy.logits[0], np.float64))
    np.testing.assert_allclose(probs[1:], 0.3 / NUM_ACTIONS, rtol=1e-5)


# -------------------------------------------------------------- the returns


def _hand_lambda_return(reward, live, cont, boot):
    """Algorithm H by the definition: ``R_{L-1} = boot_{L-1}``, ``R_t =
    r_{t+1} + live_{t+1} ((1 - cont_{t+1}) boot_{t+1} + cont_{t+1} R_{t+1})``
    in float64."""
    reward, live, cont, boot = (
        np.asarray(x, np.float64) for x in (reward, live, cont, boot)
    )
    length = reward.shape[1]
    ret = np.zeros((reward.shape[0], length))
    ret[:, -1] = boot[:, -1]
    for t in reversed(range(length - 1)):
        bootstrap = (1 - cont[:, t + 1]) * boot[:, t + 1] + cont[:, t + 1] * ret[
            :, t + 1
        ]
        ret[:, t] = reward[:, t + 1] + live[:, t + 1] * bootstrap
    return ret[:, :-1]


def test_lambda_return_matches_a_hand_recursion():
    """With fractional continuations and episode ends (``cont = 0``), and
    with the scalar lambda of imagination."""
    rng = np.random.default_rng(0)
    n, length = 5, 9
    reward = rng.normal(0, 2, (n, length)).astype(np.float32)
    live = rng.uniform(0.2, 1.0, (n, length)).astype(np.float32)
    cont = (0.95 * (rng.uniform(size=(n, length)) > 0.2)).astype(np.float32)
    boot = rng.normal(0, 5, (n, length)).astype(np.float32)
    ret = jax.jit(lambda_return)(reward, live, cont, boot)
    assert ret.shape == (n, length - 1) and ret.dtype == jnp.float32
    np.testing.assert_allclose(
        ret, _hand_lambda_return(reward, live, cont, boot), rtol=1e-5, atol=1e-5
    )
    scalar = lambda_return(reward, live, 0.95, boot)
    expected = _hand_lambda_return(reward, live, np.full_like(cont, 0.95), boot)
    np.testing.assert_allclose(scalar, expected, rtol=1e-5, atol=1e-5)
    # lambda = 0: one-step targets r_{t+1} + live_{t+1} boot_{t+1}.
    one_step = lambda_return(reward, live, 0.0, boot)
    np.testing.assert_allclose(
        one_step, reward[:, 1:] + live[:, 1:] * boot[:, 1:], rtol=1e-6
    )


def test_lambda_return_reads_the_next_state():
    """``R_t`` uses the reward, continuation and value of state ``t + 1``
    (dreamerv3_spec 3.7): index 0 of the inputs is never read, and changing
    step ``k`` changes ``R_0 .. R_{k-1}`` only."""
    rng = np.random.default_rng(1)
    inputs = [rng.uniform(0.1, 1.0, (3, 6)).astype(np.float32) for _ in range(4)]
    base = np.asarray(lambda_return(*inputs))
    for i in range(4):
        changed = [x.copy() for x in inputs]
        changed[i][:, 0] += 7.0
        np.testing.assert_array_equal(lambda_return(*changed), base)
        changed = [x.copy() for x in inputs]
        changed[i][:, 3] += 7.0
        moved = np.any(np.asarray(lambda_return(*changed)) != base, 0)
        assert moved.tolist() == [True, True, True, False, False], i


# --------------------------------------------------------- imagination loss


def _imagination_inputs(seed: int, n: int = 6, discrete: bool = False):
    """Heads on ``n`` trajectories of ``H + 1`` states: a policy from
    parameters ``theta`` (continuous with 2 dimensions, or 3 discrete
    actions) and the noise of its samples, two-hot critic logits favouring
    the middle bins, the slow critic's values, rewards and continuations
    (index 0 a hard data flag, the others fractional)."""
    rng = np.random.default_rng(seed)
    if discrete:
        theta = {"logits": jnp.asarray(rng.normal(0, 1, (n, H + 1, 3)), jnp.float32)}
        noise = draw_onehot_noise(_key(seed), (n, H + 1, 3))
    else:
        theta = {
            "m": jnp.asarray(rng.normal(0, 1, (n, H + 1, 2)), jnp.float32),
            "s": jnp.asarray(rng.normal(0, 1, (n, H + 1, 2)), jnp.float32),
        }
        noise = jnp.asarray(rng.normal(0, 1, (n, H + 1, 2)), jnp.float32)
    j = np.arange(CONFIG.bins)
    logits = -0.5 * np.abs(j - j.size // 2) + rng.normal(0, 1, (n, H + 1, j.size))
    cont = rng.uniform(0.5, 1.0, (n, H + 1))
    cont[:, 0] = 1.0
    cont[0, 0] = 0.0  # a terminal start state
    return {
        "theta": theta,
        "noise": noise,
        "value_logits": jnp.asarray(logits, jnp.float32),
        "slow_value": jnp.asarray(rng.normal(0, 1, (n, H + 1)), jnp.float32),
        "reward": jnp.asarray(rng.normal(0, 1, (n, H + 1)), jnp.float32),
        "cont": jnp.asarray(cont, jnp.float32),
    }


def _policy(theta) -> Policy:
    if "logits" in theta:
        return OneHotPolicy.from_logits(theta["logits"], CONFIG.actor_unimix)
    return BoundedNormal.from_outputs(
        theta["m"], theta["s"], CONFIG.minstd, CONFIG.maxstd
    )


_imagination_loss = jax.jit(
    imagination_loss, static_argnums=0, static_argnames="update_retnorm"
)


def _imagination(inputs, retnorm=None, config=CONFIG, update=True, action=None):
    """Algorithm F on :func:`_imagination_inputs` (jitted)."""
    policy = _policy(inputs["theta"])
    if action is None:
        action = policy.sample(inputs["noise"])
    retnorm = retnorm or ReturnNormalizer.create()
    return _imagination_loss(
        config,
        policy,
        action,
        inputs["value_logits"],
        inputs["slow_value"],
        inputs["reward"],
        inputs["cont"],
        retnorm,
        update_retnorm=update,
    )


def test_weight_is_the_cumulative_continuation_including_the_start():
    """``w_t = prod_{i <= t} cont_i`` with the start state's continuation
    (2411f7d: the data's ``1 - is_terminal``; dreamerv3_spec 3.5-3.6): a
    terminal start state gets no actor or critic loss at all."""
    inputs = _imagination_inputs(0)
    out = _imagination(inputs)
    np.testing.assert_allclose(out.weight, np.cumprod(inputs["cont"], 1), rtol=1e-6)
    assert not np.any(out.weight[0])
    assert not np.any(out.actor[0]) and not np.any(out.critic[0])
    assert np.all(np.asarray(out.actor[1:]) != 0)
    assert out.actor.shape == out.critic.shape == (6, H)


def test_return_normaliser_is_updated_then_read_and_the_advantage_not_offset():
    """The normaliser folds this step's 5th and 95th return percentiles in
    *before* its scale is read (dreamerv3_spec 3.8), and the advantage is
    ``(R - v) / max(1, hi - lo)`` without subtracting ``lo`` (3.10)."""
    inputs = _imagination_inputs(1)
    old = ReturnNormalizer.create().replace(lo=jnp.float32(-3.0), hi=jnp.float32(5.0))
    out = _imagination(inputs, old)
    value = TWOHOT.decode(inputs["value_logits"])
    # Jitted (inside the imagination) vs eager decode: equal up to float32
    # rounding, which is absolute near 0 (CI on Linux x86 differs by one ulp
    # of the expectation), hence an absolute floor of a few float32 eps.
    np.testing.assert_allclose(out.value, value, rtol=1e-6, atol=1e-7)
    ret = np.asarray(out.ret, np.float64)
    np.testing.assert_allclose(
        ret,
        _hand_lambda_return(
            inputs["reward"],
            inputs["cont"],
            np.full(ret.shape[:1] + (H + 1,), CONFIG.lam),
            value,
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    lo = 0.99 * -3.0 + 0.01 * np.percentile(ret, 5)
    hi = 0.99 * 5.0 + 0.01 * np.percentile(ret, 95)
    np.testing.assert_allclose((out.retnorm.lo, out.retnorm.hi), (lo, hi), rtol=1e-6)
    assert out.scale == pytest.approx(hi - lo, rel=1e-6)
    np.testing.assert_allclose(
        out.adv, (ret - value[:, :-1]) / (hi - lo), rtol=1e-5, atol=1e-6
    )
    # Not updated (the reference's report pass): the old scale, 8.
    frozen = _imagination(inputs, old, update=False)
    assert (frozen.retnorm.lo, frozen.retnorm.hi) == (old.lo, old.hi)
    np.testing.assert_allclose(
        frozen.adv, (ret - value[:, :-1]) / 8.0, rtol=1e-5, atol=1e-6
    )
    # The scale is floored at 1 (small returns are not amplified).
    assert _imagination(inputs).scale == 1.0


@pytest.mark.parametrize("discrete", [False, True])
def test_actor_loss_is_reinforce_on_the_stop_gradiented_action(discrete):
    """``actor = sg(w) * -(log pi(sg(a)) sg(adv) + actent H[pi])`` for both
    action types (dreamerv3_spec 3.11): the gradient is the same whether the
    action is computed from the parameters (a reparameterised normal sample,
    or a straight-through one-hot, both carrying gradient) or given as a
    constant, so no gradient flows through the sample; and the entropy
    enters with the coefficient ``actent``."""
    inputs = _imagination_inputs(2, discrete=discrete)
    out = _imagination(inputs)
    policy = _policy(inputs["theta"])
    action = policy.sample(inputs["noise"])
    log_pi = policy.log_prob(action)[:, :-1]
    entropy = policy.entropy()[:, :-1]
    expected = out.weight[:, :-1] * -(log_pi * out.adv + CONFIG.actent * entropy)
    np.testing.assert_allclose(out.actor, expected, rtol=1e-5, atol=1e-7)

    def loss(theta, constant_action):
        given = action if constant_action else None
        return _imagination({**inputs, "theta": theta}, action=given).actor.mean()

    through_sample = jax.grad(loss)(inputs["theta"], False)
    constant = jax.grad(loss)(inputs["theta"], True)
    for key in through_sample:
        np.testing.assert_allclose(
            through_sample[key], constant[key], rtol=1e-6, atol=1e-9
        )
    # The entropy coefficient.
    big = dataclasses.replace(CONFIG, actent=0.25)
    difference = _imagination(inputs, config=big).actor - out.actor
    np.testing.assert_allclose(
        difference,
        -out.weight[:, :-1] * (0.25 - CONFIG.actent) * entropy,
        rtol=1e-4,
        atol=1e-6,
    )


def test_critic_loss_targets_the_return_and_the_slow_critic():
    """``critic = sg(w) (CE(sg(R)) + slowreg CE(sg(v_slow)))`` on states
    ``0..H-1`` (dreamerv3_spec 3.17): two-hot cross-entropies toward the
    return and toward the slow critic's *prediction*; only the critic's
    logits receive gradient."""
    inputs = _imagination_inputs(3)
    out = _imagination(inputs)
    logits = inputs["value_logits"][:, :-1]
    expected = out.weight[:, :-1] * (
        TWOHOT.loss(logits, out.ret) + TWOHOT.loss(logits, inputs["slow_value"][:, :-1])
    )
    np.testing.assert_allclose(out.critic, expected, rtol=1e-5)
    no_reg = _imagination(inputs, config=dataclasses.replace(CONFIG, slowreg=0.0))
    np.testing.assert_allclose(
        no_reg.critic, out.weight[:, :-1] * TWOHOT.loss(logits, out.ret), rtol=1e-5
    )

    def critic_loss(slow_value, reward, cont, value_logits):
        changed = {
            **inputs,
            "slow_value": slow_value,
            "reward": reward,
            "cont": cont,
            "value_logits": value_logits,
        }
        return _imagination(changed).critic.mean()

    grads = jax.grad(critic_loss, argnums=(0, 1, 2, 3))(
        inputs["slow_value"], inputs["reward"], inputs["cont"], inputs["value_logits"]
    )
    assert not np.any(grads[0]) and not np.any(grads[1]) and not np.any(grads[2])
    assert np.any(grads[3][:, :-1]) and not np.any(grads[3][:, -1])


def test_replay_critic_mask_discount_and_bootstrap():
    """The replay critic (dreamerv3_spec 3.19): replayed rewards, the fixed
    discount ``1 - 1 / 333`` with hard ``is_terminal``, its own lambda
    ``repval_lam`` (not the imagination's) cut at ``is_last``, bootstrapped
    from the imagination returns; loss masked by ``1 - is_last``, the last
    step without loss."""
    rng = np.random.default_rng(4)
    b, t = 2, 7
    reward = rng.normal(0, 1, (b, t)).astype(np.float32)
    boot = rng.normal(0, 3, (b, t)).astype(np.float32)
    is_terminal = np.zeros((b, t), bool)
    is_terminal[0, 3] = True
    is_last = is_terminal.copy()
    is_last[1, 4] = True  # a time-limit truncation
    j = np.arange(CONFIG.bins)
    logits = jnp.asarray(
        -0.5 * np.abs(j - j.size // 2) + rng.normal(0, 1, (b, t, j.size)), jnp.float32
    )
    slow = jnp.asarray(rng.normal(0, 1, (b, t)), jnp.float32)
    loss, ret = replay_critic_loss(
        CONFIG,
        logits,
        slow,
        jnp.asarray(boot),
        jnp.asarray(reward),
        jnp.asarray(is_terminal),
        jnp.asarray(is_last),
    )
    gamma = 1 - 1 / 333
    live = gamma * (1 - is_terminal)
    cont = CONFIG.repval_lam * (1 - is_last)
    np.testing.assert_allclose(
        ret, _hand_lambda_return(reward, live, cont, boot), rtol=1e-5, atol=1e-5
    )
    # Into a terminal state: its reward only; into a truncation: no lambda.
    np.testing.assert_allclose(ret[0, 2], reward[0, 3], rtol=1e-6)
    np.testing.assert_allclose(ret[1, 3], reward[1, 4] + gamma * boot[1, 4], rtol=1e-5)
    assert loss.shape == (b, t - 1)
    mask = 1 - is_last[:, :-1]
    head = logits[:, :-1]
    expected = mask * (TWOHOT.loss(head, ret) + TWOHOT.loss(head, slow[:, :-1]))
    np.testing.assert_allclose(loss, expected, rtol=1e-5)
    assert loss[0, 3] == 0.0 and loss[1, 4] == 0.0


# ------------------------------------------------------------- imagination


def test_imagine_rolls_out_the_rssm_with_the_policy():
    """Each step is the RSSM's imagine step on the previous action, then an
    action sampled from ``pi(concat(h, sg(z)))`` (Algorithm C,
    ``agent.py:252-257``); the start is stop-gradiented and the sampled
    latents carry no gradient (the deterministic states and actions do, to
    the RSSM's parameters)."""
    state = tiny_learner()
    rssm, rssm_params = RSSM(CONFIG), state.world_model_state.params["rssm"]

    def policy(feat):
        return state.actor_state.apply_fn({"params": state.actor_state.params}, feat)

    n = 3
    keys = jax.random.split(_key(5), 5)
    classes = jax.random.randint(keys[1], (n, CONFIG.stoch), 0, CONFIG.classes)
    start = RSSMState(
        jax.random.normal(keys[0], (n, CONFIG.deter)),
        jax.nn.one_hot(classes, CONFIG.classes),
    )
    start_action = jax.random.normal(keys[2], (n, ACTION_DIM))
    prior_noise = draw_onehot_noise(keys[3], (n, H, CONFIG.stoch, CONFIG.classes))
    action_noise = jax.random.normal(keys[4], (n, H, ACTION_DIM))

    def rollout(start, rssm_params):
        return imagine(
            rssm, rssm_params, policy, start, start_action, prior_noise, action_noise
        )

    @jax.jit
    def step(carry, action, prior_noise, action_noise):
        carry, out = rssm.apply(
            {"params": rssm_params},
            carry,
            action,
            prior_noise,
            method=RSSM.imagine_step,
        )
        return carry, out, policy(features(out.deter, out.stoch)).sample(action_noise)

    traj = jax.jit(rollout)(start, rssm_params)
    carry, action = start, start_action
    for i in range(H):
        carry, out, action = step(carry, action, prior_noise[:, i], action_noise[:, i])
        np.testing.assert_allclose(traj.deter[:, i], out.deter, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(traj.stoch[:, i], out.stoch)
        np.testing.assert_allclose(traj.action[:, i], action, rtol=1e-5, atol=1e-6)

    @jax.jit
    def grads(start, rssm_params):
        def total(start, rssm_params, which):
            return jnp.sum(getattr(rollout(start, rssm_params), which))

        return {
            which: jax.grad(total, (0, 1))(start, rssm_params, which)
            for which in ("deter", "stoch", "action")
        }

    reached = {
        which: [any(np.any(g) for g in jax.tree.leaves(arg)) for arg in grad]
        for which, grad in grads(start, rssm_params).items()
    }
    # [start, rssm parameters]
    assert reached == {
        "deter": [False, True],
        "stoch": [False, False],
        "action": [False, True],
    }


# ------------------------------------------------------------ the learner


_GROUPS = {
    "enc": lambda path: path[:2] == ("world_model", "enc"),
    "core": lambda path: path[1] == "rssm" and path[2].startswith("dyn"),
    "post": lambda path: path[1] == "rssm" and path[2] in ("obs", "obslogit"),
    "prior": lambda path: path[1] == "rssm" and path[2] in ("prior", "priorlogit"),
    "dec": lambda path: path[:2] == ("world_model", "dec"),
    "rew": lambda path: path[:2] == ("world_model", "rew"),
    "con": lambda path: path[:2] == ("world_model", "con"),
    "actor": lambda path: path[0] == "actor",
    "critic": lambda path: path[0] == "critic",
}
#: Which parameter groups each term trains (dreamerv3_spec 2.16, 3.4 and
#: the routing matrix of section 4): the world-model terms only the world
#: model; the actor and the imagination critic only themselves (every input
#: stop-gradiented, ``ac_grads: none``); the replay critic the critic and,
#: through the replayed posterior features, the encoder, posterior and core.
ROUTING = {
    "rec": {"enc", "core", "post", "dec"},
    "rew": {"enc", "core", "post", "rew"},
    "con": {"enc", "core", "post", "con"},
    "dyn": {"enc", "core", "post", "prior"},
    "rep": {"enc", "core", "post"},
    "actor": {"actor"},
    "critic": {"critic"},
    "repval": {"enc", "core", "post", "critic"},
}


def _reached(grads: learner.Params) -> set[str]:
    reached = set()
    for path, leaf in jax.tree_util.tree_flatten_with_path(grads._asdict())[0]:
        keys = tuple(k.key for k in path)
        group = next(g for g, match in _GROUPS.items() if match(keys))
        if np.any(np.asarray(leaf)):
            reached.add(group)
    return reached


@pytest.mark.parametrize("discrete", [False, True])
def test_gradient_routing(discrete):
    """The gradient of each term of the joint loss, with respect to the
    world model, actor and critic together: the actor loss reaches only the
    actor, the imagination critic loss only the critic, the replay critic
    the critic and the world model, the world-model terms never the actor
    or the critic."""
    config = dataclasses.replace(CONFIG, free_nats=0.0)  # every KL has gradient
    state = tiny_learner(discrete, config)
    batch, noise = replay_batch(5, discrete, config), _noise(6, discrete, config)

    @jax.jit
    def term_grads(params):
        def means(params):
            _, aux = learner.compute_loss(params, state, batch, noise, config=config)
            return jnp.stack([aux.losses[k].mean() for k in learner.ALL_TERMS])

        return jax.jacrev(means)(params)

    jacobian = term_grads(learner.learner_params(state))
    for i, term in enumerate(learner.ALL_TERMS):
        grads = jax.tree.map(lambda g, i=i: g[i], jacobian)
        assert _reached(grads) == ROUTING[term], term


@functools.lru_cache(maxsize=None)
def _jitted(config: DreamerV3Config):
    return jax.jit(functools.partial(learner.train_step, config=config))


#: Full learning rate from the first update, and a large one, so that one
#: update visibly moves every parameter; a slow-critic rate other than its
#: default 0.02.
FAST = dataclasses.replace(CONFIG, warmup=0, learning_rate=1e-3, slow_rate=0.25)


def _difference(a, b) -> float:
    """Largest difference of two trees, relative to each tensor's scale."""
    return max(
        float(np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-30))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def _moments(state: learner.LearnerState) -> tuple[learner.Params, learner.Params]:
    """The RMS and momentum moments of the three optimizers."""
    train_states = (state.world_model_state, state.actor_state, state.critic_state)
    nu = learner.Params(*(ts.opt_state[1].nu for ts in train_states))
    mu = learner.Params(*(ts.opt_state[2].ema for ts in train_states))
    return nu, mu


def test_one_joint_gradient_for_three_optimizers_is_one_optimizer():
    """Three LaProp instances (world model, actor, critic) fed one joint
    gradient give the single optimizer of the reference, which updates all
    modules together (dreamerv3_spec 4.3; DESIGN.md section 6.2): same
    parameters and moments over two updates, to float32 rounding (the two
    programs are compiled separately). Updating the world model first and
    taking the actor-critic gradient at the updated world model is not the
    reference: the world model's update is the same, the actor's and
    critic's differ."""
    state = tiny_learner(False, FAST)
    batches = [(replay_batch(7 + i), _noise(8 + i, config=FAST)) for i in range(2)]
    single_tx = make_optimizer(FAST)

    @jax.jit
    def single_step(state, opt_state, batch, noise):
        (_, aux), grads = learner.loss_and_grads(state, batch, noise, config=FAST)
        params = learner.learner_params(state)
        updates, opt_state = single_tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        critic = state.critic_state.replace(params=params.critic)
        critic = learner.update_slow_critic(
            critic, state.critic_state.step, FAST.slow_rate
        )
        new = state.replace(
            world_model_state=state.world_model_state.replace(
                params=params.world_model
            ),
            actor_state=state.actor_state.replace(params=params.actor),
            critic_state=critic.replace(step=critic.step + 1),
            retnorm=aux.retnorm,
        )
        return new, opt_state

    @jax.jit
    def sequential_step(state, batch, noise):
        _, grads = learner.loss_and_grads(state, batch, noise, config=FAST)
        world_model, _ = learner.apply_optimizer(
            state.world_model_state, grads.world_model
        )
        middle = state.replace(world_model_state=world_model)
        (_, aux), grads = learner.loss_and_grads(middle, batch, noise, config=FAST)
        actor, _ = learner.apply_optimizer(state.actor_state, grads.actor)
        critic, _ = learner.apply_optimizer(state.critic_state, grads.critic)
        critic = learner.update_slow_critic(
            critic, state.critic_state.step, FAST.slow_rate
        )
        return middle.replace(
            actor_state=actor, critic_state=critic, retnorm=aux.retnorm
        )

    joint, single, sequential = state, state, state
    opt_state = single_tx.init(learner.learner_params(state))
    for step, (batch, noise) in enumerate(batches):
        joint, _, _ = _jitted(FAST)(joint, batch, noise)
        single, opt_state = single_step(single, opt_state, batch, noise)
        sequential = sequential_step(sequential, batch, noise)
        ours = (learner.learner_params(joint), *_moments(joint))
        theirs = (learner.learner_params(single), opt_state[1].nu, opt_state[2].ema)
        assert _difference(ours, theirs) <= 4 * EPS, step
        if step == 0:  # one gradient for the world model in both variants
            wm = (sequential.world_model_state, joint.world_model_state)
            assert _difference(*(s.params for s in wm)) <= 4 * EPS
        for name in ("actor_state", "critic_state"):
            other, mine = getattr(sequential, name), getattr(joint, name)
            nu_difference = _difference(other.opt_state[1].nu, mine.opt_state[1].nu)
            assert nu_difference > 1e-4, name
    for name in ("actor_state", "critic_state"):
        start = getattr(state, name).params
        moved = [
            jax.tree.map(jnp.subtract, getattr(s, name).params, start)
            for s in (sequential, joint)
        ]
        assert _difference(*moved) > 1e-4, name
    for train_state in (joint.world_model_state, joint.actor_state, joint.critic_state):
        assert int(train_state.step) == 2
        assert all(int(s.count) == 2 for s in train_state.opt_state[1:])
    assert int(opt_state[1].count) == int(opt_state[3].count) == 2


def test_slow_critic_is_a_copy_after_the_first_update_then_an_ema():
    """2411f7d's ``SlowUpdater`` after each optimizer step: ``mix = 1`` at
    the first update -- the slow critic becomes the updated critic, exactly
    -- then ``slow_rate`` (dreamerv3_spec 3.18; 0.25 here, not the default,
    so the test reads it from the config). With ``warmup = 0`` the first
    update moves the critic, so the copy is not the initial critic."""
    state = tiny_learner(False, FAST)
    first, _, _ = _jitted(FAST)(state, replay_batch(9), _noise(10, config=FAST))
    for slow, critic, initial in zip(
        *(
            jax.tree.leaves(t)
            for t in (
                first.critic_state.target_params,
                first.critic_state.params,
                state.critic_state.params,
            )
        )
    ):
        np.testing.assert_array_equal(slow, critic)
        assert np.any(np.asarray(critic) != np.asarray(initial))
    second, _, _ = _jitted(FAST)(first, replay_batch(11), _noise(12, config=FAST))
    for slow, critic, previous in zip(
        *(
            jax.tree.leaves(t)
            for t in (
                second.critic_state.target_params,
                second.critic_state.params,
                first.critic_state.target_params,
            )
        )
    ):
        rate = np.float32(FAST.slow_rate)
        expected = rate * critic + (1 - rate) * previous
        np.testing.assert_allclose(
            slow, expected, rtol=4 * EPS, atol=4 * EPS * np.max(np.abs(expected))
        )


def test_train_step_returns_the_pre_update_posterior_and_the_new_normaliser():
    """The write-back entries are the posterior of the step's own forward
    pass, at the parameters before the update (dreamerv3_spec 5.9); the
    state carries the return normaliser after its update, which a
    non-training loss (the reference's report pass) only reads; the
    counters of the three train states advance together; the parameter
    norm is logged after the update (``jaxutils.py:527-530``)."""
    state = tiny_learner(False, FAST)
    batch, noise = replay_batch(13), _noise(14, config=FAST)
    new, entries, metrics = _jitted(FAST)(state, batch, noise)
    assert entries.deter.shape == (B, T, FAST.deter)
    assert entries.stoch.shape == (B, T, FAST.stoch)
    loss_fn = jax.jit(functools.partial(world_model_loss, WorldModel(FAST, OBS_DIM)))
    before = loss_fn(state.world_model_state.params, batch, noise.posterior)
    np.testing.assert_array_equal(entries.stoch, before.entries.stoch)
    np.testing.assert_allclose(entries.deter, before.entries.deter, atol=1e-6)

    loss = jax.jit(learner.loss_and_grads, static_argnames=("config", "training"))
    (_, aux), _ = loss(state, batch, noise, config=FAST)
    updated = (aux.retnorm.lo, aux.retnorm.hi)
    np.testing.assert_allclose((new.retnorm.lo, new.retnorm.hi), updated, rtol=1e-6)
    assert new.retnorm.hi != state.retnorm.hi
    (_, report), _ = loss(state, batch, noise, config=FAST, training=False)
    assert (report.retnorm.lo, report.retnorm.hi) == (
        state.retnorm.lo,
        state.retnorm.hi,
    )

    train_states = (new.world_model_state, new.actor_state, new.critic_state)
    assert {int(ts.step) for ts in train_states} == {1}
    assert int(metrics["opt_grad_steps"]) == 1

    after = float(optax.global_norm(learner.learner_params(new)))
    before_norm = float(optax.global_norm(learner.learner_params(state)))
    assert abs(after - before_norm) > 1e-4 * after  # the check can tell them apart
    np.testing.assert_allclose(float(metrics["opt_param_norm"]), after, rtol=1e-6)


def test_init_learner():
    """The return normaliser is built from the config's rate and limit, at
    ``lo = hi = 0``; the slow critic is a copy of the critic in buffers of
    its own, as are ``lo`` and ``hi``, so that a fresh state can be donated
    to the jitted :func:`learner.train_step` (XLA refuses to donate one
    buffer twice)."""
    custom = dataclasses.replace(CONFIG, retnorm_rate=0.05, retnorm_limit=2.0)
    for config in (CONFIG, custom):
        state = learner.init_learner(_key(0), config, OBS_DIM, ACTION_DIM, False)
        retnorm = state.retnorm
        assert (retnorm.rate, retnorm.limit) == (
            config.retnorm_rate,
            config.retnorm_limit,
        )
        assert float(retnorm.lo) == float(retnorm.hi) == 0.0
    critic = state.critic_state
    for slow, online in zip(
        *(jax.tree.leaves(t) for t in (critic.target_params, critic.params))
    ):
        np.testing.assert_array_equal(slow, online)
    step = jax.jit(learner.train_step, static_argnames="config", donate_argnums=0)
    new, _, _ = step(state, replay_batch(15), _noise(16, config=custom), config=custom)
    assert int(jax.block_until_ready(new.critic_state.step)) == 1


def test_draw_train_noise():
    """Every draw of a step: Gumbel noise for the latents (and a discrete
    actor), standard normal noise for a continuous actor, start-major
    imagination noise with the start action at index 0. The streams are
    independent: no two share a key (with ``threefry_partitionable``, two
    draws from one key share their leading values), and no two starts share
    their prior or action noise."""
    for discrete, mean in ((False, 0.0), (True, np.euler_gamma)):
        noise = _noise(0, discrete)
        a = NUM_ACTIONS if discrete else ACTION_DIM
        assert noise.posterior.shape == (B, T, CONFIG.stoch, CONFIG.classes)
        assert noise.prior.shape == (B * T, H, CONFIG.stoch, CONFIG.classes)
        assert noise.action.shape == (B * T, H + 1, a)
        big = learner.draw_train_noise(
            _key(1), dataclasses.replace(CONFIG, imag_horizon=200), B, T, a, discrete
        )
        assert abs(float(np.mean(big.action)) - mean) < 0.05
        assert abs(float(np.mean(big.prior)) - np.euler_gamma) < 0.05
        streams = [np.asarray(x).ravel() for x in noise]
        for i in range(len(streams)):
            for j in range(i + 1, len(streams)):
                size = min(streams[i].size, streams[j].size)
                assert not np.any(streams[i][:size] == streams[j][:size]), (i, j)
        for x in (noise.prior, noise.action):
            rows = np.asarray(x).reshape(B * T, -1)
            assert len(np.unique(rows, axis=0)) == B * T


# --------------------------------------------------------------- optimizer


def test_make_optimizer_honours_every_optimizer_field():
    """:func:`make_optimizer` passes every optimizer field of the config to
    :func:`laprop` (all at non-default values here): gradients alternating
    between scale 1 and 1e-4 and a zero parameter tensor make ``agc``,
    ``agc_pmin``, ``eps`` and the betas matter."""
    config = dataclasses.replace(
        CONFIG,
        learning_rate=3e-3,
        agc=0.5,
        agc_pmin=1e-2,
        beta1=0.8,
        beta2=0.99,
        eps=1e-6,
        warmup=2,
    )
    ours = make_optimizer(config)
    theirs = laprop(3e-3, agc=0.5, pmin=1e-2, beta1=0.8, beta2=0.99, eps=1e-6, warmup=2)
    defaults = make_optimizer(CONFIG)
    params = {"a": jnp.linspace(-1.0, 1.0, 12).reshape(4, 3), "b": jnp.zeros((5,))}
    states = [tx.init(params) for tx in (ours, theirs, defaults)]
    rng = np.random.default_rng(0)
    for k in range(4):
        scale = 1e-4 if k % 2 else 1.0
        grads = {
            name: jnp.asarray(rng.normal(0, scale, p.shape), jnp.float32)
            for name, p in params.items()
        }
        updates = []
        for i, tx in enumerate((ours, theirs, defaults)):
            update, states[i] = tx.update(grads, states[i], params)
            updates.append(update)
        for mine, expected in zip(*(jax.tree.leaves(u) for u in updates[:2])):
            np.testing.assert_array_equal(mine, expected)
        if k:  # the default optimizer's updates differ (update 0 is 0 for both)
            assert _difference(updates[0], updates[2]) > 0.1


def test_warmup_gives_learning_rate_zero_at_the_first_update():
    """``lr min(k / warmup, 1)`` with ``k`` read before incrementing
    (dreamerv3_spec 4.7): update 0 is exactly 0 but accumulates the
    moments; update ``k`` is ``k / warmup`` of the full one."""
    params = {"w": jnp.ones((3,))}
    grads = [{"w": jnp.asarray([1.0, -2.0, 0.5]) * (i + 1)} for i in range(5)]
    warm, full = laprop(1e-2, warmup=3, agc=0.0), laprop(1e-2, warmup=0, agc=0.0)
    warm_state, full_state = warm.init(params), full.init(params)
    for k, g in enumerate(grads):
        warm_update, warm_state = warm.update(g, warm_state, params)
        full_update, full_state = full.update(g, full_state, params)
        factor = min(k / 3, 1.0)
        np.testing.assert_allclose(
            warm_update["w"], factor * full_update["w"], rtol=1e-6
        )
        if k == 0:
            assert not np.any(warm_update["w"]) and np.any(warm_state[0].nu["w"])


def test_agc_clips_each_tensor_by_its_own_norm():
    """``g / max(1, |g| / (0.3 max(1e-3, |p|)))`` with the Euclidean norms of
    the whole tensor (dreamerv3_spec 4.4): a tensor within its bound is
    unchanged, one beyond it is scaled to the bound, a zero tensor's bound
    uses ``pmin``; tensors do not affect each other."""
    params = {
        "small": jnp.asarray([3.0, 4.0]),
        "zero": jnp.zeros((2, 2)),
        "big": jnp.full((4,), 10.0),
    }
    grads = {
        "small": jnp.asarray([6.0, 8.0]),
        "zero": jnp.ones((2, 2)),
        "big": jnp.asarray([1.0, -1.0, 0.0, 2.0]),
    }
    clipped, _ = scale_by_agc(0.3, 1e-3).update(grads, optax.EmptyState(), params)
    np.testing.assert_allclose(jnp.linalg.norm(clipped["small"]), 0.3 * 5.0, rtol=1e-6)
    np.testing.assert_allclose(
        clipped["small"] / jnp.linalg.norm(clipped["small"]),
        grads["small"] / 10.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(jnp.linalg.norm(clipped["zero"]), 0.3 * 1e-3, rtol=1e-5)
    np.testing.assert_array_equal(clipped["big"], grads["big"])  # |g| = 2.4 < 6


def test_laprop_matches_a_hand_recursion():
    """Algorithm I in float64: per-tensor AGC on the raw gradient, RMS
    normalisation with bias correction and ``eps`` after the square root,
    momentum on the normalised update with bias correction, ``-lr`` and the
    warmup (dreamerv3_spec 4.3-4.7). The first RMS step is ``sign(g)``.
    Ajax computes in float32 as the reference does, where the bias
    correction ``1 - 0.999^t`` loses about ``eps / 0.001`` relative at small
    ``t``: the updates agree to 1e-4 of a learning rate."""
    rng = np.random.default_rng(0)
    lr, warmup, b1, b2, eps = 1e-2, 3, 0.9, 0.999, 1e-20
    params = {"a": rng.normal(0, 1, (4, 3)), "b": rng.normal(0, 1e-4, (5,))}
    tx = laprop(lr, warmup=warmup)
    state = tx.init(jax.tree.map(jnp.float32, params))
    nu = jax.tree.map(np.zeros_like, params)
    mu = jax.tree.map(np.zeros_like, params)
    for k in range(6):
        grads = {"a": rng.normal(0, 2, (4, 3)), "b": rng.normal(0, 1, (5,))}
        ours, state = tx.update(
            jax.tree.map(jnp.float32, grads), state, jax.tree.map(jnp.float32, params)
        )
        for name, g in grads.items():
            upper = 0.3 * max(1e-3, np.linalg.norm(params[name]))
            g = g / max(1.0, np.linalg.norm(g) / upper)
            nu[name] = b2 * nu[name] + (1 - b2) * g * g
            u = g / (np.sqrt(nu[name] / (1 - b2 ** (k + 1))) + eps)
            if k == 0:
                np.testing.assert_allclose(u, np.sign(g))
            mu[name] = b1 * mu[name] + (1 - b1) * u
            update = -lr * min(k / warmup, 1.0) * mu[name] / (1 - b1 ** (k + 1))
            np.testing.assert_allclose(ours[name], update, rtol=1e-4, atol=1e-4 * lr)
            params[name] = params[name] + update


def test_learning_rate_schedule_is_composed_with_the_warmup():
    """A callable learning rate is read at the update count ``k`` and the
    warmup multiplies it."""
    params = {"w": jnp.ones((2,))}
    grads = {"w": jnp.asarray([1.0, -1.0])}
    tx = laprop(lambda k: 1e-2 / (k + 1), warmup=2, agc=0.0)
    state = tx.init(params)
    for k in range(4):
        update, state = tx.update(grads, state, params)
        expected = -1e-2 / (k + 1) * min(k / 2, 1.0) * np.sign([1.0, -1.0])
        np.testing.assert_allclose(update["w"], expected, rtol=1e-4)


def test_actor_critic_config_validation():
    for field, value in (
        ("actor_layers", 0),
        ("imag_horizon", 0),
        ("actor_unimix", 1.0),
        ("minstd", 0.0),
        ("maxstd", 0.05),
        ("lam", 1.5),
        ("slow_rate", -0.1),
        ("beta2", 1.0),
        ("warmup", -1),
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(CONFIG, **{field: value})
    assert CONFIG.gamma == pytest.approx(1 - 1 / 333)
