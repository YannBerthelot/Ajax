"""TD-MPC2 update pieces: gradient routing, clipping, optimizers, sampling.

The end-to-end numbers are pinned against the real reference in
``test_tdmpc2_parity.py``; these tests pin the properties the parity fixture
cannot isolate (stop-gradients, the torch clip formula at small norms, the
saturated policy, the independence of the draws, dropout keys (the fixture
has no dropout), the encoder learning rate).
References are to ``nicklashansen/tdmpc2@5f6fade:tdmpc2/`` and
``docs/world_models/tdmpc2_spec.md``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from ajax.agents.TDMPC2 import core
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2UpdateState
from ajax.normalizers import RunningScale
from ajax.state import LoadedTrainState

TINY = TDMPC2Config(latent_dim=16, enc_dim=32, mlp_dim=32, num_q=5, dropout=0.01)
OBS, ACT, B = 5, 2, 8


def _with_random_heads(state: TDMPC2UpdateState, key: jax.Array) -> TDMPC2UpdateState:
    """Non-zero reward / Q heads, so every world-model parameter gets gradient."""
    wm = state.world_model_state
    params = jax.tree_util.tree_map(lambda x: x, wm.params)
    k1, k2, k3 = jax.random.split(key, 3)
    rew = params["reward"]["out"]
    q = params["q"]["members"]["out"]
    rew["kernel"] = 0.5 * jax.random.normal(k1, rew["kernel"].shape)
    q["kernel"] = 0.5 * jax.random.normal(k2, q["kernel"].shape)
    target = jax.tree_util.tree_map(lambda x: x, params["q"])
    target["members"]["out"]["kernel"] = 0.5 * jax.random.normal(k3, q["kernel"].shape)
    return state.replace(
        world_model_state=wm.replace(params=params, target_params=target)
    )


@pytest.fixture(scope="module")
def state() -> TDMPC2UpdateState:
    init = core.create_update_state(jax.random.PRNGKey(0), TINY, OBS, ACT)
    return _with_random_heads(init, jax.random.PRNGKey(1))


@pytest.fixture(scope="module")
def batch() -> core.TDMPC2Batch:
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
    return core.TDMPC2Batch(
        obs=jax.random.normal(k1, (TINY.horizon + 1, B, OBS)),
        action=jax.random.uniform(k2, (TINY.horizon, B, ACT), minval=-1, maxval=1),
        reward=3.0 * jax.random.normal(k3, (TINY.horizon, B)),
    )


@pytest.fixture(scope="module")
def noise() -> core.UpdateNoise:
    return core.draw_update_noise(jax.random.PRNGKey(3), TINY, B, ACT)


def _is_zero(tree) -> bool:
    return all(not np.any(np.asarray(x)) for x in jax.tree_util.tree_leaves(tree))


def _all_nonzero(tree) -> bool:
    return all(np.any(np.asarray(x)) for x in jax.tree_util.tree_leaves(tree))


# ------------------------------------------------------------------ utilities


@pytest.mark.parametrize(
    "length, gamma", [(500, 0.99), (100, 0.95), (1000, 0.995), (200, 0.975), (20, 0.95)]
)
def test_discount_heuristic(length, gamma):
    """``tdmpc2.py:36-49``; Pendulum-v1 (T = 200) gives 0.975 (DESIGN §2)."""
    assert core.discount_from_episode_length(length) == pytest.approx(gamma)


def _torch_clip_grad_norm(grads, max_norm, extra=()):
    """``torch.nn.utils.clip_grad_norm_`` (torch 2.2.2) on numpy arrays; ``extra``
    are gradients included in the norm (and scaled) but not returned."""
    norms = [np.linalg.norm(g.ravel()) for g in (*grads, *extra)]
    total = np.linalg.norm(np.array(norms))
    coef = min(max_norm / (total + 1e-6), 1.0)
    return [g * coef for g in grads], total


@pytest.mark.parametrize("scale", [1e-7, 1e-3, 1.0, 50.0])
def test_clip_grad_norm_is_torch_clip_grad_norm(scale):
    rng = np.random.default_rng(0)
    grads = [scale * rng.normal(size=s).astype(np.float32) for s in ((3, 4), (7,))]
    stale = [scale * rng.normal(size=(5,)).astype(np.float32)]
    max_norm = 2.0 * scale  # binds; at 1e-7 the +1e-6 dominates the coefficient
    expected, expected_norm = _torch_clip_grad_norm(grads, max_norm, stale)
    clipped, norm = core.clip_grad_norm(
        [jnp.asarray(g) for g in grads],
        max_norm,
        extra_sq_norm=float(np.sum(stale[0].astype(np.float64) ** 2)),
    )
    np.testing.assert_allclose(norm, expected_norm, rtol=1e-6)
    for c, e in zip(clipped, expected):
        np.testing.assert_allclose(c, e, rtol=1e-5)
    unclipped, _ = core.clip_grad_norm(
        [jnp.asarray(g) for g in grads], 1e3 * scale + 1.0
    )
    for u, g in zip(unclipped, grads):
        np.testing.assert_array_equal(u, g)


def test_draw_update_noise_shapes_and_pairs():
    draw = jax.jit(lambda k: core.draw_update_noise(k, TINY, B, ACT))
    n = draw(jax.random.PRNGKey(0))
    assert n.td_eps.shape == (3, B, ACT) and n.pi_eps.shape == (4, B, ACT)
    seen = set()
    for i in range(200):
        pair = np.asarray(core.draw_q_pair(jax.random.PRNGKey(i), 5))
        assert pair.dtype == np.int32 and pair[0] != pair[1]
        assert 0 <= pair.min() and pair.max() < 5
        seen.add(tuple(pair))
    assert len(seen) == 20  # every ordered pair of 5 heads
    assert sorted(np.asarray(core.draw_q_pair(jax.random.PRNGKey(0), 2))) == [0, 1]
    again = draw(jax.random.PRNGKey(0))
    jax.tree_util.tree_map(np.testing.assert_array_equal, n, again)


def test_update_noise_draws_are_independent():
    """Every draw of one update comes from its own key: the TD target and the
    policy loss get different noise and pairs, and each Q pass its own dropout
    key (a reused key would give equal masks on the equal-shape passes)."""
    draw = jax.jit(lambda k: core.draw_update_noise(k, TINY, B, ACT))
    same_pair, td_eps, pi_eps = 0, [], []
    n_keys = 300
    for i in range(n_keys):
        n = draw(jax.random.PRNGKey(i))
        dropout_keys = {
            tuple(np.asarray(k).ravel())
            for k in (n.td_dropout, n.value_dropout, n.pi_dropout)
        }
        assert len(dropout_keys) == 3
        assert not np.any(n.td_eps == n.pi_eps[: TINY.horizon])
        same_pair += int(np.array_equal(n.td_pair, n.pi_pair))
        td_eps.append(np.ravel(n.td_eps))
        pi_eps.append(np.ravel(n.pi_eps[: TINY.horizon]))
    assert same_pair < 0.15 * n_keys  # 1/20 of the draws for independent pairs
    corr = np.corrcoef(np.concatenate(td_eps), np.concatenate(pi_eps))[0, 1]
    assert abs(corr) < 0.05


def test_reduce_q_pair_on_decoded_values():
    two_hot = TINY.two_hot
    logits = jax.random.normal(jax.random.PRNGKey(0), (5, 3, 101)) * 3.0
    decoded = two_hot.decode(logits)
    pair = jnp.array([4, 1], jnp.int32)
    np.testing.assert_array_equal(
        core.reduce_q_pair(logits, pair, "min", two_hot),
        jnp.minimum(decoded[4], decoded[1]),
    )
    np.testing.assert_array_equal(
        core.reduce_q_pair(logits, pair, "avg", two_hot), (decoded[4] + decoded[1]) / 2
    )
    with pytest.raises(ValueError, match="kind"):
        core.reduce_q_pair(logits, pair, "max", two_hot)  # type: ignore[arg-type]


def test_q_pair_logits_evaluates_only_the_pair(state):
    """Without dropout, the full ensemble's rows ``pair`` in the pair's order
    (traced pair included); with a dropout key, deterministic per key and an
    independent mask per member (a member paired with itself differs)."""
    wm = state.world_model_state
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(4), 4)
    z = jax.random.normal(k1, (B, TINY.latent_dim))
    action = jax.random.uniform(k2, (B, ACT), minval=-1, maxval=1)
    full = core.q_logits(wm.apply_fn, wm.params, z, action)
    tol = {"rtol": 1e-5, "atol": 1e-6}
    for pair in ([4, 1], [0, 2]):
        idx = jnp.array(pair, jnp.int32)
        logits = core.q_pair_logits(TINY, wm.params, z, action, idx)
        assert logits.shape == (2, B, TINY.num_bins)
        np.testing.assert_allclose(logits, full[idx], **tol)
    traced = jax.jit(lambda p: core.q_pair_logits(TINY, wm.params, z, action, p))
    np.testing.assert_allclose(
        traced(jnp.array([3, 0])), full[jnp.array([3, 0])], **tol
    )

    config = TINY.replace(dropout=0.3)

    def dropped(pair, key):
        return core.q_pair_logits(
            config, wm.params, z, action, jnp.array(pair), dropout_key=key
        )

    np.testing.assert_array_equal(dropped([4, 1], k3), dropped([4, 1], k3))
    assert not np.allclose(dropped([4, 1], k3), dropped([4, 1], k4))
    assert not np.allclose(dropped([4, 1], k3), full[jnp.array([4, 1])])
    twice = dropped([1, 1], k3)
    assert not np.allclose(twice[0], twice[1])


# ------------------------------------------------------------ policy sampling


def _reference_pi(mu, raw, eps, low=-10.0, high=2.0):
    """``world_model.py:132-148`` with ``math.py:12-45``, float64 numpy."""
    mu, raw, eps = (np.asarray(x, np.float64) for x in (mu, raw, eps))
    log_std = low + 0.5 * (high - low) * (np.tanh(raw) + 1)
    size = eps.shape[-1]
    residual = (-0.5 * eps**2 - log_std).sum(-1, keepdims=True)
    log_pi = (residual - 0.5 * math.log(2 * math.pi)) * size
    pi = np.tanh(mu + eps * np.exp(log_std))
    log_pi = log_pi - np.log(np.maximum(1 - pi**2, 0) + 1e-6).sum(-1, keepdims=True)
    return np.tanh(mu), pi, log_pi[..., 0], log_std


def test_squashed_gaussian_is_the_paper_era_policy():
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    mu = jax.random.normal(keys[0], (4, 6, 3))
    # log_std in about (-9.7, -2.8): away from tanh saturation, where
    # 1 - tanh(u)^2 cancels catastrophically in float32 (the reference's dtype).
    raw = -1.0 + 0.5 * jax.random.normal(keys[1], (4, 6, 3))
    eps = jax.random.normal(keys[2], (4, 6, 3))
    out = core.squashed_gaussian(mu, raw, eps, TINY)
    mean, action, log_pi, log_std = _reference_pi(mu, raw, eps)
    np.testing.assert_allclose(out.mean, mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.action, action, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.log_std, log_std, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out.log_pi, log_pi, rtol=1e-5, atol=1e-5)


def test_log_pi_stays_finite_when_tanh_saturates():
    """For |u| above ~9, tanh(u) is exactly +-1 in float32 (and here in float64
    too); the ``+ 1e-6`` of ``math.py:35-37`` keeps log_pi and its gradient
    finite."""
    mean = jnp.array([[20.0, -20.0]])
    raw, eps = jnp.zeros((1, 2)), jnp.ones((1, 2))
    out = core.squashed_gaussian(mean, raw, eps, TINY)
    np.testing.assert_array_equal(np.abs(out.action), 1.0)
    np.testing.assert_allclose(out.log_pi, _reference_pi(mean, raw, eps)[2], rtol=1e-6)

    def total_log_pi(m):
        return jnp.sum(core.squashed_gaussian(m, raw, eps, TINY).log_pi)

    assert np.all(np.isfinite(jax.grad(total_log_pi)(mean)))


def test_log_pi_tanh_correction_carries_gradient():
    """The Gaussian part of the paper-era log_pi does not depend on the mean;
    only the unscaled tanh correction does, and it is differentiated."""
    raw = jnp.zeros((5, 2))
    eps = jax.random.normal(jax.random.PRNGKey(0), (5, 2))
    mu = jax.random.normal(jax.random.PRNGKey(1), (5, 2))

    def total_log_pi(m):
        return jnp.sum(core.squashed_gaussian(m, raw, eps, TINY).log_pi)

    grad = jax.grad(total_log_pi)(mu)
    assert np.all(np.abs(grad) > 0)
    # d/dmu [-log(1 - tanh(u)^2 + 1e-6)] ~= 2 tanh(u) away from saturation.
    u = mu + eps * jnp.exp(TINY.log_std_min + 0.5 * 12.0 * (jnp.tanh(raw) + 1))
    np.testing.assert_allclose(grad, 2 * jnp.tanh(u), rtol=1e-3)


# ------------------------------------------------------------ gradient routing


def test_td_target_carries_no_gradient(state, batch, noise):
    wm, pi = state.world_model_state, state.actor_state

    def target_sum(wm_params, target_q, pi_params):
        y, next_z = core.td_target(
            wm_apply=wm.apply_fn,
            pi_apply=pi.apply_fn,
            wm_params=wm_params,
            target_q_params=target_q,
            pi_params=pi_params,
            next_obs=batch.obs[1:],
            reward=batch.reward,
            gamma=0.99,
            eps=noise.td_eps,
            pair=noise.td_pair,
            dropout_key=noise.td_dropout,
            config=TINY,
        )
        return jnp.sum(y) + jnp.sum(next_z)

    grads = jax.grad(target_sum, argnums=(0, 1, 2))(
        wm.params, wm.target_params, pi.params
    )
    assert _is_zero(grads)


def test_world_model_loss_reaches_every_world_model_part_and_not_the_policy(
    state, batch, noise
):
    wm, pi = state.world_model_state, state.actor_state

    def loss(wm_params, pi_params):
        td, next_z = core.td_target(
            wm_apply=wm.apply_fn,
            pi_apply=pi.apply_fn,
            wm_params=wm_params,
            target_q_params=wm.target_params,
            pi_params=pi_params,
            next_obs=batch.obs[1:],
            reward=batch.reward,
            gamma=0.99,
            eps=noise.td_eps,
            pair=noise.td_pair,
            dropout_key=noise.td_dropout,
            config=TINY,
        )
        total, _ = core.world_model_loss(
            wm_params,
            wm_apply=wm.apply_fn,
            batch=batch,
            next_z=next_z,
            td_targets=td,
            dropout_key=noise.value_dropout,
            config=TINY,
        )
        return total

    wm_grads, pi_grads = jax.grad(loss, argnums=(0, 1))(wm.params, pi.params)
    assert _is_zero(pi_grads)
    for part in ("encoder", "dynamics", "reward", "q"):
        assert _all_nonzero(wm_grads[part]), part


@pytest.mark.parametrize("entropy_coef", [1e-4, 0.0])
def test_policy_loss_reaches_only_the_policy_through_the_action(
    state, batch, noise, entropy_coef
):
    """Q parameters and latents are stop-gradiented; the policy still learns
    from Q through its action input (entropy_coef = 0 leaves only that path)."""
    config = TINY.replace(entropy_coef=entropy_coef)
    wm, pi = state.world_model_state, state.actor_state
    zs = jax.random.uniform(jax.random.PRNGKey(5), (4, B, 16))

    def loss(pi_params, wm_params, latents):
        value, _ = core.policy_loss(
            pi_params,
            pi_apply=pi.apply_fn,
            wm_apply=wm.apply_fn,
            wm_params=wm_params,
            zs=latents,
            q_scale=state.q_scale,
            eps=noise.pi_eps,
            pair=noise.pi_pair,
            dropout_key=noise.pi_dropout,
            config=config,
        )
        return value

    pi_grads, wm_grads, z_grads = jax.grad(loss, argnums=(0, 1, 2))(
        pi.params, wm.params, zs
    )
    assert _is_zero(wm_grads) and _is_zero(z_grads)
    assert _all_nonzero(pi_grads)


def test_policy_loss_updates_the_scale_before_dividing(state, noise):
    wm, pi = state.world_model_state, state.actor_state
    zs = jax.random.uniform(jax.random.PRNGKey(6), (4, B, 16))
    scale = RunningScale.create(rate=TINY.tau)
    _, (new_scale, aux) = core.policy_loss(
        pi.params,
        pi_apply=pi.apply_fn,
        wm_apply=wm.apply_fn,
        wm_params=wm.params,
        zs=zs,
        q_scale=scale,
        eps=noise.pi_eps,
        pair=noise.pi_pair,
        dropout_key=None,
        config=TINY,
    )
    action = core.policy_sample(pi.apply_fn, pi.params, zs, noise.pi_eps, TINY).action
    logits = core.q_logits(wm.apply_fn, wm.params, zs, action)
    q0 = core.reduce_q_pair(logits, noise.pi_pair, "avg", TINY.two_hot)[0]
    expected = scale.update(q0)
    np.testing.assert_allclose(new_scale.value, expected.value, rtol=1e-6)
    assert float(new_scale.value) > 1.0  # the random heads spread Q beyond 1
    np.testing.assert_allclose(aux["pi_scale"], new_scale.value)


def test_paper_era_td_target_has_q_dropout(state, batch, noise):
    """With a dropout key the target-Q pass is stochastic (5f6fade), without
    one it is deterministic."""
    config = TINY.replace(dropout=0.3)
    stateful = core.create_update_state(jax.random.PRNGKey(0), config, OBS, ACT)
    stateful = _with_random_heads(stateful, jax.random.PRNGKey(1))
    wm, pi = stateful.world_model_state, stateful.actor_state

    def target(key):
        y, _ = core.td_target(
            wm_apply=wm.apply_fn,
            pi_apply=pi.apply_fn,
            wm_params=wm.params,
            target_q_params=wm.target_params,
            pi_params=pi.params,
            next_obs=batch.obs[1:],
            reward=batch.reward,
            gamma=0.99,
            eps=noise.td_eps,
            pair=noise.td_pair,
            dropout_key=key,
            config=config,
        )
        return y

    np.testing.assert_array_equal(target(None), target(None))
    assert not np.allclose(target(jax.random.PRNGKey(0)), target(None))
    assert not np.allclose(target(jax.random.PRNGKey(0)), target(jax.random.PRNGKey(1)))


# ----------------------------------------------------------------- optimizers


@pytest.mark.parametrize("schedule", [False, True])
def test_world_model_adam_scales_the_encoder_learning_rate(state, schedule):
    lr = (lambda count: 3e-4 * 0.5 ** (count / 10)) if schedule else 3e-4
    tx = core.make_world_model_tx(lr, enc_lr_scale=0.3)
    params = state.world_model_state.params
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    updates, _ = tx.update(grads, tx.init(params), params)
    # Adam's first step is lr * g / (|g| + eps) ~ lr. optax rounds its bias
    # correction 1 - 0.999 in float32 (torch uses Python floats): 6.4e-6 relative.
    for part in ("encoder", "dynamics", "reward", "q"):
        expected = 9e-5 if part == "encoder" else 3e-4
        for leaf in jax.tree_util.tree_leaves(updates[part]):
            np.testing.assert_allclose(leaf, -expected, rtol=1e-5)


def test_policy_adam_uses_eps_1e5(state):
    tx = core.make_policy_tx(3e-4)
    params = state.actor_state.params
    grads = jax.tree_util.tree_map(lambda x: jnp.full_like(x, 1e-5), params)
    updates, _ = tx.update(grads, tx.init(params), params)
    for leaf in jax.tree_util.tree_leaves(updates):
        np.testing.assert_allclose(leaf, -3e-4 * 0.5, rtol=1e-5)


# ------------------------------------------------------------------- update


@struct.dataclass
class _AgentStateLike:
    """Stands in for M4's agent state: the four fields plus others."""

    world_model_state: LoadedTrainState
    actor_state: LoadedTrainState
    q_scale: RunningScale
    pi_gradnorm_sq: jax.Array
    other: jax.Array


def test_update_is_jittable_reproducible_and_threads_its_state(state, batch, noise):
    config = TINY
    step = jax.jit(lambda s, b, n: core.update(s, b, n, config=config, gamma=0.99))
    agent_like = _AgentStateLike(
        world_model_state=state.world_model_state,
        actor_state=state.actor_state,
        q_scale=state.q_scale,
        pi_gradnorm_sq=state.pi_gradnorm_sq,
        other=jnp.arange(3),
    )
    new, aux = step(agent_like, batch, noise)
    again, aux_again = step(agent_like, batch, noise)
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, (new, aux), (again, aux_again)
    )
    np.testing.assert_array_equal(new.other, agent_like.other)
    assert int(new.world_model_state.step) == 1 and int(new.actor_state.step) == 1
    assert set(aux) == {
        "consistency_loss",
        "reward_loss",
        "value_loss",
        "total_loss",
        "pi_loss",
        "pi_entropy",
        "pi_log_std",
        "pi_scale",
        "grad_norm",
        "pi_grad_norm",
    }
    # The target moved tau of the way to the post-step online Q.
    expected_target = jax.tree_util.tree_map(
        lambda q, t: 0.01 * q + 0.99 * t,
        new.world_model_state.params["q"],
        state.world_model_state.target_params,
    )
    jax.tree_util.tree_map(
        lambda a, e: np.testing.assert_allclose(a, e, rtol=1e-6, atol=1e-8),
        new.world_model_state.target_params,
        expected_target,
    )
    # The carried norm is the post-clip policy gradient norm.
    clipped_norm = min(float(aux["pi_grad_norm"]), config.grad_clip_norm)
    np.testing.assert_allclose(np.sqrt(new.pi_gradnorm_sq), clipped_norm, rtol=1e-5)


def test_each_dropout_key_reaches_its_q_pass(batch, noise):
    """Paper-era Q dropout is on in all three Q passes of an update (module
    docstring of ``core``): each key changes its own pass and nothing
    computed before it."""
    config = TINY.replace(dropout=0.3)
    state = core.create_update_state(jax.random.PRNGKey(0), config, OBS, ACT)
    state = _with_random_heads(state, jax.random.PRNGKey(1))
    step = jax.jit(lambda n: core.update(state, batch, n, config=config, gamma=0.99))
    _, base = step(noise)
    other = jax.random.PRNGKey(99)
    _, td = step(noise.replace(td_dropout=other))  # target-Q pass of the TD target
    _, value = step(noise.replace(value_dropout=other))  # value loss' online Q
    _, pi = step(noise.replace(pi_dropout=other))  # policy loss' online Q
    for aux in (td, value):
        assert float(aux["value_loss"]) != float(base["value_loss"])
        assert float(aux["reward_loss"]) == float(base["reward_loss"])
    assert float(pi["pi_loss"]) != float(base["pi_loss"])
    assert float(pi["value_loss"]) == float(base["value_loss"])


def test_aux_reports_the_policy_loss_sample(state, batch, noise):
    """``pi_entropy = -mean(log_pi)`` and ``pi_log_std = mean(log_std)`` of
    the policy loss' sample at the pre-step latents."""
    wm, pi = state.world_model_state, state.actor_state
    td, next_z = core.td_target(
        wm_apply=wm.apply_fn,
        pi_apply=pi.apply_fn,
        wm_params=wm.params,
        target_q_params=wm.target_params,
        pi_params=pi.params,
        next_obs=batch.obs[1:],
        reward=batch.reward,
        gamma=0.99,
        eps=noise.td_eps,
        pair=noise.td_pair,
        dropout_key=noise.td_dropout,
        config=TINY,
    )
    _, (_, zs) = core.world_model_loss(
        wm.params,
        wm_apply=wm.apply_fn,
        batch=batch,
        next_z=next_z,
        td_targets=td,
        dropout_key=noise.value_dropout,
        config=TINY,
    )
    sample = core.policy_sample(pi.apply_fn, pi.params, zs, noise.pi_eps, TINY)
    _, aux = core.update(state, batch, noise, config=TINY, gamma=0.99)
    np.testing.assert_allclose(aux["pi_entropy"], -np.mean(sample.log_pi), rtol=1e-6)
    np.testing.assert_allclose(aux["pi_log_std"], np.mean(sample.log_std), rtol=1e-6)
