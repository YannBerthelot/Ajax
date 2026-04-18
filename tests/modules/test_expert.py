"""Tests for ajax.modules.expert — composable expert-guidance modifiers."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from ajax.modules.expert import (
    augment_obs_if_needed,
    blend_modify_target,
    compute_behavior_kpis,
    compute_expert_diagnostics,
    compute_online_bc_loss,
    detach_obs_expert_dims,
    ibrl_modify_target,
    mc_correction_modify_target,
    residual_action_transform,
)


def _fake_critic_state(fn):
    """Build a minimal LoadedTrainState-like object.

    ``predict_value`` only calls critic_state.apply_fn(params, x), so a
    SimpleNamespace with callable apply_fn is sufficient.
    """
    return SimpleNamespace(
        apply_fn=lambda params, x: fn(params, x),
        params={"online": 1.0},
        target_params={"target": 1.0},
    )


# ---------------------------------------------------------------------------
# IBRL target modifier
# ---------------------------------------------------------------------------


def test_ibrl_modifier_adds_positive_gap_when_expert_better():
    def apply_fn(params, x):
        # Two-critic ensemble: first critic outputs 2.0, second outputs 3.0
        # → min over axis=0 keepdims=False is 2.0 for every element.
        batch = jnp.ones((x.shape[0], 1)) * 2.0
        return jnp.stack([batch, batch + 1.0], axis=0)

    critic_state = _fake_critic_state(apply_fn)
    target_q = jnp.zeros((4, 1))
    min_q_from_core = jnp.ones((4, 1)) * -1.0  # pretend policy Q = -1
    out = ibrl_modify_target(
        target_q,
        min_q_from_core,
        critic_state,
        next_observations=jnp.zeros((4, 3)),
        next_expert_actions=jnp.zeros((4, 2)),
    )
    # Gap = min(expert_q) - min_q_from_core = 2.0 - (-1.0) = 3.0
    assert jnp.allclose(out, target_q + 3.0)


def test_ibrl_modifier_zero_gap_when_expert_worse():
    def apply_fn(params, x):
        batch = jnp.ones((x.shape[0], 1)) * -5.0
        return jnp.stack([batch, batch], axis=0)

    critic_state = _fake_critic_state(apply_fn)
    target_q = jnp.zeros((3, 1))
    min_q_from_core = jnp.ones((3, 1)) * 1.0  # policy beats expert
    out = ibrl_modify_target(
        target_q,
        min_q_from_core,
        critic_state,
        next_observations=jnp.zeros((3, 3)),
        next_expert_actions=jnp.zeros((3, 2)),
    )
    # Gap clipped to 0 → output unchanged.
    assert jnp.allclose(out, target_q)


# ---------------------------------------------------------------------------
# Blend target modifier
# ---------------------------------------------------------------------------


def test_blend_modify_target_matches_convex_combination():
    target_q = jnp.array([[1.0], [2.0], [4.0]])
    v_expert = jnp.array([[5.0], [5.0], [5.0]])
    alpha = jnp.array([[0.25]])
    out, diag = blend_modify_target(target_q, v_expert, alpha)
    expected = 0.75 * target_q + 0.25 * v_expert
    assert jnp.allclose(out, expected)
    assert diag.shape == (1,)


# ---------------------------------------------------------------------------
# MC correction modifier
# ---------------------------------------------------------------------------


def test_mc_correction_preserves_low_variance_entries():
    def apply_fn(params, x):
        batch = jnp.ones((x.shape[0], 1)) * 9.0
        return jnp.stack([batch], axis=0)

    critic_state = _fake_critic_state(apply_fn)
    target_q = jnp.array([[1.0], [2.0], [3.0]])
    q_var = jnp.array([0.0, 0.0, 0.0])  # below threshold
    out, frac = mc_correction_modify_target(
        target_q,
        critic_state,
        critic_params_mc={"mc": 1.0},
        observations=jnp.zeros((3, 2)),
        actions=jnp.zeros((3, 1)),
        q_var=q_var,
        mc_variance_threshold=0.5,
    )
    assert jnp.allclose(out, target_q)
    assert float(frac.squeeze()) == pytest.approx(0.0)


def test_mc_correction_replaces_high_variance_entries():
    def apply_fn(params, x):
        batch = jnp.ones((x.shape[0], 1)) * 9.0
        return jnp.stack([batch], axis=0)

    critic_state = _fake_critic_state(apply_fn)
    target_q = jnp.array([[1.0], [2.0], [3.0]])
    q_var = jnp.array([1.0, 1.0, 1.0])  # above threshold everywhere
    out, frac = mc_correction_modify_target(
        target_q,
        critic_state,
        critic_params_mc={"mc": 1.0},
        observations=jnp.zeros((3, 2)),
        actions=jnp.zeros((3, 1)),
        q_var=q_var,
        mc_variance_threshold=0.5,
    )
    # All entries replaced by q_mc_target = 9.0.
    assert jnp.allclose(out, jnp.full_like(target_q, 9.0))
    assert float(frac.squeeze()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Obs augmentation / detachment
# ---------------------------------------------------------------------------


def test_augment_obs_if_needed_no_op_when_disabled():
    obs = jnp.ones((2, 5))
    out = augment_obs_if_needed(
        obs, raw_observations=obs, expert_policy=None, augment=False
    )
    assert out is obs


def test_augment_obs_if_needed_no_op_when_expert_policy_is_none():
    obs = jnp.ones((2, 5))
    out = augment_obs_if_needed(
        obs, raw_observations=obs, expert_policy=None, augment=True
    )
    assert out is obs


def test_augment_obs_inserts_expert_action_before_train_frac():
    obs = jnp.array([[1.0, 2.0, 3.0, 0.5]])  # last dim = train_frac=0.5
    raw = obs

    def expert(x):
        return jnp.array([[10.0, 20.0]])

    out = augment_obs_if_needed(obs, raw, expert_policy=expert, augment=True)
    # Expected layout [env_obs (first 3), a_expert (2), train_frac (1)].
    assert out.shape == (1, 6)
    assert float(out[0, -1]) == pytest.approx(0.5)
    assert jnp.allclose(out[0, 3:5], jnp.array([10.0, 20.0]))


def test_detach_obs_expert_dims_is_identity_in_value():
    action_dim = 2
    obs = jnp.array([[1.0, 2.0, 3.0, 10.0, 20.0, 0.5]])
    out = detach_obs_expert_dims(obs, action_dim)
    assert jnp.allclose(out, obs)


def test_detach_obs_expert_dims_blocks_gradients_on_expert_slice():
    """Grad w.r.t. the expert-action slice must be zero."""
    action_dim = 2

    def loss(obs):
        augmented = detach_obs_expert_dims(obs, action_dim)
        return jnp.sum(augmented**2)

    obs = jnp.array([[1.0, 2.0, 3.0, 10.0, 20.0, 0.5]])
    grads = jax.grad(loss)(obs)
    # Gradient on the expert-action slice [-(action_dim+1):-1] must be zero.
    assert jnp.allclose(grads[..., -(action_dim + 1) : -1], 0.0)
    # Other slices still carry gradient.
    assert not jnp.allclose(grads[..., : -(action_dim + 1)], 0.0)


# ---------------------------------------------------------------------------
# Residual action transform
# ---------------------------------------------------------------------------


def test_residual_action_transform_clips_to_unit_box():
    actions = jnp.array([[0.5, -2.0, 0.9]])
    a_expert = jnp.array([[0.6, 0.1, 0.3]])
    out = residual_action_transform(actions, a_expert)
    assert out.shape == actions.shape
    assert float(out.max()) <= 1.0
    assert float(out.min()) >= -1.0


def test_residual_action_transform_sum_inside_box():
    actions = jnp.array([[0.1, -0.2]])
    a_expert = jnp.array([[0.3, 0.4]])
    out = residual_action_transform(actions, a_expert)
    assert jnp.allclose(out, jnp.array([[0.4, 0.2]]))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_compute_expert_diagnostics_shapes():
    def apply_fn(params, x):
        batch = jnp.ones((x.shape[0], 1)) * 0.5
        return jnp.stack([batch], axis=0)

    critic_state = _fake_critic_state(apply_fn)
    q_mean, l2_mean, above_frac = compute_expert_diagnostics(
        critic_state,
        observations=jnp.zeros((4, 3)),
        q_min=jnp.ones((4, 1)),  # q_min > q_expert=0.5 → above = 1.0
        a_expert=jnp.zeros((4, 2)),
        pi_loc=jnp.zeros((4, 2)),
    )
    assert q_mean.shape == ()
    assert l2_mean.shape == ()
    assert float(above_frac) == pytest.approx(1.0)


def test_compute_online_bc_loss_decays_after_warmup():
    def apply_fn(params, x):
        batch = jnp.ones((x.shape[0], 1)) * 0.8
        return jnp.stack([batch], axis=0)

    critic_state = _fake_critic_state(apply_fn)
    common = dict(
        pi_loc=jnp.zeros((3, 2)),
        a_expert=jnp.ones((3, 2)),
        critic_state=critic_state,
        expert_critic_params={"e": 0.0},
        observations=jnp.zeros((3, 4)),
        critic_warmup_frac=0.5,
        expert_v_min=jnp.asarray(0.0),
        expert_v_max=jnp.asarray(1.0),
        bc_coef=1.0,
    )
    loss_during = compute_online_bc_loss(train_frac=jnp.asarray(0.0), **common)
    loss_after = compute_online_bc_loss(train_frac=jnp.asarray(0.9), **common)
    assert float(loss_during) > 0.0
    assert float(loss_after) == pytest.approx(0.0)


def test_compute_behavior_kpis_altitude_error_and_z_dot():
    # raw_obs layout: [current, _, z_dot, target]
    raw = jnp.array([[1.0, 0.0, -0.3, 4.0]])
    alt_err, z_dot = compute_behavior_kpis(
        raw, altitude_obs_idx=0, target_obs_idx=3
    )
    assert float(alt_err) == pytest.approx(3.0)
    assert float(z_dot) == pytest.approx(0.3)
