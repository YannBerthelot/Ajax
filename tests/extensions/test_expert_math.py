"""The expert-guidance maths: the expert extensions' and target modifiers'
helpers and SAC's expert plumbing."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.environments.interaction import augment_obs
from ajax.extensions.expert import (
    compute_online_bc_loss,
    residual_action_transform,
)
from ajax.extensions.target_mods import blend_modify_target, mc_correction_modify_target


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
# Blend target modifier
# ---------------------------------------------------------------------------


def test_blend_modify_target_matches_convex_combination():
    target_q = jnp.array([[1.0], [2.0], [4.0]])
    v_expert = jnp.array([[5.0], [5.0], [5.0]])
    alpha = jnp.array([[0.25]])
    out = blend_modify_target(target_q, v_expert, alpha)
    expected = 0.75 * target_q + 0.25 * v_expert
    assert jnp.allclose(out, expected)


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
    out = mc_correction_modify_target(
        target_q,
        critic_state,
        critic_params_mc={"mc": 1.0},
        observations=jnp.zeros((3, 2)),
        actions=jnp.zeros((3, 1)),
        q_var=q_var,
        mc_variance_threshold=0.5,
    )
    assert jnp.allclose(out, target_q)


def test_mc_correction_replaces_high_variance_entries():
    def apply_fn(params, x):
        batch = jnp.ones((x.shape[0], 1)) * 9.0
        return jnp.stack([batch], axis=0)

    critic_state = _fake_critic_state(apply_fn)
    target_q = jnp.array([[1.0], [2.0], [3.0]])
    q_var = jnp.array([1.0, 1.0, 1.0])  # above threshold everywhere
    out = mc_correction_modify_target(
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


# ---------------------------------------------------------------------------
# Obs augmentation / detachment
# ---------------------------------------------------------------------------


def test_augment_obs_appends_the_expert_state_then_its_action():
    obs = jnp.array([[1.0, 2.0, 3.0, 0.5]])  # the training fraction last
    state, action = jnp.array([[7.0]]), jnp.array([[10.0, 20.0]])
    np.testing.assert_array_equal(augment_obs(obs), obs)
    np.testing.assert_array_equal(
        augment_obs(obs, expert_action=action)[0, 4:], [10, 20]
    )
    full = augment_obs(obs, state, action)
    np.testing.assert_array_equal(full, [[1.0, 2.0, 3.0, 0.5, 7.0, 10.0, 20.0]])


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


def test_compute_online_bc_loss_decays_after_warmup():
    def apply_fn(params, x):
        batch = jnp.ones((x.shape[0], 1)) * 0.8
        return jnp.stack([batch], axis=0)

    critic_state = _fake_critic_state(apply_fn)
    common = {
        "pi_loc": jnp.zeros((3, 2)),
        "a_expert": jnp.ones((3, 2)),
        "critic_state": critic_state,
        "expert_critic_params": {"e": 0.0},
        "observations": jnp.zeros((3, 4)),
        "critic_warmup_frac": 0.5,
        "expert_v_min": jnp.asarray(0.0),
        "expert_v_max": jnp.asarray(1.0),
        "bc_coef": 1.0,
    }
    loss_during = compute_online_bc_loss(train_frac=jnp.asarray(0.0), **common)
    loss_after = compute_online_bc_loss(train_frac=jnp.asarray(0.9), **common)
    assert float(loss_during) > 0.0
    assert float(loss_after) == pytest.approx(0.0)
