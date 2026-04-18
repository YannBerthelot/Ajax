"""Tests for ajax.modules.exploration — EDGE gates and value-box override."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from ajax.modules.exploration import (
    EDGEAuxiliaries,
    box_action_override,
    box_compute_state,
    box_compute_threshold,
    box_modify_reward_done,
    compute_edge_diagnostics,
    edge_argmax_gate,
    edge_boltzmann_gate,
    edge_compute_decay,
    edge_compute_value_gap,
    edge_fixed_gate,
)


def _fake_critic_state(fn):
    return SimpleNamespace(
        apply_fn=lambda params, x: fn(params, x),
        params={"online": 1.0},
        target_params={"target": 1.0},
    )


# ---------------------------------------------------------------------------
# EDGE: value gap
# ---------------------------------------------------------------------------


def test_edge_value_gap_positive_when_expert_is_better():
    def apply_fn(params, x):
        # Action concatenated after obs in last 2 dims. If action > 0, return
        # +1 (pretend expert preferred); otherwise return -1.
        q = jnp.where(x[..., -2:].sum(-1, keepdims=True) > 0, 1.0, -1.0)
        return jnp.stack([q], axis=0)

    critic_state = _fake_critic_state(apply_fn)
    obs = jnp.zeros((3, 4))
    policy_action = -jnp.ones((3, 2))
    expert_action = jnp.ones((3, 2))
    gap, q_policy = edge_compute_value_gap(
        obs, policy_action, expert_action, critic_state, critic_state.params
    )
    assert jnp.all(gap > 0)
    assert q_policy.shape == (3, 1)


# ---------------------------------------------------------------------------
# EDGE: decay schedule
# ---------------------------------------------------------------------------


def test_edge_decay_zero_frac_is_always_one():
    decay = edge_compute_decay(jnp.asarray(100), total_timesteps=1000, decay_frac=0.0)
    assert float(decay) == pytest.approx(1.0)


def test_edge_decay_linear_reaches_zero_at_decay_frac():
    decay_frac = 0.5
    decay_at_start = edge_compute_decay(jnp.asarray(0), 1000, decay_frac)
    decay_at_mid = edge_compute_decay(jnp.asarray(250), 1000, decay_frac)
    decay_at_end = edge_compute_decay(jnp.asarray(500), 1000, decay_frac)
    decay_past = edge_compute_decay(jnp.asarray(800), 1000, decay_frac)
    assert float(decay_at_start) == pytest.approx(1.0)
    assert float(decay_at_mid) == pytest.approx(0.5)
    assert float(decay_at_end) == pytest.approx(0.0)
    assert float(decay_past) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EDGE: gates
# ---------------------------------------------------------------------------


def test_edge_argmax_gate_strictly_positive_gap_and_decay():
    rng = jax.random.PRNGKey(0)
    gap = jnp.array([-1.0, 0.0, 0.5, 1.0])
    decay = jnp.asarray(0.5)
    mask, rng_out = edge_argmax_gate(gap, decay, rng)
    assert jnp.array_equal(mask, jnp.array([False, False, True, True]))
    # argmax gate does not consume the rng.
    assert jnp.array_equal(rng_out, rng)


def test_edge_argmax_gate_disabled_when_decay_zero():
    mask, _ = edge_argmax_gate(
        jnp.array([2.0, 3.0]), jnp.asarray(0.0), jax.random.PRNGKey(0)
    )
    assert jnp.array_equal(mask, jnp.array([False, False]))


def test_edge_boltzmann_gate_returns_bool_mask_and_fresh_rng():
    rng = jax.random.PRNGKey(0)
    gap = jnp.array([[2.0], [2.0], [2.0]])
    decay = jnp.asarray(1.0)
    q_policy = jnp.ones_like(gap)
    mask, rng_out = edge_boltzmann_gate(gap, decay, rng, q_policy, tau=1.0)
    assert mask.dtype == jnp.bool_
    assert mask.shape == gap.shape
    assert not jnp.array_equal(rng_out, rng)


def test_edge_fixed_gate_zero_prob_never_fires():
    rng = jax.random.PRNGKey(0)
    decay = jnp.ones((3, 1))
    mask, _ = edge_fixed_gate(jnp.zeros((3, 1)), decay, rng, fixed_prob=0.0)
    assert not jnp.any(mask)


def test_edge_fixed_gate_full_prob_always_fires():
    rng = jax.random.PRNGKey(0)
    decay = jnp.ones((3, 1))
    mask, _ = edge_fixed_gate(jnp.zeros((3, 1)), decay, rng, fixed_prob=1.0)
    assert jnp.all(mask)


# ---------------------------------------------------------------------------
# Box: threshold curriculum
# ---------------------------------------------------------------------------


def test_box_compute_threshold_interpolates_linearly():
    v_min = jnp.asarray(0.0)
    v_max = jnp.asarray(10.0)
    assert float(box_compute_threshold(v_min, v_max, jnp.asarray(0.0))) == 0.0
    assert float(box_compute_threshold(v_min, v_max, jnp.asarray(0.5))) == 5.0
    assert float(box_compute_threshold(v_min, v_max, jnp.asarray(1.0))) == 10.0


# ---------------------------------------------------------------------------
# Box: state, override, reward/done
# ---------------------------------------------------------------------------


def test_box_compute_state_initializes_last_in_box_when_none():
    def apply_fn(params, x):
        batch = jnp.ones((x.shape[0], 1)) * 5.0
        return jnp.stack([batch], axis=0)

    critic_state = _fake_critic_state(apply_fn)

    def expert(obs):
        return jnp.zeros((obs.shape[0], 2))

    in_box, entry_bonus, v_box = box_compute_state(
        obs=jnp.zeros((3, 4)),
        raw_obs=jnp.zeros((3, 4)),
        expert_policy=expert,
        critic_state=critic_state,
        expert_critic_params={"e": 1.0},
        threshold=jnp.asarray(1.0),
        last_in_box=None,
    )
    # All samples above threshold (v_box=5 > 1) → all in_box, and since
    # last_in_box starts at zero, every sample is a fresh entry → bonus = v_box.
    assert jnp.all(in_box)
    assert jnp.allclose(entry_bonus, v_box)


def test_box_compute_state_no_bonus_when_already_inside():
    def apply_fn(params, x):
        batch = jnp.ones((x.shape[0], 1)) * 5.0
        return jnp.stack([batch], axis=0)

    critic_state = _fake_critic_state(apply_fn)

    def expert(obs):
        return jnp.zeros((obs.shape[0], 2))

    _, entry_bonus, _ = box_compute_state(
        obs=jnp.zeros((2, 4)),
        raw_obs=jnp.zeros((2, 4)),
        expert_policy=expert,
        critic_state=critic_state,
        expert_critic_params={"e": 1.0},
        threshold=jnp.asarray(1.0),
        last_in_box=jnp.ones((2, 1)),  # already inside
    )
    assert jnp.allclose(entry_bonus, 0.0)


def test_box_action_override_picks_expert_inside_box():
    action = jnp.array([[0.1, 0.2]])
    expert = jnp.array([[0.9, -0.9]])
    out = box_action_override(action, expert, in_box=jnp.asarray(True))
    assert jnp.allclose(out, expert)


def test_box_action_override_passes_through_outside_box():
    action = jnp.array([[0.1, 0.2]])
    expert = jnp.array([[0.9, -0.9]])
    out = box_action_override(action, expert, in_box=jnp.asarray(False))
    assert jnp.allclose(out, action)


def test_box_modify_reward_done_adds_entry_bonus_and_sets_terminal():
    reward = jnp.array([0.5, 0.5])
    terminated = jnp.array([False, False])
    entry_bonus = jnp.array([[2.0], [0.0]])
    new_reward, new_term = box_modify_reward_done(reward, terminated, entry_bonus)
    assert jnp.allclose(new_reward, jnp.array([2.5, 0.5]))
    # First sample had a non-zero bonus so it must be marked terminal.
    assert bool(new_term[0])
    assert not bool(new_term[1])


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_compute_edge_diagnostics_returns_expected_aux_shape():
    gap = jnp.array([[1.0], [2.0]])
    q_pred = jnp.array([[3.0], [3.0]])
    aux = compute_edge_diagnostics(
        q_gap=gap,
        q_pred_min=q_pred,
        exploration_tau=1.0,
        expert_frac_in_buffer=jnp.asarray(0.25),
    )
    assert isinstance(aux, EDGEAuxiliaries)
    assert aux.value_gap.shape == (2,)
    assert aux.p_expert_mean.shape == (1,)
    assert aux.expert_action_fraction.shape == (1,)
    # With a positive gap, p_expert_mean > 0.5.
    assert float(aux.p_expert_mean.squeeze()) > 0.5
