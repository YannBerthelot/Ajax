"""The expert-guidance maths: the expert extensions' and target modifiers'
helpers and SAC's expert plumbing."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.extensions import target_mods
from ajax.extensions.base import ExtensionContext
from ajax.extensions.expert import (
    OnlineBC,
    compute_online_bc_loss,
    residual_action_transform,
)
from ajax.extensions.target_mods import (
    IBRL,
    blend_modify_target,
    mc_correction_modify_target,
)

CTX = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0), total_steps=100)

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


def test_mc_correction_preserves_low_variance_entries(critic_stub):
    critic_state = critic_stub(lambda x: jnp.full((x.shape[0], 1), 9.0))
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


def test_mc_correction_replaces_high_variance_entries(critic_stub):
    critic_state = critic_stub(lambda x: jnp.full((x.shape[0], 1), 9.0))
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


def test_compute_online_bc_loss_decays_after_warmup(critic_stub):
    critic_state = critic_stub(lambda x: jnp.full((x.shape[0], 1), 0.8))
    common = {
        "pi_mean": jnp.zeros((3, 2)),
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


def test_online_bc_weighs_each_row_by_its_place_in_the_experts_value_range(
    critic_stub,
):
    """OnlineBC scales each row's cloning error by clip((V*(s, a_e) - v_min)
    / (v_max - v_min), 0, 1)^2 on the expert's value range that MCPretrain
    measured: V* at 1, 3, 5 and 7 on the range [1, 5] weighs 0, 1/4, 1 and 1,
    so unit errors cost their mean, 9/16, at the start of training."""
    obs = jnp.zeros((4, 3)).at[:, 0].set(jnp.array([1.0, 3.0, 5.0, 7.0]))
    agent_state = SimpleNamespace(
        critic_state=critic_stub(lambda x: x[:, :1]),
        expert_critic_params={"e": 0.0},
        expert_v_min=jnp.asarray(1.0),
        expert_v_max=jnp.asarray(5.0),
    )
    batch = {"a_expert": jnp.zeros((4, 1)), "pi_mean": jnp.ones((4, 1))}
    bc = OnlineBC(bc_coef=1.0, critic_warmup_frac=0.5)
    term, _ = bc.actor_loss(agent_state, None, {**batch, "observations": obs}, CTX)
    np.testing.assert_allclose(term, 9 / 16, rtol=1e-5)


def test_ibrl_adds_the_discounted_expert_advantage_where_the_episode_goes_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IBRL's bonus is gamma (1 - d) max(min Q(s', a_expert') - min Q(s', a_pi'),
    0) (Hu et al., 2023): with the next state's values at 3 for the expert
    and 1 for the policy (two critics) and gamma 0.5, it is 1 on rows that go
    on and 0 on rows that end; a policy ahead of the expert adds nothing."""
    expert = jnp.full((2, 4, 1), 3.0)
    policy = jnp.full((2, 4, 1), 1.0).at[:, 3].set(5.0)
    monkeypatch.setattr(target_mods, "_next_target_q", lambda *a: (expert, policy))
    batch = {"dones": jnp.array([[0.0], [1.0], [0.0], [0.0]]), "gamma": 0.5}
    target, metrics = IBRL(expert_policy=abs).on_target(
        None, None, batch, jnp.zeros((4, 1)), CTX
    )
    np.testing.assert_allclose(target, [[1.0], [0.0], [1.0], [0.0]])
    np.testing.assert_allclose(metrics["ibrl/bonus"], 0.5)
