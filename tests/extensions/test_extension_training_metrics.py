"""What the target extensions and OnlineBC log beside their value, on a
fixed batch of a fresh SAC: each metric read against the value the phase
returns (the critic as its own expert critic phi*)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.SAC.SAC import SAC
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import ExtensionContext
from ajax.extensions.expert import OnlineBC, online_bc_weight
from ajax.extensions.target_mods import (
    IBRL,
    CriticBlend,
    LCBGatedBootstrap,
    MCVarianceCorrection,
)

KEY = jax.random.PRNGKey(0)
BATCH = 8


def expert(obs: jax.Array) -> jax.Array:
    return jnp.clip(-obs[..., -1:] / 8.0, -1.0, 1.0)


def _setup() -> tuple[Any, dict, jax.Array]:
    """A fresh SAC state with phi* = its critic, the batch SAC's critic step
    hands ``on_target`` and a target."""
    agent = SAC("Pendulum-v1", actor_architecture=("16", "relu"))
    state = agent.init(KEY, agent.make_context(1)[1])
    critic = state.critic_state
    state = state.replace(
        expert_critic_params=critic.params,
        expert_v_min=jnp.asarray(-1.0),
        expert_v_max=jnp.asarray(1.0),
    )
    obs_shape, act_shape = get_state_action_shapes(agent.env_args.env)
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    obs = jax.random.normal(keys[0], (BATCH, *obs_shape))
    actions = jnp.tanh(jax.random.normal(keys[1], (BATCH, *act_shape)))
    next_obs = jax.random.normal(keys[2], (BATCH, *obs_shape))
    next_actions = jnp.tanh(jax.random.normal(keys[4], (BATCH, *act_shape)))
    batch = {
        "observations": obs,
        "actions": actions,
        "next_observations": next_obs,
        "rewards": jax.random.normal(keys[3], (BATCH, 1)),
        "reward_scale": 1.0,
        "dones": jnp.zeros((BATCH, 1)),
        "gamma": 0.99,
        "raw_next_observations": next_obs,
        "next_a_expert": None,
        "next_actions": next_actions,
        "critic_subset": None,
    }
    return state, batch, jnp.zeros((BATCH, 1))


def _on_target(ext: Any, step: int = 0) -> tuple[jax.Array, jax.Array, dict]:
    state, batch, target = _setup()
    context = ExtensionContext(step=jnp.asarray(step), rng=KEY, total_steps=100)
    state = state.replace(
        collector_state=state.collector_state.replace(timestep=jnp.asarray(step))
    )
    new, metrics = ext.on_target(state, (), batch, target, context)
    return target, new, metrics


def test_ibrl_logs_its_mean_bonus_and_where_the_expert_wins() -> None:
    target, new, metrics = _on_target(IBRL(expert_policy=expert))
    np.testing.assert_allclose(metrics["ibrl/bonus"], jnp.mean(new - target), rtol=1e-6)
    wins = float(metrics["ibrl/expert_wins"])
    assert 0.0 <= wins <= 1.0 and (wins > 0) == (float(metrics["ibrl/bonus"]) > 0)


def test_lcb_logs_the_mean_gate() -> None:
    _, _, metrics = _on_target(LCBGatedBootstrap(expert_policy=expert))
    assert 0.0 < float(metrics["lcb_bootstrap/p_expert"]) < 1.0


def test_critic_blend_logs_the_gap_its_weight_closes() -> None:
    """At step 0 the weight is 1: the target becomes the expert's, so the
    gap is the mean change; half way through warmup, half of it."""
    blend = CriticBlend(expert_policy=expert, critic_warmup_frac=0.5)
    target, new, metrics = _on_target(blend)
    gap = metrics["critic_blend/target_gap"]
    np.testing.assert_allclose(gap, jnp.mean(new - target), rtol=1e-5)
    target, new, metrics = _on_target(blend, step=25)
    np.testing.assert_allclose(jnp.mean(new - target), 0.5 * gap, rtol=1e-5)


def test_mc_variance_logs_the_share_of_targets_it_replaces() -> None:
    target, new, metrics = _on_target(MCVarianceCorrection(threshold=-1.0))
    assert float(metrics["mc_variance/replaced_frac"]) == 1.0
    assert not np.allclose(new, target)
    target, new, metrics = _on_target(MCVarianceCorrection(threshold=1e9))
    assert float(metrics["mc_variance/replaced_frac"]) == 0.0
    np.testing.assert_array_equal(new, target)


def test_online_bc_logs_its_term_and_schedule(critic_stub: Any) -> None:
    critic_state = critic_stub(lambda x: jnp.full((x.shape[0], 1), 0.8))
    state = SimpleNamespace(
        critic_state=critic_state,
        expert_critic_params={"e": 0.0},
        expert_v_min=jnp.asarray(0.0),
        expert_v_max=jnp.asarray(1.0),
    )
    batch = {
        "pi_mean": jnp.zeros((3, 2)),
        "a_expert": jnp.ones((3, 2)),
        "observations": jnp.zeros((3, 4)),
    }
    context = ExtensionContext(step=jnp.asarray(25), rng=KEY, total_steps=100)
    term, metrics = OnlineBC(critic_warmup_frac=0.5).actor_loss(
        state, (), batch, context
    )
    assert metrics["online_bc/term"] is term and float(term) > 0.0
    np.testing.assert_allclose(metrics["online_bc/weight"], 0.5, rtol=1e-6)
    assert float(online_bc_weight(jnp.asarray(0.6), 0.5)) == 0.0
