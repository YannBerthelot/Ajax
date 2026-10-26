"""IBRL's and LCBGatedBootstrap's target as IBRL's authors compute it
(hengyuan-hu/ibrl 3c457da): the policy's next action is the one the agent's
target scored, and on a target over a random subset of its critics (REDQ's)
IBRL selects on a fresh subset of the same size and values on the target's."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import TD3
from ajax.extensions import target_mods
from ajax.extensions.base import ExtensionContext
from ajax.extensions.target_mods import (
    IBRL,
    LCBGatedBootstrap,
    draw_selection_critics,
)
from ajax.networks.networks import predict_value

KEY = jax.random.PRNGKey(0)
CONTEXT = ExtensionContext(step=jnp.asarray(0), rng=KEY, total_steps=100)
# Four critics, three rows: the expert's and the policy's next-action values.
Q_EXPERT = jnp.array([[1, 1, 1], [1, 1, 1], [3, 3, 0], [3, 3, 0.0]])[..., None]
Q_POLICY = jnp.array([[2, 0, 2], [2, 0, 2], [1, 1, 1], [1, 1, 1.0]])[..., None]


def half(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def test_ibrl_scores_the_next_actions_its_target_scored() -> None:
    """On TD3 the policy's next action is the target's smoothed one, handed
    in as next_actions, not a draw of the online actor: the bonus is
    gamma max(min Q(s', a_E) - min Q(s', a'), 0) at the a' handed in."""
    net = ("8", "relu")
    agent = TD3("Pendulum-v1", actor_architecture=net, critic_architecture=net)
    state = agent.init(KEY, agent.make_context(1)[1])
    next_observations = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
    next_actions = jnp.linspace(-1.0, 1.0, 6)[:, None]
    batch = {
        "next_observations": next_observations,
        "raw_next_observations": next_observations,
        "next_a_expert": None,
        "next_actions": next_actions,
        "critic_subset": None,
        "dones": jnp.zeros((6, 1)),
        "gamma": 0.9,
    }
    target, _ = IBRL(expert_policy=half).on_target(
        state, None, batch, jnp.zeros((6, 1)), CONTEXT
    )

    critic, target_params = state.critic_state, state.critic_state.target_params
    assert target_params is not None

    def value(actions: jax.Array) -> jax.Array:
        x = jnp.concatenate((next_observations, actions), axis=-1)
        return jnp.min(predict_value(critic, target_params, x), axis=0)

    advantage = value(half(next_observations)) - value(next_actions)
    want = 0.9 * jnp.maximum(advantage, 0.0)
    np.testing.assert_allclose(target, want, rtol=1e-6, atol=1e-7)


def test_ibrl_selects_on_a_fresh_subset_and_values_on_the_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handed REDQ's subset (critics 0, 1), IBRL selects on a fresh one (2, 3
    here) and values the selected action on the target's: row 0 selects the
    expert, worth 1 - 2 on the target's critics (a bonus below zero, as the
    authors' target can fall below the base one), row 1 selects it at 1 - 0,
    row 2 selects the policy (no bonus)."""
    monkeypatch.setattr(
        target_mods, "score_next_actions", lambda *a: (Q_EXPERT, Q_POLICY)
    )
    monkeypatch.setattr(
        target_mods, "draw_selection_critics", lambda *a: jnp.array([2, 3])
    )
    batch = {"dones": jnp.zeros((3, 1)), "gamma": 0.5}
    batch["critic_subset"] = jnp.array([0, 1])
    target, metrics = IBRL(expert_policy=abs).on_target(
        None, None, batch, jnp.zeros((3, 1)), CONTEXT
    )
    np.testing.assert_allclose(target, [[-0.5], [0.5], [0.0]])
    np.testing.assert_allclose(metrics["ibrl/expert_wins"], 2 / 3)


def test_the_selection_subset_is_a_fresh_one_of_the_targets_size() -> None:
    """None (every critic) for a target over all its critics; else distinct
    critics, as many as the target's subset, a new draw per key."""
    assert draw_selection_critics(None, 5, KEY) is None
    keys = jax.random.split(KEY, 50)
    subsets = [draw_selection_critics(jnp.array([0, 1]), 5, k) for k in keys]
    draws = [np.asarray(subset).tolist() for subset in subsets]
    assert all(len(set(d)) == 2 and set(d).issubset(range(5)) for d in draws)
    assert len({tuple(sorted(d)) for d in draws}) > 1


def test_lcb_blends_the_minima_over_the_targets_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The blended values are minima over the target's subset (critics 0, 1),
    the gate's LCB scores (beta 0: minima) over the whole ensemble:
    gamma p (min_01 Q_E - min_01 Q_pi), p = sigmoid(min Q_E - min Q_pi)."""
    monkeypatch.setattr(
        target_mods, "score_next_actions", lambda *a: (Q_EXPERT, Q_POLICY)
    )
    timestep = jnp.asarray(0)
    state = SimpleNamespace(collector_state=SimpleNamespace(timestep=timestep))
    batch = {"dones": jnp.zeros((3, 1)), "gamma": 0.5}
    batch["critic_subset"] = jnp.array([0, 1])
    lcb = LCBGatedBootstrap(expert_policy=abs, lcb_beta_init=0.0)
    target, _ = lcb.on_target(state, None, batch, jnp.zeros((3, 1)), CONTEXT)
    gate = jax.nn.sigmoid(jnp.min(Q_EXPERT, 0) - jnp.min(Q_POLICY, 0))
    want = 0.5 * gate * (jnp.min(Q_EXPERT[:2], 0) - jnp.min(Q_POLICY[:2], 0))
    np.testing.assert_allclose(target, want, rtol=1e-6)
