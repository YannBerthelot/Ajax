"""Residual RL (ResidualPolicy) on the agents whose critics score executed
actions: collection executes ``clip(a_expert + scale·a_pi)`` and stores it,
the critics and the actor score it (that the agents learn the residual is
tests/probing's)."""

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import APG, UDRL
from ajax.agents.recurrent import q_values
from ajax.agents.SAC.SAC import compute_soft_policy_loss, sample_next_actions
from ajax.agents.TD3.TD3 import compute_td3_td_target
from ajax.controller import make_scored_actions
from ajax.extensions.expert import ResidualPolicy
from tests.agents.test_agent_contract import CASES
from tests.environments import paid_action

#: The replay agents a residual policy runs on (AVG keeps no replay).
REPLAY = ("SAC", "TD3", "REDQ", "TQC", "CrossQ", "ASAC")


def _one(obs: jax.Array) -> jax.Array:
    """1 in Pendulum's units, half its bound: 0.5 in the agent's."""
    return jnp.ones(obs.shape[:-1] + (1,))


def _residual_run(name: str, n_timesteps: int, **extra: Any) -> tuple:
    case = CASES[name]
    cls: Any = case.cls  # each agent's own keywords
    agent = cls(
        env_id="Pendulum-v1",
        n_envs=2,
        expert_policy=_one,
        extensions=(ResidualPolicy(scale=0.0),),
        **case.kwargs,
        **extra,
    )
    return agent.train(seed=0, n_timesteps=n_timesteps)[0], case


@pytest.mark.parametrize("name", REPLAY)
def test_at_scale_zero_the_replay_stores_the_experts_executed_action(name):
    """With scale 0 the executed action is the expert's: every row past the
    warm-up (uniform executed actions; TQC has none, as its authors' code)
    stores it, as its a_expert."""
    steps = CASES[name].n_timesteps
    state, case = _residual_run(name, steps)
    rows = state.collector_state.buffer_state.experience
    start = 0 if name == "TQC" else case.kwargs["learning_starts"] // 2
    stop = steps // 2  # rows per env
    action = np.asarray(rows["action"])[..., start:stop, 0]
    np.testing.assert_allclose(action, 0.5, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(rows["a_expert"])[..., start:stop, 0], 0.5)
    if start:
        assert not np.allclose(np.asarray(rows["action"])[..., :start, 0], 0.5)


def test_td3s_actor_scores_the_executed_action():
    """With scale 0 the executed action is the expert's whatever the actor's:
    TD3's actor loss, -Q1(s, executed), has no gradient, and Adam leaves the
    actor where it started."""
    trained, case = _residual_run("TD3", CASES["TD3"].n_timesteps)
    untrained, _ = _residual_run("TD3", 0)
    assert int(np.asarray(trained.actor_state.step).max()) > 0  # it stepped
    jax.tree.map(
        np.testing.assert_array_equal,
        trained.actor_state.params,
        untrained.actor_state.params,
    )


@pytest.mark.parametrize("name", ["PPO", "APO"])
def test_an_on_policy_agent_executes_the_residual_and_keeps_its_own_sample(name):
    """With scale 0 Pendulum receives the expert's torque (1, its last_u),
    while the rollout keeps the policy's own samples, which the ratio reads,
    and the expert's actions beside them."""
    steps = CASES[name].n_timesteps
    state, _ = _residual_run(name, steps, expose_recent_rollout=True)
    last_u = np.asarray(state.collector_state.env_state.last_u)
    np.testing.assert_allclose(last_u, 1.0, rtol=1e-6)
    rollout = state.last_rollout
    np.testing.assert_allclose(np.asarray(rollout.a_expert), 0.5, rtol=1e-6)
    assert not np.allclose(np.asarray(rollout.action), 0.5)


def test_iql_stores_the_executed_action_and_the_residual_it_fits():
    """IQL's actor fits the actions it chose: its replay keeps them
    (policy_action, uniform residuals during the warm-up) beside the
    executed ones the critics score, the expert's at scale 0."""
    steps = CASES["IQL"].n_timesteps
    state, _ = _residual_run("IQL", steps)
    rows = state.collector_state.buffer_state.experience
    filled = slice(0, steps // 2)
    np.testing.assert_allclose(np.asarray(rows["action"])[..., filled, 0], 0.5)
    assert not np.allclose(np.asarray(rows["policy_action"])[..., filled, 0], 0.5)


def test_udrl_executes_the_residual_and_keeps_its_own_action():
    """UDRL acts through the residual in its own collector: with scale 0
    Pendulum receives the expert's torque, its segments keep the policy's
    actions, which its actor fits."""
    agent = UDRL(
        "Pendulum-v1",
        n_envs=2,
        n_steps=8,
        batch_size=8,
        n_updates_per_iter=1,
        buffer_capacity=8,
        actor_architecture=("16", "relu"),
        expert_policy=_one,
        extensions=(ResidualPolicy(scale=0.0),),
    )
    state, _ = agent.train(seed=0, n_timesteps=32, num_episode_test=1)
    last_u = np.asarray(state.collector_state.env_state.last_u)
    np.testing.assert_allclose(last_u, 1.0, rtol=1e-6)
    assert not np.allclose(np.asarray(state.buffer.actions), 0.5)


def test_apg_differentiates_through_the_residual():
    """APG's rollout runs through the residual: at scale 0 the executed
    action is the expert's whatever the controller's, so the return has no
    gradient and the controller stays where it started; at scale 1 it
    learns."""
    env, params = paid_action.make()

    def half(obs: jax.Array) -> jax.Array:
        return jnp.full(obs.shape[:-1] + (1,), 0.5)

    def run(scale: float, n_timesteps: int) -> Any:
        agent = APG(
            env,
            env_params=params,
            n_envs=2,
            horizon=4,
            actor_architecture=("8", "tanh"),
            pid=None,
            expert_policy=half,
            extensions=(ResidualPolicy(scale=scale),),
        )
        return agent.train(seed=0, n_timesteps=n_timesteps)[0].actor_state.params

    start = run(0.0, 0)
    jax.tree.map(np.testing.assert_array_equal, run(0.0, 32), start)
    moved = jax.tree.map(lambda a, b: bool((a != b).any()), run(1.0, 32), start)
    assert any(jax.tree.leaves(moved))


def _initial(name: str) -> Any:
    """The agent's untrained state on Pendulum, its seed axis dropped."""
    case = CASES[name]
    agent = case.cls(env_id="Pendulum-v1", n_envs=2, **case.kwargs)
    return jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])


def _put(a: jax.Array, a_expert: Any) -> jax.Array:
    """An executed-action function ignoring the policy: a_expert alone."""
    return jnp.zeros_like(a) + a_expert


def test_the_soft_core_scores_executed_actions():
    """SAC's equations (REDQ, TQC, ASAC, CrossQ, AVG read them): the
    bootstrap's next actions and the actor's objective, at the scored
    actions make_scored_actions gives, are the executed ones."""
    state = _initial("REDQ")
    obs, key = jnp.ones((4, 3)), jax.random.PRNGKey(0)
    a_expert = jnp.full((4, 1), 0.25)
    scored = make_scored_actions(
        _put, SimpleNamespace(a_expert=a_expert, next_a_expert=-a_expert)
    )
    next_actions, _ = sample_next_actions(
        state.actor_state, obs, key, map_next_actions=scored.map_next_actions
    )
    np.testing.assert_array_equal(next_actions, -0.25)
    actor, critic = state.actor_state, state.critic_state
    alpha = jnp.asarray(1.0)
    _, (_, q, _) = compute_soft_policy_loss(
        actor.params, actor, critic, obs, alpha, key, None, jnp.min, scored.map_actions
    )
    at_expert = jnp.min(q_values(critic, critic.params, obs, a_expert), axis=0)
    np.testing.assert_allclose(q, at_expert, rtol=1e-6)


def test_td3s_target_scores_the_executed_next_action():
    state = _initial("TD3")
    obs, zeros = jnp.ones((4, 3)), jnp.zeros((4, 1))
    rows = SimpleNamespace(a_expert=zeros, next_a_expert=jnp.full((4, 1), 0.25))
    _, _, next_actions = compute_td3_td_target(
        state.actor_state,
        state.critic_state,
        jax.random.PRNGKey(0),
        obs,
        zeros,
        zeros,
        0.99,
        0.2,
        0.5,
        1.0,
        map_next_actions=make_scored_actions(_put, rows).map_next_actions,
    )
    np.testing.assert_array_equal(next_actions, 0.25)
