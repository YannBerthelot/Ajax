"""The action stage's reach: an action extension runs on every off-policy
agent of the shared collector with an expert. JSRL, its expert acting on
every step here, has each agent execute and store the expert's action (in
the agent's own layout: DAgger's discrete index column), flagged as the
expert's; AVG, which keeps no replay, learns from the step's executed
action. IQL under a residual policy refuses it (that the agents learn is
tests/probing's)."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from ajax import IQL
from ajax.extensions.capabilities import NotWired
from ajax.extensions.expert import JSRLCurriculum, ResidualPolicy
from tests.agents.test_agent_contract import CASES

#: The expert acts on every step: a horizon far beyond any episode, which
#: shrinks over a fraction of training far beyond the run.
ALWAYS = JSRLCurriculum(episode_length=10_000, decay_frac=1e6)


def _one(obs: jnp.ndarray) -> jnp.ndarray:
    """1 in Pendulum's units, half its bound: 0.5 in the agent's."""
    return jnp.ones(obs.shape[:-1] + (1,))


def _first(obs: jnp.ndarray) -> jnp.ndarray:
    """CartPole's action 1, as an (n, 1) int column."""
    return jnp.ones(obs.shape[:-1] + (1,), jnp.int32)


def _run(name: str, env_id: str, expert: Any, **extra: Any) -> tuple:
    case = CASES[name]
    kwargs = dict(case.kwargs)
    kwargs.update({"expert_policy": expert, "extensions": (ALWAYS,)})
    kwargs.update(extra)
    agent = case.cls(env_id=env_id, n_envs=2, **kwargs)
    return agent.train(seed=0, n_timesteps=case.n_timesteps)[0], case


@pytest.mark.parametrize(
    "name", ["TD3", "REDQ", "TQC", "CrossQ", "ASAC", "IQL", "DAgger"]
)
def test_every_row_past_the_warm_up_stores_the_experts_action(name: str) -> None:
    """Past the warm-up (uniform actions, which the pre-warm-up slot leaves:
    TQC and DAgger have none) every row stores the expert's action, flagged
    as the expert's where the replay keeps the flags (IQL stores no expert
    data)."""
    state, case = _run(name, "Pendulum-v1", _one)
    rows = state.collector_state.buffer_state.experience
    start = case.kwargs.get("learning_starts", 0) // 2
    if name in ("TQC", "DAgger"):
        start = 0
    stop = case.n_timesteps // 2  # rows per env
    np.testing.assert_allclose(np.asarray(rows["action"])[..., start:stop, 0], 0.5)
    if name != "IQL":
        flags = np.asarray(rows["is_expert"])[..., start:stop, 0]
        np.testing.assert_array_equal(flags, 1)


def test_a_discrete_agent_stores_the_experts_index_in_its_column() -> None:
    """DAgger on CartPole stores its own (n, 1) int column: the stage writes
    the expert's index there, not the actor's, also once DAgger's own mix
    hands the actor every step (beta 0 from its second iteration)."""
    state, case = _run("DAgger", "CartPole-v1", _first, beta_decay=0.0)
    rows = state.collector_state.buffer_state.experience
    stop = case.n_timesteps // 2
    np.testing.assert_array_equal(np.asarray(rows["action"])[..., :stop, 0], 1)
    np.testing.assert_array_equal(np.asarray(rows["is_expert"])[..., :stop, 0], 1)


def test_avg_learns_from_the_action_its_step_executed() -> None:
    """AVG's last step executed the expert's action, which its critic reads
    (executed_action); the transition's action stays the policy's."""
    state, _ = _run("AVG", "Pendulum-v1", _one, expose_recent_rollout=True)
    step = state.last_rollout
    np.testing.assert_allclose(np.asarray(step.executed_action), 0.5)
    assert not np.allclose(np.asarray(step.action), 0.5)
    np.testing.assert_array_equal(np.asarray(state.collector_state.last_expert_frac), 1)


def test_iql_under_a_residual_policy_refuses_an_action_extension() -> None:
    """Its replay keeps the residual the actor chose beside the executed
    action: a substituted action leaves none to keep."""
    IQL("Pendulum-v1", expert_policy=_one, extensions=(ALWAYS,))
    with pytest.raises(NotWired, match="residual"):
        IQL("Pendulum-v1", expert_policy=_one, extensions=(ALWAYS, ResidualPolicy()))
