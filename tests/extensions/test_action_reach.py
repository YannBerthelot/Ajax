"""The action stage's reach: an action extension runs on every off-policy
agent of the shared collector with an expert. JSRL, its expert acting on
every step here, has each agent execute and store the expert's action (in
the agent's own layout: DAgger's discrete index column), flagged as the
expert's; AVG, which keeps no replay, learns from the step's executed
action. IQL under a residual policy keeps the residual executing the
expert's action, 0, and refuses an extension acting with another expert
(that the agents learn is tests/probing's)."""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from ajax import IQL
from ajax.extensions.expert import JSRLCurriculum, ResidualPolicy
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.target_mods import IBRL
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


@pytest.mark.parametrize(
    "acting",
    [
        JSRLCurriculum(episode_length=16, decay_frac=1e6),
        EDGEExploration(gate="fixed", fixed_prob=0.5, decay_frac=0.0),
        IBRL(expert_policy=_one),
    ],
    ids=["jsrl", "edge", "ibrl"],
)
def test_iql_under_a_residual_keeps_the_residual_each_row_executed(acting) -> None:
    """IQL's replay keeps, beside each executed action, the residual its
    actor fits: where the extension handed the agent's expert the step, 0,
    which executes the expert's action (0.5); the residual chosen elsewhere.
    Every row executed clip(a_E + residual)."""
    state, case = _run(
        "IQL", "Pendulum-v1", _one, extensions=(acting, ResidualPolicy())
    )
    rows = state.collector_state.buffer_state.experience
    stop = case.n_timesteps // 2  # rows per env
    executed = np.asarray(rows["action"])[..., :stop, 0]
    kept = np.asarray(rows["policy_action"])[..., :stop, 0]
    np.testing.assert_allclose(executed, np.clip(0.5 + kept, -1, 1), atol=1e-6)
    handed = kept == 0.0
    assert handed.any() and not handed.all()
    np.testing.assert_array_equal(executed[handed], 0.5)


def test_iql_under_a_residual_refuses_an_extension_acting_with_another_expert():
    """A residual reaches only clip(a_E + residual): another expert's action
    need not lie in reach, so IQL's replay would have no residual to keep."""
    IQL("Pendulum-v1", expert_policy=_one, extensions=(IBRL(_one), ResidualPolicy()))
    with pytest.raises(ValueError, match="other than the agent's"):
        IQL(
            "Pendulum-v1",
            expert_policy=_one,
            extensions=(IBRL(_first), ResidualPolicy()),
        )


@pytest.mark.parametrize("gate", ["fixed", "argmax", "lcb", "thompson"])
@pytest.mark.parametrize(
    "name",
    ["SAC", "TD3", "REDQ", "TQC", "CrossQ", "ASAC", "AVG", "IQL", "DAgger", "UDRL"],
)
def test_edge_runs_its_gates_on_every_continuous_agent(name: str, gate: str) -> None:
    """Each gate scores the agent's live critic (an ensemble's for the LCB
    and Thompson gates) and hands the expert some steps of a short run."""
    case = CASES[name]
    caps = case.cls.capabilities
    if gate != "fixed" and "q_critic" not in caps:
        pytest.skip(f"{name} has no Q critic for the value gates")
    num_critics = case.kwargs.get("num_critics", getattr(case.cls, "num_critics", 2))
    if gate in ("lcb", "thompson") and (
        "critic_ensemble" not in caps or num_critics < 2
    ):
        pytest.skip(f"{name} runs no critic ensemble for the {gate} gate")
    edge = EDGEExploration(gate=gate, fixed_prob=1.0, decay_frac=0.0)
    state, _ = _run(name, "Pendulum-v1", _one, extensions=(edge,))
    assert np.isfinite(np.asarray(state.collector_state.last_expert_frac)).all()
