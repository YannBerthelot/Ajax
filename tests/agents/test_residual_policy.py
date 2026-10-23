"""Residual RL (ResidualPolicy) on the agents whose critics score executed
actions: collection executes ``clip(a_expert + scale·a_pi)`` and stores it,
the critics and the actor score it (that the agents learn the residual is
tests/probing's)."""

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.extensions.expert import ResidualPolicy
from tests.agents.test_agent_contract import CASES

#: The replay agents a residual policy runs on (AVG keeps no replay).
REPLAY = ("SAC", "TD3", "REDQ", "TQC", "CrossQ", "ASAC")


def _one(obs: jax.Array) -> jax.Array:
    """1 in Pendulum's units, half its bound: 0.5 in the agent's."""
    return jnp.ones(obs.shape[:-1] + (1,))


def _residual_run(name: str, n_timesteps: int) -> tuple:
    case = CASES[name]
    cls: Any = case.cls  # each agent's own keywords
    # No expert prefill (SAC's, TQC's): the replay holds collection's rows.
    takes = inspect.signature(cls).parameters
    prefill = {"expert_buffer_n_steps": 0} if "expert_buffer_n_steps" in takes else {}
    agent = cls(
        env_id="Pendulum-v1",
        n_envs=2,
        expert_policy=_one,
        extensions=(ResidualPolicy(scale=0.0),),
        **case.kwargs,
        **prefill,
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
