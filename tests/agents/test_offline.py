"""Offline training (``ActorCritic.train_offline``, ``TrainLoop.offline``):
the agent's own updates on a replay holding a dataset, with no environment
step, evaluated every ``log_frequency`` updates, then online fine-tuning
from that state. Whether it learns is the probes' (tests/probing/test_offline)."""

from typing import Any

import numpy as np
import pytest

from ajax import PPO
from ajax.buffers.dataset import TransitionDataset
from ajax.logging.wandb_logging import LoggingConfig
from tests.agents.test_agent_contract import CASES

AGENTS = ("SAC", "TD3", "REDQ", "ASAC", "DQN")
ROWS, UPDATES, EVERY = 16, 6, 3


def episodes(obs_dim: int, discrete: bool) -> TransitionDataset:
    """``ROWS`` one-step episodes on random observations."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(ROWS, obs_dim))
    action = (
        rng.integers(0, 2, (ROWS, 1)) if discrete else rng.uniform(-1, 1, (ROWS, 1))
    )
    ends = np.ones(ROWS, bool)
    return TransitionDataset(obs, action, rng.normal(size=ROWS), ends, ~ends, obs)


def build(name: str) -> tuple[Any, TransitionDataset]:
    case = CASES[name]
    env = case.envs[-1]
    agent = case.cls(env_id=env, n_envs=1, **(case.kwargs | {"learning_starts": 0}))
    obs_dim = 4 if env == "CartPole-v1" else 3
    return agent, episodes(obs_dim, discrete=env == "CartPole-v1")


def rows(state: Any) -> int:
    return int(np.asarray(state.collector_state.buffer_state.current_index)[0])


def steps(state: Any) -> int:
    """Optimiser steps of the actor and the critic (DQN's Q-network is its
    actor_state), seed 0."""
    nets = (state.actor_state, state.critic_state)
    return sum(int(np.asarray(n.step).reshape(-1)[0]) for n in nets if n is not None)


@pytest.mark.parametrize("name", AGENTS)
def test_an_agent_trains_offline_then_fine_tunes_online(name: str) -> None:
    """The replay holds the dataset's rows and nothing collected; the
    evaluations come every EVERY updates, logging the updates done; the
    state comes back at timestep 0, so an online call counts its own steps
    from there and adds its rows after the dataset's."""
    agent, data = build(name)
    logging = LoggingConfig(config={}, log_frequency=EVERY, use_wandb=False)
    state, out = agent.train_offline(data, UPDATES, seed=[0], logging_config=logging)
    np.testing.assert_array_equal(out["timestep"], [[EVERY, 2 * EVERY]])
    assert rows(state) == ROWS
    np.testing.assert_array_equal(state.collector_state.timestep, 0)
    np.testing.assert_array_equal(state.n_logs, 0)
    updated = steps(state)
    assert updated > 0

    online, out = agent.train(
        seed=[0], n_timesteps=EVERY, initial_state=state, logging_config=logging
    )
    np.testing.assert_array_equal(out["timestep"], [[EVERY]])
    assert rows(online) == ROWS + EVERY
    assert steps(online) > updated


def test_an_agent_without_a_transition_replay_refuses_offline_training() -> None:
    with pytest.raises(ValueError, match="cannot train offline"):
        PPO("Pendulum-v1").train_offline(episodes(3, False), UPDATES)
