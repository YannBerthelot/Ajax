"""The off-policy agents with memory replay Ni et al. (2022)'s windows.

Contract (owner, 2026-11-13: Ajax's windows, at Ni et al.'s sizes;
twni2016/pomdp-baselines @e7c19c3):

- 64 windows a batch, of 64 trained steps (configs/pomdp/*/rnn.yml:19-20),
  each from a zero memory with no burn-in (policies/models/policy_rnn.py:
  138, recurrent_critic.py:138); batch_size counts rows without memory
  (256, the agents' references').
- A window's first row reads as an episode's first: no previous action and
  no previous reward (Ni et al.'s a[0] = r[0] = 0, policy_rnn.py:247-257),
  whatever the stored observation held; its later rows read them. A window
  starting from a stored memory (R2D2's stored state) keeps them, as that
  memory read them.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from ajax import ASAC, REDQ, SAC, TD3
from ajax.agents.recurrent import sample_and_burnin_sequences
from ajax.environments.create import build_env_from_id
from ajax.networks.memory import MemoryConfig

GRU = MemoryConfig("gru", 8)
PREVIOUS_STEP = slice(3, 5)  # Pendulum's 3 columns, then the action and reward


@pytest.mark.parametrize("cls", [SAC, TD3, REDQ, ASAC])
def test_a_run_with_memory_replays_ni_et_als_windows(cls):
    agent = cls("Pendulum-v1", memory=GRU)
    assert (agent.batch_size, agent.sequence_length, agent.burn_in) == (64, 64, 0)
    assert cls("Pendulum-v1").batch_size == 256
    assert cls("Pendulum-v1", memory=GRU, batch_size=8).batch_size == 8


def _windows(stored_state: bool) -> Any:
    env, params = build_env_from_id("Pendulum-v1")
    assert params is not None
    agent = SAC(
        env_id=env,
        env_params=params.replace(max_steps_in_episode=50),
        n_envs=2,
        memory=GRU,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=400,
        batch_size=16,
        sequence_length=8,
        stored_state=stored_state,
    )
    state, _ = agent.train(seed=0, n_timesteps=400)
    state = jax.tree.map(lambda leaf: leaf[0], state)  # the one seed
    batch, _ = sample_and_burnin_sequences(
        state, agent.buffer, jax.random.PRNGKey(1), 0, stored_state=stored_state
    )
    return np.asarray(batch.obs)  # (steps, windows, columns)


def test_a_windows_first_row_reads_no_previous_action_or_reward():
    obs = _windows(stored_state=False)
    np.testing.assert_array_equal(obs[0, :, PREVIOUS_STEP], 0.0)
    # The rows after it read them: Pendulum pays a negative reward at every
    # step, so a row reading none starts an episode, one in 50 at most.
    later_rewards = obs[1:, :, 4]
    assert (later_rewards != 0.0).mean() > 0.9
    assert np.abs(obs[0, :, :3]).sum() > 0  # the observation itself is kept


def test_a_window_from_a_stored_memory_keeps_its_first_rows_previous_step():
    obs = _windows(stored_state=True)
    assert (obs[0, :, 4] != 0.0).mean() > 0.5
