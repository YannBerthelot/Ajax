"""Logging leaves training as it is: a logged run ends in the state the same
run ends in unlogged, every field training reads included.

Why it can break: since the evaluations run between chunks of training
(build_chunked_train), a logged run is a chain of programs (chunks of ticks,
then each evaluation on its own) where an unlogged run is one scan. A chunk
boundary that drops or repeats a tick, or an evaluation that draws from the
training key or writes into the training state, would change what the agent
learns only when it is logged.

One agent per loop family: the replay off-policy loop (SAC, and DQN's
discrete one), the on-policy loop (PPO), an agent iterating its own way
(UDRL), and the own-replay agents, one evaluating every so many ticks
(DreamerV3) and one every so many env steps (TDMPC2). The two runs compile
apart, so floats may differ in rounding: rtol 1e-5.
"""

from __future__ import annotations

from typing import Any

import gymnax
import jax
import numpy as np
import pytest

from ajax import DQN, PPO, SAC
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.agents.UDRL.UDRL import UDRL
from ajax.logging.wandb_logging import LoggingConfig
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY
from tests.agents.TDMPC2.common import TINY_AGENT as TDMPC2_TINY

NET = ("16", "relu")
#: Episodes of 20 steps: the world models' evaluations run whole episodes.
SHORT_PENDULUM = gymnax.make("Pendulum-v1")[1].replace(max_steps_in_episode=20)
SHORT_CARTPOLE = gymnax.make("CartPole-v1")[1].replace(max_steps_in_episode=20)

#: Each family's agent, its budget and its log_frequency: two evaluations a
#: run, the first after updates have started, so that training goes on
#: across a chunk boundary.
RUNS: dict[str, tuple[Any, int, int]] = {
    "SAC": (
        lambda: SAC(
            "Pendulum-v1",
            n_envs=2,
            actor_architecture=NET,
            critic_architecture=NET,
            learning_starts=8,
            batch_size=8,
            buffer_size=128,
            policy_update_start=0,
        ),
        64,
        32,
    ),
    "DQN": (
        lambda: DQN(
            "CartPole-v1",
            n_envs=2,
            architecture=NET,
            learning_starts=16,
            batch_size=8,
            buffer_size=128,
            target_update_interval=10,
        ),
        64,
        32,
    ),
    "PPO": (
        lambda: PPO(
            "Pendulum-v1",
            n_envs=2,
            actor_architecture=NET,
            critic_architecture=NET,
            n_steps=16,
            batch_size=16,
            n_epochs=1,
        ),
        128,
        64,
    ),
    "UDRL": (
        lambda: UDRL(
            "CartPole-v1",
            n_envs=2,
            actor_architecture=NET,
            n_steps=8,
            batch_size=8,
            n_updates_per_iter=2,
            buffer_capacity=8,
        ),
        64,
        32,
    ),
    "DreamerV3": (
        lambda: DreamerV3(
            "CartPole-v1",
            env_params=SHORT_CARTPOLE,
            n_envs=2,
            train_ratio=8,
            **DREAMER_TINY,
        ),
        40,
        20,
    ),
    "TDMPC2": (
        lambda: TDMPC2(
            "Pendulum-v1", env_params=SHORT_PENDULUM, seed_steps=8, **TDMPC2_TINY
        ),
        24,
        12,
    ),
}
#: The fields only logging writes: the count of evaluations made, and
#: DreamerV3's training-metric means, which restart after each log.
LOGGING_ONLY = frozenset({"n_logs", "train_metrics"})


def training_leaves(state: Any) -> dict[str, np.ndarray]:
    """Every leaf of the state outside LOGGING_ONLY, by path."""
    leaves = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        if getattr(path[0], "name", None) not in LOGGING_ONLY:
            leaves[jax.tree_util.keystr(path)] = np.asarray(leaf)
    return leaves


@pytest.mark.parametrize("name", sorted(RUNS))
def test_a_logged_run_trains_as_the_same_run_unlogged(name: str) -> None:
    build, n_timesteps, log_frequency = RUNS[name]
    agent = build()
    unlogged, _ = agent.train(seed=0, n_timesteps=n_timesteps)
    logging = LoggingConfig(log_frequency=log_frequency, backends=())
    logged, rows = agent.train(
        seed=0, n_timesteps=n_timesteps, num_episode_test=1, logging_config=logging
    )

    evaluated = np.asarray(rows["timestep"]) >= 0
    assert evaluated.sum() == 2  # else the logged run was the unlogged one
    expected = training_leaves(unlogged)
    actual = training_leaves(logged)
    assert actual.keys() == expected.keys()
    for path, value in expected.items():
        if np.issubdtype(value.dtype, np.inexact):
            np.testing.assert_allclose(actual[path], value, rtol=1e-5, err_msg=path)
        else:
            np.testing.assert_array_equal(actual[path], value, err_msg=path)
