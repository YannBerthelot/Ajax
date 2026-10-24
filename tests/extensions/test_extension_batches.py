"""What the agents hand the extensions' phases beyond the phase contract
(test_phase_contract): PQN's target the cuts it bootstrapped on, and no raw
observations the replay does not keep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pytest

import ajax
from ajax import PQN
from ajax.extensions.base import Extension

SEEN: list[tuple[np.ndarray, ...]] = []  # one row of arrays per fold call


def _record(*arrays: Any) -> float:
    jax.debug.callback(lambda *a: SEEN.append(tuple(map(np.asarray, a))), *arrays)
    return 0.0


def _train(cls: Any, kwargs: dict, extension: Extension) -> list:
    SEEN.clear()
    cls(**kwargs, extensions=(extension,)).train(seed=0, n_timesteps=48)
    jax.effects_barrier()
    return list(SEEN)


_PQN: dict[str, Any] = {
    "env_id": "CartPole-v1",
    "n_envs": 2,
    "architecture": ("8",),
    "n_steps": 8,
    "num_minibatches": 2,
}


@dataclass(frozen=True)
class RecordDones(Extension):
    name: str = "record_dones"

    def on_target(self, agent_state, ext_state, batch, target, context):
        _record(batch["dones"], batch["terminated"])
        return target


def test_pqn_hands_its_target_extensions_the_dones() -> None:
    """``dones`` is where PQN's target cut the bootstrap: its terminations."""
    rows = _train(PQN, _PQN, RecordDones())
    assert rows
    for dones, terminated in rows:
        np.testing.assert_array_equal(dones, terminated)


RAW: list = []  # the actor-loss batch's raw_observations, at trace time


@dataclass(frozen=True)
class RawObservationsSeen(Extension):
    name: str = "raw_observations_seen"

    def actor_loss(self, agent_state, ext_state, batch, context):
        RAW.append(batch["raw_observations"])
        return 0.0


@pytest.mark.parametrize("agent", ["SAC", "TD3", "REDQ", "ASAC"])
def test_without_an_expert_the_actor_loss_gets_no_raw_observations(agent: str) -> None:
    """The replay keeps the env's own observations only for a run with an
    expert (init_buffer's include_raw_obs): without one, the actor-loss
    batch's raw_observations is None, not observations passed off as raw."""
    RAW.clear()
    getattr(ajax, agent)(
        "Pendulum-v1",
        n_envs=2,
        learning_starts=8,
        batch_size=8,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        extensions=(RawObservationsSeen(),),
    ).train(seed=0, n_timesteps=16)
    assert RAW and all(raw is None for raw in RAW)
