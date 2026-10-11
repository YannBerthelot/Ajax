"""What the agents hand the extensions' phases: a key of their own for every
loss term, PQN's target the cuts it bootstrapped on, and no raw observations
the replay does not keep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pytest

import ajax
from ajax import APO, DQN, PPO, PQN
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


@dataclass(frozen=True)
class CriticKeys(Extension):
    """Records the key each loss fold gets and the agent's own key then."""

    name: str = "critic_keys"

    def critic_loss(self, agent_state, ext_state, batch, ctx):
        return _record(ctx.rng, agent_state.rng)


@dataclass(frozen=True)
class LossKeys(CriticKeys):
    name: str = "loss_keys"

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        return _record(ctx.rng, agent_state.rng)


_ROLLOUT: dict[str, Any] = {
    "env_id": "Pendulum-v1",
    "n_envs": 2,
    "n_steps": 8,
    "batch_size": 8,
    "n_epochs": 2,
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
}
_Q: dict[str, Any] = {"env_id": "CartPole-v1", "n_envs": 2, "architecture": ("8",)}
_PQN: dict[str, Any] = {**_Q, "n_steps": 8, "num_minibatches": 2}
AGENTS: dict[str, tuple[Any, dict[str, Any], Extension]] = {
    "PPO": (PPO, _ROLLOUT, LossKeys()),
    "APO": (APO, _ROLLOUT, LossKeys()),
    "DQN": (DQN, {**_Q, "learning_starts": 8, "batch_size": 8}, CriticKeys()),
    "PQN": (PQN, _PQN, CriticKeys()),
}


@pytest.mark.parametrize("agent", list(AGENTS))
def test_every_loss_term_gets_a_key_of_its_own(agent: str) -> None:
    """Every critic- and actor-loss fold of a run gets a different key, and
    never the one the agent carries on to draw its own."""
    rows = _train(*AGENTS[agent])
    keys, own = (
        np.concatenate([np.reshape(r[i], (-1, 2)) for r in rows]) for i in (0, 1)
    )
    assert len(keys) > 2
    assert len(np.unique(keys, axis=0)) == len(keys)
    assert not (keys == own).all(axis=1).any()


@dataclass(frozen=True)
class RecordDones(Extension):
    name: str = "record_dones"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
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

    def actor_loss(self, agent_state, ext_state, batch, ctx):
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
