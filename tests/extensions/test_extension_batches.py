"""What the agents hand the extensions' phases: a key of their own for every
loss term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
import pytest

from ajax import APO, DQN, PPO, PQN
from ajax.extensions.base import Extension

# (the key the fold got, the agent's own key then), one row per fold call.
SEEN: list[tuple[np.ndarray, np.ndarray]] = []


def _record(ctx: Any, agent_state: Any) -> float:
    jax.debug.callback(
        lambda key, own: SEEN.append((np.asarray(key), np.asarray(own))),
        ctx.rng,
        agent_state.rng,
    )
    return 0.0


@dataclass(frozen=True)
class CriticKeys(Extension):
    name: str = "critic_keys"

    def critic_loss(self, agent_state, ext_state, batch, ctx):
        return _record(ctx, agent_state)


@dataclass(frozen=True)
class LossKeys(CriticKeys):
    name: str = "loss_keys"

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        return _record(ctx, agent_state)


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
AGENTS: dict[str, tuple[Any, dict[str, Any], Extension]] = {
    "PPO": (PPO, _ROLLOUT, LossKeys()),
    "APO": (APO, _ROLLOUT, LossKeys()),
    "DQN": (DQN, {**_Q, "learning_starts": 8, "batch_size": 8}, CriticKeys()),
    "PQN": (PQN, {**_Q, "n_steps": 8, "num_minibatches": 2}, CriticKeys()),
}


@pytest.mark.parametrize("agent", list(AGENTS))
def test_every_loss_term_gets_a_key_of_its_own(agent: str) -> None:
    """Every critic- and actor-loss fold of a run gets a different key, and
    never the one the agent carries on to draw its own."""
    cls, kwargs, extension = AGENTS[agent]
    SEEN.clear()
    cls(**kwargs, extensions=(extension,)).train(seed=0, n_timesteps=48)
    jax.effects_barrier()
    keys = np.concatenate([np.reshape(key, (-1, 2)) for key, _ in SEEN])
    own = np.concatenate([np.reshape(own, (-1, 2)) for _, own in SEEN])
    assert len(keys) > 2
    assert len(np.unique(keys, axis=0)) == len(keys)
    assert not (keys == own).all(axis=1).any()
