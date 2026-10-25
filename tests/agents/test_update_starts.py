"""The update starts every replay actor-critic takes (families'
PolicyUpdateStart and EntropyTemperature.alpha_update_start): before its
start an actor's or a temperature's step is not kept, and at 0 (the default)
nothing waits. SAC's family has its own tests of the same knobs."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from ajax import IQL, TD3, CrossQ

TINY = {"env_id": "Pendulum-v1", "n_envs": 1, "buffer_size": 64, "batch_size": 8}
TINY.update({"learning_starts": 8})
NETS: dict[type, dict[str, Any]] = {
    TD3: {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")},
    IQL: {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")},
    CrossQ: {"actor_width": 8, "critic_width": 8, "policy_delay": 1},
}
STEPS = 24


def _train(cls: type, **kwargs: Any) -> Any:
    agent = cls(**{**TINY, **NETS[cls], **kwargs})
    return agent.train(seed=0, n_timesteps=STEPS)[0]


def _same(first: object, second: object) -> bool:
    return all(
        np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True)
    )


@pytest.mark.parametrize("cls", [TD3, IQL, CrossQ], ids=lambda c: c.__name__)
def test_the_actor_is_kept_from_policy_update_start(cls: type) -> None:
    """Started after the run, the actor stays as initialised while the
    critics learn; started at learning_starts, it moves."""
    untrained = _train(cls, learning_starts=STEPS + 1)
    waiting = _train(cls, policy_update_start=STEPS + 1)
    started = _train(cls, policy_update_start=8)
    assert _same(waiting.actor_state.params, untrained.actor_state.params)
    assert not _same(waiting.critic_state.params, untrained.critic_state.params)
    assert not _same(started.actor_state.params, untrained.actor_state.params)


def test_crossqs_temperature_is_kept_from_alpha_update_start() -> None:
    untrained = _train(CrossQ, learning_starts=STEPS + 1)
    waiting = _train(CrossQ, alpha_update_start=STEPS + 1)
    assert _same(waiting.alpha.params, untrained.alpha.params)
    assert not _same(waiting.actor_state.params, untrained.actor_state.params)
    assert not _same(_train(CrossQ).alpha.params, untrained.alpha.params)
