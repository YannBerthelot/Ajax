"""The update starts every replay actor-critic takes (families'
PolicyUpdateStart and EntropyTemperature.alpha_update_start): before its
start an actor's or a temperature's step is not kept, and at 0 (the default)
nothing waits. SAC's family has its own tests of the same knobs."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from ajax import ASAC, AVG, IQL, REDQ, SAC, TD3, TQC, CrossQ

TINY = {"env_id": "Pendulum-v1", "n_envs": 1, "buffer_size": 64, "batch_size": 8}
TINY.update({"learning_starts": 8})
NETS: dict[type, dict[str, Any]] = {
    AVG: {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")},
    TD3: {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")},
    IQL: {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")},
    CrossQ: {"actor_width": 8, "critic_width": 8, "policy_delay": 1},
    SAC: {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")},
    TQC: {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")},
    # One critic step per update: the fast tier's budget of gradient steps.
    REDQ: {
        "actor_architecture": ("8", "relu"),
        "critic_architecture": ("8", "relu"),
        "num_critic_updates": 1,
    },
    # Its temperature learned (ASAC fixes it by default).
    ASAC: {
        "actor_architecture": ("8", "relu"),
        "critic_architecture": ("8", "relu"),
        "fixed_alpha": False,
    },
}
STEPS = 24


def _train(cls: Any, steps: int = STEPS, **kwargs: Any) -> Any:
    tiny = {key: value for key, value in TINY.items() if key in cls.field_names()}
    agent = cls(**{**tiny, **NETS[cls], **kwargs})
    return agent.train(seed=0, n_timesteps=steps)[0]


def _same(first: object, second: object) -> bool:
    return all(
        np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True)
    )


@pytest.mark.parametrize(
    "cls", [TD3, IQL, CrossQ, SAC, TQC, REDQ, ASAC, AVG], ids=lambda c: c.__name__
)
def test_the_actor_is_kept_from_policy_update_start(cls: type) -> None:
    """Started after the run, the actor stays as initialised while the
    critics learn; started at learning_starts, it moves."""
    untrained = _train(cls, learning_starts=STEPS + 1)
    waiting = _train(cls, policy_update_start=STEPS + 1)
    started = _train(cls, policy_update_start=8)
    assert _same(waiting.actor_state.params, untrained.actor_state.params)
    assert not _same(waiting.critic_state.params, untrained.critic_state.params)
    assert not _same(started.actor_state.params, untrained.actor_state.params)
    # The updates the actor skips log its last step's metrics
    # (update_steps.run_actor_if): none yet while it waits.
    assert all(np.isnan(leaf).all() for leaf in _actor_logs(waiting))
    assert all(np.isfinite(leaf).all() for leaf in _actor_logs(started))


def _actor_logs(state: Any) -> list:
    leaves = jax.tree.leaves(state.last_actor_metrics)
    assert leaves  # the loop gave the state the slot
    return leaves


@pytest.mark.parametrize(
    "cls", [CrossQ, SAC, TQC, REDQ, ASAC], ids=lambda c: c.__name__
)
def test_the_temperature_is_kept_from_alpha_update_start(cls: type) -> None:
    untrained = _train(cls, learning_starts=STEPS + 1)
    waiting = _train(cls, alpha_update_start=STEPS + 1)
    assert _same(waiting.alpha.params, untrained.alpha.params)
    assert not _same(waiting.actor_state.params, untrained.actor_state.params)
    assert not _same(_train(cls).alpha.params, untrained.alpha.params)


def test_online_the_actor_and_temperature_wait_their_own_start() -> None:
    """Online, the updates start at learning_starts (8) and the actor's and
    the temperature's steps at their own start (16 env steps collected):
    kept from the update at 16 on, none before. The bug: the online folding
    of the starts (ReplayAgent.from_update_start) reading a start of 16 as
    already over at the first update, so both step from 8."""
    starts = {"policy_update_start": 16, "alpha_update_start": 16}
    untrained = _train(SAC, learning_starts=STEPS + 1)
    before = _train(SAC, steps=15, **starts)
    at_start = _train(SAC, steps=16, **starts)
    assert not _same(before.critic_state.params, untrained.critic_state.params)
    assert _same(before.actor_state.params, untrained.actor_state.params)
    assert _same(before.alpha.params, untrained.alpha.params)
    assert not _same(at_start.actor_state.params, untrained.actor_state.params)
    assert not _same(at_start.alpha.params, untrained.alpha.params)


def test_offline_the_actor_waits_its_start_whatever_learning_starts() -> None:
    """Offline (train_offline) every tick updates and the timestep counts
    the updates, so a policy_update_start waits that many updates, even one
    under learning_starts: online, such a start is over at every update and
    its gate goes (ReplayAgent.from_update_start). The bug: that folding
    reaching train_offline, whose actor would then step from its first
    update."""
    from ajax.buffers import dataset as ds
    from tests.environments.identity_env import IdentityEnv

    def offline_actor(policy_update_start: int) -> list:
        agent = SAC(
            IdentityEnv(discrete=False),
            n_envs=1,
            buffer_size=64,
            batch_size=8,
            learning_starts=100,
            policy_update_start=policy_update_start,
            alpha_update_start=0,
            **NETS[SAC],
        )
        rows = ds.record(lambda o: o[1:], agent.env_args, jax.random.PRNGKey(0), 24)
        state = agent.train_offline(rows, n_updates=8, seed=0)[0]
        return jax.tree.leaves(state.actor_state.params)

    never = offline_actor(10**6)
    assert _same(offline_actor(50), never)
    assert not _same(offline_actor(0), never)
