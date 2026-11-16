"""A shared encoder is stored twice: the critic owns it, the actor holds a
copy to act with. Every agent that can share one keeps the two equal.

Contract (LoadedTrainState.shared_params, ajax/state.py; the owner's
request for share_encoder, 2026-11-13): with ``share_encoder=True`` the
critic's optimiser steps the encoder (and the memory block, when there is
one), and after every update the actor's copy is the critic's, so acting
and evaluating read the encoder as the last update left it. The expected
value is the critic's own parameters in the same state, compared bitwise:
a copy, not a computation.

The agents are found from the mixin that gives them the option
(EncoderSharing), not listed: one added later is tested, or the test asks
for its small settings.
"""

from __future__ import annotations

import inspect
from typing import Any

import jax
import numpy as np
import pytest

import ajax
from ajax.agents.families import EncoderSharing

SHARING_AGENTS = sorted(
    name
    for name in ajax.__all__
    if inspect.isclass(getattr(ajax, name))
    and issubclass(getattr(ajax, name), EncoderSharing)
)

NETWORKS: dict[str, Any] = {
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
}
REPLAY: dict[str, Any] = {"buffer_size": 256, "batch_size": 8, "learning_starts": 16}
ROLLOUT: dict[str, Any] = {"n_steps": 8, "n_envs": 2, "n_epochs": 1}
ROLLOUT.update({"num_minibatches": 2})
#: Each agent's settings for a run of a few updates on Pendulum.
SETTINGS: dict[str, dict[str, Any]] = {
    "APO": ROLLOUT,
    "PPO": ROLLOUT,
    "AVG": {},
    "IQL": REPLAY,
    "RVISAC": REPLAY,
    "SAC": REPLAY,
    "TD3": REPLAY,
    "TQC": REPLAY,
    # Its per-member clip, with a shared encoder.
    "ASAC": {**REPLAY, "max_grad_norm": 1.0},
    # A fast test's few updates, not the default 20 a step.
    "REDQ": {**REPLAY, "num_critic_updates": 2},
}


def test_every_agent_that_can_share_an_encoder_has_settings_here() -> None:
    """Catches an agent gaining share_encoder without this contract being
    run on it."""
    assert SHARING_AGENTS == sorted(SETTINGS)


@pytest.mark.parametrize("name", sorted(SETTINGS))
def test_the_actors_copy_is_the_critics_encoder_after_training(name: str) -> None:
    """Catches an update that steps the critic's encoder and leaves the
    actor acting on the one before it (a mirror_shared_encoder call dropped
    from the loop, or an agent's own update path that never had one)."""
    agent = getattr(ajax, name)(
        env_id="Pendulum-v1", share_encoder=True, **NETWORKS, **SETTINGS[name]
    )

    state = jax.tree.map(lambda leaf: leaf[0], agent.train(seed=0, n_timesteps=48)[0])

    assert int(state.critic_state.step) > 0  # its optimiser stepped the encoder
    copy = dict(state.actor_state.shared_params)
    owned = state.critic_state.params["params"]
    assert set(copy) == {"encoder"}
    for module, held in copy.items():
        same = jax.tree.map(np.array_equal, held, owned[module])
        assert all(jax.tree.leaves(same)), module
