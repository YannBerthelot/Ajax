"""End-to-end tests for recurrent TD3 (memory=MemoryConfig(...)).

TD3 exercises two paths the other agents don't: a deterministic actor
with its own memory hook, and a TARGET-actor carry (burn_target_actor)
for the bootstrap action.
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.TD3.TD3 import TD3
from ajax.networks.memory import MemoryConfig


def _train_smoke(memory, n_timesteps=260):
    agent = TD3(
        env_id="Pendulum-v1",
        n_envs=1,
        batch_size=8,
        buffer_size=1000,
        learning_starts=200,
        actor_architecture=("32", "relu"),
        critic_architecture=("32", "relu"),
        burn_in=4,
        sequence_length=8,
        memory=memory,
    )
    state, _ = agent.train(seed=0, n_timesteps=n_timesteps)
    return state


@pytest.mark.parametrize("kind", ["gru", "lstm", "transformer", "mamba"])
def test_recurrent_td3_trains_without_nans(kind):
    state = _train_smoke(MemoryConfig(kind=kind, hidden_size=8, window=8))
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    for leaf in jax.tree.leaves(state.critic_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    for leaf in jax.tree.leaves(state.actor_state.target_params):
        assert jnp.all(jnp.isfinite(leaf))
    # the actor's live carry must have advanced during collection (this
    # goes through TD3's action pipeline, which must hand the carry back)
    assert any(
        jnp.any(leaf != 0) for leaf in jax.tree.leaves(state.actor_state.hidden_state)
    )


def test_recurrent_td3_learning_starts_guard():
    with pytest.raises(ValueError, match="learning_starts"):
        TD3(
            env_id="Pendulum-v1",
            n_envs=1,
            learning_starts=10,
            burn_in=8,
            sequence_length=16,
            memory=MemoryConfig(kind="gru", hidden_size=8),
        )
