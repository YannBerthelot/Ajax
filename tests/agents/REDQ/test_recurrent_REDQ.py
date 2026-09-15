"""End-to-end tests for recurrent REDQ (memory=MemoryConfig(...))."""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.REDQ.REDQ import REDQ
from ajax.networks.memory import MemoryConfig


def _train_smoke(memory, n_timesteps=260):
    agent = REDQ(
        env_id="Pendulum-v1",
        n_envs=1,
        batch_size=8,
        buffer_size=1000,
        learning_starts=200,
        num_critics=3,
        subset_size=2,
        num_critic_updates=2,
        burn_in=4,
        sequence_length=8,
        memory=memory,
    )
    state, _ = agent.train(seed=0, n_timesteps=n_timesteps)
    return state


@pytest.mark.parametrize("kind", ["gru", "lstm", "transformer", "mamba"])
def test_recurrent_redq_trains_without_nans(kind):
    state = _train_smoke(MemoryConfig(kind=kind, hidden_size=8, window=8))
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    for leaf in jax.tree.leaves(state.critic_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    # the actor's live carry must have advanced during collection
    assert any(
        jnp.any(leaf != 0) for leaf in jax.tree.leaves(state.actor_state.hidden_state)
    )
    # critic ensemble carry keeps the (num_critics,) leading axis
    # (train vmaps over seeds, so it's the second axis here)
    for leaf in jax.tree.leaves(state.critic_state.hidden_state):
        assert leaf.shape[1] == 3


def test_recurrent_redq_learning_starts_guard():
    with pytest.raises(ValueError, match="learning_starts"):
        REDQ(
            env_id="Pendulum-v1",
            n_envs=1,
            learning_starts=10,
            burn_in=8,
            sequence_length=16,
            memory=MemoryConfig(kind="gru", hidden_size=8),
        )
