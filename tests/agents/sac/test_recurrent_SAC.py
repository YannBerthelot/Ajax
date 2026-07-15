"""End-to-end tests for recurrent SAC (memory=MemoryConfig(...)).

Small budgets: these verify the sequence-replay plumbing (trajectory
buffer, burn-in carries, sequence losses), not performance.
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.SAC.SAC import SAC
from ajax.networks.memory import MemoryConfig


def _train_smoke(memory, n_timesteps=260):
    agent = SAC(
        env_id="Pendulum-v1",
        n_envs=1,
        batch_size=8,
        buffer_size=1000,
        learning_starts=200,
        burn_in=4,
        sequence_length=8,
        memory=memory,
    )
    state, _ = agent.train(seed=0, n_timesteps=n_timesteps)
    return state


@pytest.mark.parametrize("kind", ["gru", "lstm", "transformer", "mamba"])
def test_recurrent_sac_trains_without_nans(kind):
    state = _train_smoke(MemoryConfig(kind=kind, hidden_size=8, window=8))
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    for leaf in jax.tree.leaves(state.critic_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    for leaf in jax.tree.leaves(state.critic_state.target_params):
        assert jnp.all(jnp.isfinite(leaf))
    # the actor's live carry must have advanced during collection
    assert any(
        jnp.any(leaf != 0) for leaf in jax.tree.leaves(state.actor_state.hidden_state)
    )


def test_recurrent_sac_rejects_expert_options():
    def fake_expert(obs):
        return jnp.zeros(obs.shape[:-1] + (1,))

    agent = SAC(
        env_id="Pendulum-v1",
        n_envs=1,
        learning_starts=200,
        memory=MemoryConfig(kind="gru", hidden_size=8),
        expert_policy=fake_expert,
    )
    with pytest.raises(NotImplementedError, match="Recurrent SAC"):
        agent.train(seed=0, n_timesteps=10)


def test_recurrent_sac_learning_starts_guard():
    with pytest.raises(ValueError, match="learning_starts"):
        SAC(
            env_id="Pendulum-v1",
            n_envs=1,
            learning_starts=10,
            burn_in=8,
            sequence_length=16,
            memory=MemoryConfig(kind="gru", hidden_size=8),
        )


@pytest.mark.parametrize("kind", ["gru", "lstm", "transformer", "mamba"])
def test_recurrent_sac_stored_state(kind):
    """R2D2 stored-state replay: actor carries recorded at collection time
    are written to the buffer and used to initialize replayed sequences."""
    agent = SAC(
        env_id="Pendulum-v1",
        n_envs=1,
        batch_size=8,
        buffer_size=1000,
        learning_starts=200,
        burn_in=4,
        sequence_length=8,
        memory=MemoryConfig(kind=kind, hidden_size=8, window=8),
        stored_state=True,
    )
    state, _ = agent.train(seed=0, n_timesteps=260)
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    buf = state.collector_state.buffer_state.experience
    assert "actor_carry" in buf
    # carries recorded after warmup must be non-zero (the actor acted)
    assert jnp.any(buf["actor_carry"] != 0)


def test_stored_state_requires_memory():
    with pytest.raises(ValueError, match="stored_state"):
        SAC(env_id="Pendulum-v1", stored_state=True)
