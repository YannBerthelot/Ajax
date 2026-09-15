"""End-to-end tests for recurrent ASAC (memory=MemoryConfig(...)).

Covers the trajectory-buffer path, burn-in carry computation, and
sequence-mode losses. Small budgets: plumbing checks, not performance.
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.ASAC.ASAC import ASAC
from ajax.buffers.utils import get_buffer, get_sequence_batch_from_buffer
from ajax.networks.memory import MemoryConfig


def test_sequence_buffer_shapes():
    """get_buffer(sequence_length=L) samples time-major (L, B, ...) batches
    from single-step (n_envs, ...) adds."""
    n_envs, seq_len, batch_size, obs_dim = 2, 5, 3, 4
    buffer = get_buffer(
        buffer_size=1000,
        batch_size=batch_size,
        n_envs=n_envs,
        sequence_length=seq_len,
    )
    transition = {
        "obs": jnp.zeros((n_envs, obs_dim)),
        "reward": jnp.zeros((n_envs, 1)),
    }
    state = buffer.init({k: v[0] for k, v in transition.items()})
    for t in range(20):
        state = buffer.add(
            state,
            {
                "obs": jnp.full((n_envs, obs_dim), float(t)),
                "reward": jnp.full((n_envs, 1), float(t)),
            },
        )
    batch = get_sequence_batch_from_buffer(buffer, state, jax.random.PRNGKey(0))
    assert batch["obs"].shape == (seq_len, batch_size, obs_dim)
    assert batch["reward"].shape == (seq_len, batch_size, 1)
    # sequences must be temporally contiguous: consecutive steps differ by 1
    diffs = jnp.diff(batch["reward"][:, :, 0], axis=0)
    assert jnp.all(diffs == 1.0)


def _train_smoke(memory, n_timesteps=260):
    agent = ASAC(
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
def test_recurrent_asac_trains_without_nans(kind):
    state = _train_smoke(MemoryConfig(kind=kind, hidden_size=8, window=8))
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    for leaf in jax.tree.leaves(state.critic_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    for leaf in jax.tree.leaves(state.critic_state.target_params):
        assert jnp.all(jnp.isfinite(leaf))
    assert jnp.isfinite(state.theta)
    # the actor's live carry must have advanced during collection
    assert any(
        jnp.any(leaf != 0) for leaf in jax.tree.leaves(state.actor_state.hidden_state)
    )
    # critic ensemble carry keeps its (num=2, batch) leading axes
    # (train vmaps over seeds, so strip that leading axis first)
    for leaf in jax.tree.leaves(state.critic_state.hidden_state):
        assert leaf.shape[1] == 2


def test_recurrent_asac_rejects_expert_options():
    # expert guidance is rejected at make_train time
    agent = ASAC(
        env_id="Pendulum-v1",
        n_envs=1,
        learning_starts=200,
        memory=MemoryConfig(kind="gru", hidden_size=8),
        target_modifier=lambda *a: a,
    )
    with pytest.raises(NotImplementedError, match="Recurrent ASAC"):
        agent.train(seed=0, n_timesteps=10)


def test_recurrent_asac_learning_starts_guard():
    with pytest.raises(ValueError, match="learning_starts"):
        ASAC(
            env_id="Pendulum-v1",
            n_envs=1,
            learning_starts=10,
            burn_in=8,
            sequence_length=16,
            memory=MemoryConfig(kind="gru", hidden_size=8),
        )


def test_unsupported_agents_raise():
    from ajax.agents.AVG.AVG import AVG

    with pytest.raises(NotImplementedError, match="does not support recurrent"):
        AVG(env_id="Pendulum-v1", lstm_hidden_size=8)
