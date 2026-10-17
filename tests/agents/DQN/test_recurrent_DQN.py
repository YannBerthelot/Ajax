"""Recurrent DQN (memory=MemoryConfig(...)): R2D2's replayed sequences.

The plumbing, not performance: the Q-network's carry is burned in (or
stored at collection) for each replayed window, its target network's with
the target parameters, and acting advances the live carry.
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.DQN.DQN import DQN
from ajax.agents.DQN.networks import DuelingQNetwork
from ajax.agents.DQN.train_DQN import compute_double_dqn_td_target, mse_td_loss
from ajax.extensions.instrumentation import BiasVorePenalty
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig

OPTIONS = {
    "architecture": ("16", "relu"),
    "learning_starts": 64,
    "batch_size": 4,
    "buffer_size": 256,
    "target_update_interval": 10,
    "burn_in": 2,
    "sequence_length": 4,
}


def _train(memory, **kw):
    agent = DQN("CartPole-v1", n_envs=2, memory=memory, **OPTIONS | kw)
    return agent.train(seed=0, n_timesteps=256)[0]


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_recurrent_dqn_trains_without_nans(kind):
    state = _train(MemoryConfig(kind=kind, hidden_size=8, window=8))
    q_state = state.actor_state
    assert "memory_cell" in q_state.params["params"]
    assert int(state.n_updates.max()) > 0
    for leaf in jax.tree.leaves((q_state.params, q_state.target_params)):
        assert jnp.all(jnp.isfinite(leaf))
    # Acting advanced the live carry.
    assert any(jnp.any(c != 0) for c in jax.tree.leaves(q_state.hidden_state))


@pytest.mark.parametrize(
    "kw",
    [
        {"td_target_fn": compute_double_dqn_td_target},
        {"stored_state": True},
        {"q_network_cls": DuelingQNetwork},
    ],
    ids=["double", "stored_state", "dueling"],
)
def test_recurrent_dqn_options(kw):
    state = _train(MemoryConfig("gru", 8), **kw)
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))


def test_recurrent_dqn_refuses_extensions_and_custom_targets():
    with pytest.raises(NotImplementedError, match="Recurrent DQN"):
        _train(MemoryConfig("gru", 8), extensions=[BiasVorePenalty()])
    with pytest.raises(NotImplementedError, match="built-in TD targets"):
        _train(MemoryConfig("gru", 8), td_target_fn=lambda **kw: mse_td_loss)
