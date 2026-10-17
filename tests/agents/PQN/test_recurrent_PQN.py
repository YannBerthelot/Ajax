"""Recurrent PQN (memory=MemoryConfig(...)): the plumbing, not performance.

The Q-network reads each env's whole rollout from the carry it started
with: once for the Q(lambda) targets, then on minibatches of whole
rollouts; acting advances the live carry.
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.PQN.PQN import PQN
from ajax.extensions.instrumentation import BiasVorePenalty
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig


def _train(memory, n_envs=4, **kw):
    agent = PQN(
        "CartPole-v1",
        n_envs=n_envs,
        n_steps=16,
        n_epochs=2,
        architecture=("16", "relu"),
        memory=memory,
        **kw,
    )
    return agent.train(seed=0, n_timesteps=n_envs * 16 * 3)[0]


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_recurrent_pqn_trains_without_nans(kind):
    state = _train(MemoryConfig(kind=kind, hidden_size=8, window=8), num_minibatches=2)
    q_state = state.actor_state
    assert "MemoryCell_0" in q_state.params["params"]
    for leaf in jax.tree.leaves(q_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    assert any(jnp.any(c != 0) for c in jax.tree.leaves(q_state.hidden_state))


def test_recurrent_pqn_keeps_whole_rollouts_when_the_envs_do_not_split():
    state = _train(MemoryConfig("gru", 8), n_envs=3, num_minibatches=2)
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))


def test_recurrent_pqn_refuses_extensions():
    with pytest.raises(NotImplementedError, match="Recurrent PQN"):
        _train(MemoryConfig("gru", 8), extensions=[BiasVorePenalty()])
