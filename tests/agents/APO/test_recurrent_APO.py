"""Recurrent APO (memory=MemoryConfig(...)): the plumbing, not performance.

APO replays each env's whole rollout from the carries it started with, as
PPO does with whole-rollout sequences (ajax.agents.PPO.sequences).
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.APO.APO import APO
from ajax.extensions.instrumentation import BiasVorePenalty
from ajax.networks.memory import MEMORY_KINDS, MemoryConfig

NET = ("16", "tanh")


def _agent(memory, env_id="CartPole-v1", n_envs=2, **kw):
    return APO(
        env_id,
        n_envs=n_envs,
        n_steps=16,
        batch_size=16,
        n_epochs=2,
        actor_architecture=NET,
        critic_architecture=NET,
        memory=memory,
        **kw,
    )


@pytest.mark.parametrize("kind", MEMORY_KINDS)
def test_recurrent_apo_trains_without_nans(kind):
    agent = _agent(MemoryConfig(kind=kind, hidden_size=8, window=8))
    state, _ = agent.train(seed=0, n_timesteps=2 * 16 * 3)
    assert "memory_cell" in state.actor_state.params["params"]
    assert "memory_cell" in state.critic_state.params["params"]["ensemble"]
    for leaf in jax.tree.leaves((state.actor_state.params, state.critic_state.params)):
        assert jnp.all(jnp.isfinite(leaf))
    # The actor's carry advanced while acting, the critic's in the update.
    for network in (state.actor_state, state.critic_state):
        carries = jax.tree.leaves(network.hidden_state)
        assert carries and any(jnp.any(c != 0) for c in carries)


@pytest.mark.parametrize("n_envs, num_minibatches", [(2, 2), (3, 2)])
def test_recurrent_apo_minibatches_whole_rollouts(n_envs, num_minibatches):
    """Whole rollouts split over the envs, in one minibatch when they do not
    split evenly; both train on a continuous env."""
    agent = _agent(
        MemoryConfig("gru", 8),
        "Pendulum-v1",
        n_envs=n_envs,
        num_minibatches=num_minibatches,
    )
    state, _ = agent.train(seed=0, n_timesteps=n_envs * 16 * 2)
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))


def test_recurrent_apo_refuses_extensions():
    agent = _agent(MemoryConfig("gru", 8), extensions=[BiasVorePenalty()])
    with pytest.raises(NotImplementedError, match="Recurrent APO"):
        agent.train(seed=0, n_timesteps=32)
