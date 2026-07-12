"""End-to-end tests for recurrent PPO (memory=MemoryConfig(...)).

Small budgets: these verify the plumbing (shapes, carries, resets, BPTT
path compiles and produces finite updates), not final performance.
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.PPO.PPO import PPO
from ajax.networks.memory import MemoryConfig


def _train_smoke(memory, env_id="CartPole-v1", n_envs=2, n_steps=16):
    agent = PPO(
        env_id=env_id,
        n_envs=n_envs,
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=2,
        memory=memory,
    )
    state, _ = agent.train(seed=0, n_timesteps=n_envs * n_steps * 3)
    return state


@pytest.mark.parametrize("kind", ["gru", "lstm", "transformer", "mamba"])
def test_recurrent_ppo_trains_without_nans(kind):
    state = _train_smoke(MemoryConfig(kind=kind, hidden_size=8, window=8))
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    for leaf in jax.tree.leaves(state.critic_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    # both networks maintain a live carry
    assert state.actor_state.hidden_state is not None
    assert state.critic_state.hidden_state is not None
    for leaf in jax.tree.leaves(state.actor_state.hidden_state):
        assert jnp.all(jnp.isfinite(leaf))
    # the actor's carry must actually have advanced (non-zero after acting)
    assert any(
        jnp.any(leaf != 0) for leaf in jax.tree.leaves(state.actor_state.hidden_state)
    )


def test_recurrent_ppo_continuous_env():
    state = _train_smoke(MemoryConfig(kind="gru", hidden_size=8), env_id="Pendulum-v1")
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))


def test_recurrent_ppo_memory_dict_and_legacy_alias():
    # dict config sugar
    agent = PPO(env_id="CartPole-v1", memory={"kind": "lstm", "hidden_size": 8})
    assert agent.network_args.memory == MemoryConfig(kind="lstm", hidden_size=8)
    # legacy lstm_hidden_size maps to a GRU (historical behaviour)
    agent = PPO(env_id="CartPole-v1", lstm_hidden_size=8)
    assert agent.network_args.memory == MemoryConfig(kind="gru", hidden_size=8)


def test_recurrent_ppo_actor_params_contain_memory_cell():
    """The actor's param tree must include the memory cell's weights so the
    optimizer updates them (BPTT gradient flow itself is covered by the
    cell-level tests in tests/networks/test_memory.py)."""
    from flax.traverse_util import flatten_dict

    state = _train_smoke(MemoryConfig(kind="gru", hidden_size=8))
    params = jax.tree.map(lambda x: x[0], state.actor_state.params)  # unvmap seed
    paths = ["/".join(map(str, k)).lower() for k in flatten_dict(params).keys()]
    assert any("memory_cell" in p or "gru" in p for p in paths), paths
