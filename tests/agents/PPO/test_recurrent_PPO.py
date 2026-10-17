"""End-to-end tests for recurrent PPO (memory=MemoryConfig(...)).

Small budgets: these verify the plumbing (shapes, carries, resets, BPTT
path compiles and produces finite updates), not final performance.
"""

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.PPO import sequences
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
    # The memory cell's weights are in the actor's tree, so the optimizer
    # updates them (BPTT gradient flow itself: tests/networks/test_memory.py).
    assert "memory_cell" in state.actor_state.params["params"]
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


def test_recurrent_ppo_memory_dict():
    # dict config sugar
    agent = PPO(env_id="CartPole-v1", memory={"kind": "lstm", "hidden_size": 8})
    assert agent.network_args.memory == MemoryConfig(kind="lstm", hidden_size=8)


def test_recurrent_ppo_env_split_minibatching():
    """num_minibatches > 1 with a divisible env axis must train via
    env-split minibatches (full-T trajectories per env subset, each with
    its own rollout-start carries) — this is what decouples the gradient
    step count from n_steps for recurrent PPO."""
    agent = PPO(
        env_id="CartPole-v1",
        n_envs=4,
        n_steps=16,
        batch_size=16,
        n_epochs=2,
        num_minibatches=2,
        memory=MemoryConfig(kind="gru", hidden_size=8),
    )
    state, _ = agent.train(seed=0, n_timesteps=4 * 16 * 3)
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    assert any(
        jnp.any(leaf != 0) for leaf in jax.tree.leaves(state.actor_state.hidden_state)
    )


@pytest.mark.parametrize("kind", ["gru", "mamba"])
def test_recurrent_ppo_bptt_length(kind):
    """Truncated BPTT: rollouts split into bptt_length sequences whose
    start carries are recomputed chunk-wise (never zero mid-episode).
    Pool = (T/L) * n_envs sequences, minibatched by num_minibatches."""
    agent = PPO(
        env_id="CartPole-v1",
        n_envs=4,
        n_steps=32,
        batch_size=32,
        n_epochs=2,
        num_minibatches=4,
        bptt_length=8,
        memory=MemoryConfig(kind=kind, hidden_size=8, window=8),
    )
    state, _ = agent.train(seed=0, n_timesteps=4 * 32 * 3)
    for leaf in jax.tree.leaves(state.actor_state.params):
        assert jnp.all(jnp.isfinite(leaf))
    assert any(
        jnp.any(leaf != 0) for leaf in jax.tree.leaves(state.actor_state.hidden_state)
    )


def test_bptt_length_guards():
    with pytest.raises(ValueError, match="must divide"):
        PPO(
            env_id="CartPole-v1",
            n_steps=30,
            bptt_length=8,
            memory=MemoryConfig(kind="gru", hidden_size=8),
        )
    with pytest.raises(ValueError, match="requires a memory config"):
        PPO(env_id="CartPole-v1", n_steps=32, bptt_length=8)


def test_a_time_limit_bootstraps_on_its_own_episode():
    """At a time limit the next value is the critic on the episode's final
    observation from the carry that consumed the episode (sb3-contrib's
    RecurrentPPO), not on the reset observation from a fresh carry;
    elsewhere it is the next step's value, and the step-by-step pass gives
    the values the sequence pass does."""
    import numpy as np

    from ajax.agents.PPO import train_PPO
    from ajax.networks.networks import predict_value_sequence
    from ajax.state import Transition

    agent = PPO(
        "Pendulum-v1",
        n_envs=2,
        n_steps=8,
        batch_size=8,
        memory=MemoryConfig(kind="gru", hidden_size=8),
    )
    state = train_PPO.init_PPO(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    T, B, cut = 8, 2, 4  # env 1's episode hits its time limit at step 4
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    obs = jax.random.normal(keys[0], (T, B, 3))
    next_obs = jnp.concatenate([obs[1:], jax.random.normal(keys[1], (1, B, 3))])
    next_obs = next_obs.at[cut, 1].set(jax.random.normal(keys[2], (3,)))
    zeros = jnp.zeros((T, B, 1))
    rollout = Transition(
        obs=obs,
        action=zeros,
        reward=zeros,
        terminated=zeros,
        truncated=zeros.at[cut, 1].set(1.0),
        next_obs=next_obs,
        raw_obs=obs,
        log_prob=zeros,
        raw_action=zeros,
    )
    _, values, next_values, resets, _ = sequences.recurrent_values(
        state, rollout, state, length=4
    )
    critic, carry = state.critic_state, state.critic_state.hidden_state
    whole, _ = predict_value_sequence(critic, critic.params, obs, resets, carry)
    np.testing.assert_allclose(values, whole[0], atol=1e-6)
    _, seen = predict_value_sequence(
        critic, critic.params, obs[: cut + 1], resets[: cut + 1], carry
    )
    no_reset = jnp.zeros((1, B), bool)
    own, _ = predict_value_sequence(
        critic, critic.params, next_obs[cut][None], no_reset, seen
    )
    np.testing.assert_allclose(next_values[cut, 1], own[0, 0, 1], atol=1e-6)
    assert abs(float(next_values[cut, 1, 0] - values[cut + 1, 1, 0])) > 1e-3
    following = np.ones((T - 1, B), bool)
    following[cut, 1] = False
    np.testing.assert_allclose(
        next_values[:-1][following], values[1:][following], atol=1e-6
    )
