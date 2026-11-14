"""Memory probe: does the memory actually get USED, not just plumbed?

Velocity-masked CartPole is unsolvable without memory: the observation
only contains positions, so the policy must integrate over time to
recover velocities. A feedforward PPO plateaus around 40-60 return;
a recurrent PPO reaches several hundred (max 500) with the same budget.

The trained policy is scored as the agent's evaluation acts (agent.policy):
PPO normalises its observations by default (#451), and an evaluation fed
the env's raw observations scores a recurrent PPO that solves the task
near 10.

Calibration with PPO's defaults (normalised observations and returns, the
clipped value loss), on x86 Linux CPU, 200k steps, seed 0: feedforward 45,
gru 500, lstm 500, transformer 451, mamba 500; seeds 1-3 gru, lstm and
mamba 500, transformer 364-500. The thresholds below leave a wide margin on
both sides.
"""

import jax
import jax.numpy as jnp
import pytest
from gymnax.environments.classic_control.cartpole import CartPole

from ajax.agents.PPO.PPO import PPO
from ajax.networks.memory import MemoryConfig


class MaskedCartPole(CartPole):
    """CartPole-v1 with velocities zeroed out of the observation."""

    def get_obs(self, state, params=None, key=None):
        return jnp.array([state.x, 0.0, state.theta, 0.0])


def _train_and_eval(memory, n_timesteps=200_000, seed=0):
    env = MaskedCartPole()
    agent = PPO(
        env_id=env,
        n_envs=8,
        n_steps=128,
        batch_size=128,
        n_epochs=4,
        memory=memory,
    )
    state, _ = agent.train(seed=seed, n_timesteps=n_timesteps)
    return _policy_return(agent, state, env)


def _policy_return(agent, state, env, num_episodes=20):
    """The trained policy's mean return over num_episodes episodes of env,
    each counted to its first done (CartPole's 500 steps at most)."""
    policy = agent.policy(state)
    params = env.default_params
    reset_keys = jax.random.split(jax.random.PRNGKey(1), num_episodes)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, params)
    env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def body(carry, key):
        obs, env_state, policy_carry, ended, total = carry
        action, policy_carry = policy.act(obs, policy_carry)
        env_keys = jax.random.split(key, num_episodes)
        stepped = env_step(env_keys, env_state, action.reshape(num_episodes), params)
        obs, env_state, reward, terminated, truncated, _ = stepped
        total = total + jnp.where(ended, 0.0, reward)
        ended = ended | terminated | truncated
        return (obs, env_state, policy_carry, ended, total), None

    no, zeros = jnp.zeros(num_episodes, bool), jnp.zeros(num_episodes)
    first = (obs, env_state, policy.reset(num_episodes), no, zeros)
    step_keys = jax.random.split(jax.random.PRNGKey(2), params.max_steps_in_episode)
    final, _ = jax.lax.scan(body, first, step_keys)
    return float(final[-1].mean())


@pytest.fixture(scope="module")
def feedforward_baseline():
    return _train_and_eval(None)


MEMORIES = [
    MemoryConfig(kind="gru", hidden_size=32),
    MemoryConfig(kind="lstm", hidden_size=32),
    MemoryConfig(kind="transformer", hidden_size=32, num_heads=4, window=16),
    MemoryConfig(kind="mamba", hidden_size=32),
]


@pytest.mark.slow
@pytest.mark.parametrize("memory", MEMORIES, ids=lambda m: m.kind)
def test_recurrent_ppo_uses_memory_on_masked_cartpole(memory, feedforward_baseline):
    recurrent = _train_and_eval(memory)
    # feedforward is fundamentally capped by partial observability;
    # the recurrent agent must clearly break through that ceiling.
    assert feedforward_baseline < 150, (
        "feedforward unexpectedly solved the masked env — probe is broken:"
        f" {feedforward_baseline}"
    )
    assert recurrent > 150, (
        f"{memory.kind} failed the memory probe: {recurrent}"
        f" (feedforward baseline: {feedforward_baseline})"
    )
    assert recurrent > 2 * feedforward_baseline
