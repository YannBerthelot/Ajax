"""Memory probe: does the memory actually get USED, not just plumbed?

Velocity-masked CartPole is unsolvable without memory: the observation
only contains positions, so the policy must integrate over time to
recover velocities. A feedforward PPO plateaus around 40-60 return;
a recurrent PPO reaches several hundred (max 500) with the same budget.

Calibration on CPU (200k steps, ~10s per run): feedforward ≈ 41,
GRU ≈ 450. The thresholds below leave a wide margin on both sides.
"""

import jax
import jax.numpy as jnp
import pytest
from gymnax.environments.classic_control.cartpole import CartPole

from ajax.agents.PPO.PPO import PPO
from ajax.evaluate import evaluate
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
    actor_state = jax.tree.map(lambda x: x[0], state.actor_state)  # unvmap seed
    rewards, *_ = evaluate(
        env,
        actor_state=actor_state,
        num_episodes=20,
        rng=jax.random.PRNGKey(1),
        env_params=env.default_params,
        recurrent=memory is not None,
        max_eval_steps=500,
    )
    return float(rewards.mean())


@pytest.fixture(scope="module")
def feedforward_baseline():
    return _train_and_eval(None)


# Calibrated finals on CPU with this budget: gru 450, lstm ~4xx,
# transformer 463, mamba 500 — vs feedforward 41.
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
