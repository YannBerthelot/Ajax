import gymnax
import jax
import pytest
from brax.envs import create as create_brax_env
from flax.serialization import to_state_dict

from ajax.agents.SAC.train_SAC import init_SAC
from ajax.buffers.utils import get_buffer
from ajax.evaluate import evaluate
from ajax.state import (
    AlphaConfig,
    BufferConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)


@pytest.fixture
def fast_env_config():
    env = create_brax_env("fast", batch_size=1)
    return EnvironmentConfig(
        env=env,
        env_params=None,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture
def gymnax_env_config():
    env, env_params = gymnax.make("Pendulum-v1")
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        n_envs=1,
        continuous=True,
    )


@pytest.fixture(params=["fast_env_config", "gymnax_env_config"])
def env_config(request, fast_env_config, gymnax_env_config):
    return fast_env_config if request.param == "fast_env_config" else gymnax_env_config


@pytest.fixture
def SAC_state(env_config):
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
        lstm_hidden_size=None,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)
    buffer = get_buffer(
        **to_state_dict(
            BufferConfig(buffer_size=1000, batch_size=32, n_envs=env_config.n_envs)
        )
    )
    return init_SAC(
        key=key,
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
        buffer=buffer,
    )


def test_evaluate_with_fast_env(env_config, SAC_state):
    num_episodes = 5
    rng = jax.random.PRNGKey(0)

    rewards, avg_entropy, avg_reward, bias, avg_length, expert_rewards = evaluate(
        env=env_config.env,
        actor_state=SAC_state.actor_state,
        num_episodes=num_episodes,
        rng=rng,
        env_params=env_config.env_params,
        recurrent=False,
        lstm_hidden_size=None,
    )

    # Assertions
    assert rewards.shape == ()
    assert avg_entropy.shape == ()


def test_evaluate_with_gymnax_env(env_config, SAC_state):
    num_episodes = 3
    rng = jax.random.PRNGKey(42)

    rewards, avg_entropy, avg_reward, bias, avg_length, expert_rewards = evaluate(
        env=env_config.env,
        actor_state=SAC_state.actor_state,
        num_episodes=num_episodes,
        rng=rng,
        env_params=env_config.env_params,
        recurrent=False,
        lstm_hidden_size=128,
    )

    # Assertions
    assert rewards.shape == ()
    assert avg_entropy.shape == ()


def _eval_pendulum():
    import gymnax

    env, env_params = gymnax.make("Pendulum-v1")
    key = jax.random.PRNGKey(0)
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
        lstm_hidden_size=None,
    )
    alpha_args = AlphaConfig(learning_rate=3e-4, alpha_init=1.0)
    buffer = get_buffer(
        **to_state_dict(BufferConfig(buffer_size=1000, batch_size=32, n_envs=1))
    )
    env_cfg = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=1, continuous=True
    )
    state = init_SAC(
        key=key,
        env_args=env_cfg,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=alpha_args,
        buffer=buffer,
    )
    return env, env_params, state


def test_evaluate_regression_pendulum():
    """Pins eval numerical output on Pendulum-v1 so refactors (e.g. while_loop -> scan)
    don't silently change the reward computation."""
    env, env_params, state = _eval_pendulum()
    rewards, entropy, _, _, length, _ = evaluate(
        env=env,
        actor_state=state.actor_state,
        num_episodes=5,
        rng=jax.random.PRNGKey(42),
        env_params=env_params,
        recurrent=False,
        lstm_hidden_size=None,
    )
    # Pendulum truncates at max_steps=200 for every episode with this policy.
    assert float(length) == 200.0
    assert abs(float(rewards) - (-1319.96)) < 5.0
    assert abs(float(entropy) - 0.4189) < 1e-2


def test_evaluate_done_masking_no_accumulation_past_done():
    """Rewards must not accumulate past per-env `done`. With Pendulum (fixed
    200-step episodes), running eval for longer than 200 steps must yield the
    same reward as running it for exactly 200 steps — this is the scan
    equivalent of while_loop's early exit, and the correctness condition the
    while_loop->scan conversion must preserve."""
    env, env_params, state = _eval_pendulum()
    kwargs = dict(
        env=env,
        actor_state=state.actor_state,
        num_episodes=4,
        rng=jax.random.PRNGKey(7),
        env_params=env_params,
        recurrent=False,
        lstm_hidden_size=None,
    )
    # Baseline (natural length — 200 for Pendulum).
    r_base, *_ = evaluate(**kwargs)
    # Over-long length (2x). If done-masking is correct, rewards match.
    r_long, *_ = evaluate(max_eval_steps=400, **kwargs)
    assert float(r_base) == pytest.approx(float(r_long), abs=1e-4)
