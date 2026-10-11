import dataclasses

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
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
    kwargs = {
        "env": env,
        "actor_state": state.actor_state,
        "num_episodes": 4,
        "rng": jax.random.PRNGKey(7),
        "env_params": env_params,
        "recurrent": False,
    }
    # Baseline (natural length — 200 for Pendulum).
    r_base, *_ = evaluate(**kwargs)
    # Over-long length (2x). If done-masking is correct, rewards match.
    r_long, *_ = evaluate(max_eval_steps=400, **kwargs)
    assert float(r_base) == pytest.approx(float(r_long), abs=1e-4)


def _zero_torque(obs):
    return jnp.zeros(obs.shape[:-1] + (1,))


@dataclasses.dataclass(frozen=True)
class _StatefulZeroTorque:
    """``_zero_torque`` in the stateful convention, its state unused."""

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, state, obs):
        return _zero_torque(obs), state


def test_the_expert_rollout_calls_an_expert_in_its_own_convention():
    """A stateless expert is called ``expert(obs)``, a stateful one
    ``expert(state, obs)``, as in the agent's rollout: one expert written
    both ways earns one return."""
    env, env_params, state = _eval_pendulum()
    kwargs = {
        "env": env,
        "actor_state": state.actor_state,
        "num_episodes": 2,
        "rng": jax.random.PRNGKey(3),
        "env_params": env_params,
    }
    stateless = float(evaluate(expert_policy=_zero_torque, **kwargs)[-1])
    stateful = float(evaluate(expert_policy=_StatefulZeroTorque(), **kwargs)[-1])
    assert stateless < 0.0  # Pendulum's cost: every return is negative
    assert stateless == pytest.approx(stateful)


def test_setup_environment_accepts_prebuilt_gymnax_env_without_params():
    """Regression: a prebuilt gymnax env carries env_params=None (prepare_env)
    and the flattening check must fall back to the env's default params
    instead of handing None to observation_space (CartPole reads
    params.x_threshold there)."""
    from gymnax.environments.classic_control.cartpole import CartPole

    from ajax.evaluate import setup_environment

    env, mode, continuous = setup_environment(CartPole(), None, 2, None, 0.99)
    assert mode == "gymnax" and not continuous
    assert env.observation_space(env.default_params).shape == (4,)


_SEEN: list = []  # (batch size, observations) of every expert call


@dataclasses.dataclass(frozen=True)
class _RecordingExpert:
    """An expert recording what it is given; its action depends on the
    observation's scale (the angular velocity, raw in [-8, 8]). Stateful
    (``expert(state, obs)``), and callable as ``expert(obs)`` too, as SAC's
    update calls it to augment observations."""

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, *args):
        obs = args[-1]
        jax.debug.callback(lambda o: _SEEN.append(np.asarray(o)), obs)
        action = jnp.clip(-obs[..., 2:3] / 8.0, -1.0, 1.0)
        return (action, args[0]) if len(args) == 2 else action


@pytest.mark.parametrize("augment", [False, True], ids=["baseline", "augmented"])
def test_the_expert_sees_raw_observations_in_collection_evaluation_and_update(
    augment,
):
    """With observation normalisation on, collection, evaluation (the
    expert's own rollout, and with augmentation the agent's) and the update
    (augmenting its observations and next observations) give the expert
    Pendulum's raw observations: cos^2 + sin^2 = 1, which normalised ones
    break, and no training-fraction column."""
    from ajax import SAC
    from ajax.logging.wandb_logging import LoggingConfig

    _SEEN.clear()
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        expert_policy=_RecordingExpert(),
        expert_buffer_n_steps=0,
        normalize_observations=True,
        augment_obs_with_expert_action=augment,
        use_train_frac=augment,
    )
    logged = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    agent.train(seed=0, n_timesteps=64, num_episode_test=3, logging_config=logged)
    jax.effects_barrier()
    # By batch size: collection (2 envs), evaluation (3 episodes) and, with
    # augmentation, the update (8 rows).
    assert {seen.shape[0] for seen in _SEEN} == ({2, 3, 8} if augment else {2, 3})
    for seen in _SEEN:
        assert seen.shape[-1] == 3
        np.testing.assert_allclose(np.square(seen[..., :2]).sum(-1), 1.0, atol=1e-4)
