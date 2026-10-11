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


def _sac_state(env_config: EnvironmentConfig):
    """A freshly initialised SAC (64-wide nets) on ``env_config``."""
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=["64", "relu", "64"],
        critic_architecture=["64", "relu", "64"],
        squash=True,
    )
    buffer = get_buffer(
        **to_state_dict(
            BufferConfig(buffer_size=1000, batch_size=32, n_envs=env_config.n_envs)
        )
    )
    return init_SAC(
        key=jax.random.PRNGKey(0),
        env_args=env_config,
        actor_optimizer_args=optimizer_args,
        critic_optimizer_args=optimizer_args,
        network_args=network_args,
        alpha_args=AlphaConfig(learning_rate=3e-4, alpha_init=1.0),
        buffer=buffer,
    )


@pytest.mark.parametrize("env_id", ["fast", "Pendulum-v1"])
def test_evaluate_returns_scalar_metrics(env_id):
    if env_id == "fast":
        env, env_params = create_brax_env("fast", batch_size=1), None
    else:
        env, env_params = gymnax.make(env_id)
    env_config = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=1, continuous=True
    )
    rewards, avg_entropy, *_ = evaluate(
        env=env,
        actor_state=_sac_state(env_config).actor_state,
        num_episodes=3,
        rng=jax.random.PRNGKey(42),
        env_params=env_params,
        recurrent=False,
    )
    assert rewards.shape == ()
    assert avg_entropy.shape == ()


def _eval_pendulum():
    env, env_params = gymnax.make("Pendulum-v1")
    env_cfg = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=1, continuous=True
    )
    return env, env_params, _sac_state(env_cfg)


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


def _full_torque(obs):
    return jnp.ones(obs.shape[:-1] + (1,))


def test_the_baseline_return_is_the_evaluation_experts():
    """evaluate rolls out eval_expert_policy for the baseline,
    expert_policy building only the actor's input."""
    env, env_params, state = _eval_pendulum()
    kwargs = {
        "env": env,
        "actor_state": state.actor_state,
        "num_episodes": 2,
        "rng": jax.random.PRNGKey(3),
        "env_params": env_params,
    }
    zero = float(evaluate(expert_policy=_zero_torque, **kwargs)[-1])
    full = float(evaluate(expert_policy=_full_torque, **kwargs)[-1])
    both = evaluate(
        expert_policy=_zero_torque, eval_expert_policy=_full_torque, **kwargs
    )
    assert zero != pytest.approx(full)
    assert float(both[-1]) == pytest.approx(full)


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
def test_the_expert_sees_raw_observations_in_collection_and_evaluation(augment):
    """With observation normalisation on, collection and evaluation (the
    expert's own rollout, and with augmentation the agent's) give the expert
    Pendulum's raw observations: cos^2 + sin^2 = 1, which normalised ones
    break, and no training-fraction column. The update calls it on none: it
    reads the actions collection stored."""
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
    # By batch size: collection (2 envs) and evaluation (3 episodes); never
    # the update (8 rows).
    assert {seen.shape[0] for seen in _SEEN} == {2, 3}
    for seen in _SEEN:
        assert seen.shape[-1] == 3
        np.testing.assert_allclose(np.square(seen[..., :2]).sum(-1), 1.0, atol=1e-4)


def _actor_inputs(monkeypatch, expert, **kwargs) -> list:
    """Train a tiny SAC on Pendulum that conditions on the training fraction
    and augments its observations with ``expert``'s action; every input the
    actor is given (collection: 2 envs, evaluation: 3 episodes, the update:
    8 rows)."""
    import ajax.agents.recurrent
    import ajax.environments.interaction
    import ajax.evaluate
    from ajax import SAC
    from ajax.logging.wandb_logging import LoggingConfig

    seen, get_pi = [], ajax.environments.interaction.get_pi

    def recording_get_pi(*args, **kw):
        obs = kw["obs"] if "obs" in kw else args[2]
        jax.debug.callback(lambda o: seen.append(np.asarray(o)), obs)
        return get_pi(*args, **kw)

    for module in (ajax.environments.interaction, ajax.agents.recurrent, ajax.evaluate):
        monkeypatch.setattr(module, "get_pi", recording_get_pi)
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        augment_obs_with_expert_action=True,
        use_train_frac=True,
        **kwargs,
    )
    logged = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    agent.train(seed=0, n_timesteps=64, num_episode_test=3, logging_config=logged)
    jax.effects_barrier()
    assert {x.shape[0] for x in seen} == {2, 3, 8}
    return seen


def test_the_actor_acts_and_trains_on_one_augmented_layout(monkeypatch):
    """The expert's action (a constant -0.9 torque here, -0.45 in agent
    units) sits in the same column of every input the actor is given,
    acting, evaluated or trained, after Pendulum's three observations and
    the training fraction."""
    seen = _actor_inputs(monkeypatch, lambda obs: jnp.full((*obs.shape[:-1], 1), -0.9))
    for x in seen:
        assert x.shape[-1] == 5
        np.testing.assert_allclose(x[..., 4], -0.45, err_msg=f"batch {x.shape[0]}")
        assert ((0.0 <= x[..., 3]) & (x[..., 3] <= 1.0)).all()  # the fraction


@dataclasses.dataclass(frozen=True)
class _AlternatingExpert:
    """A stateful expert called only with its state, ``expert(state, obs)``,
    as a controller with an integrator is: it plays -0.9 and -0.8 in turn."""

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, state, obs):
        return jnp.where(state % 2 == 0, -0.9, -0.8), state + 1


def test_the_update_reads_the_stateful_experts_actions_from_the_buffer(monkeypatch):
    """The update augments the observation and the next one with the
    actions collection stored, which the expert took from its state; it
    never calls the expert without its state. Every input the actor is
    given holds -0.9 or -0.8 in the expert's column (in agent units, -0.45
    or -0.4), the update's both."""
    played = [
        np.round(x[..., 4].astype(float), 4)
        for x in _actor_inputs(monkeypatch, _AlternatingExpert())
    ]
    for column in played:
        assert set(column.tolist()) <= {-0.45, -0.4}, column
    trained = np.concatenate([column for column in played if len(column) == 8])
    assert set(trained.tolist()) == {-0.45, -0.4}
