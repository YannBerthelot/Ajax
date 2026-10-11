"""The PID policy option: its actors, its gains, and every agent taking it."""

import importlib

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.SAC.SAC import SAC
from ajax.agents.SAC.utils import SquashedNormal
from ajax.agents.TD3.networks import (
    DeterministicPIDActor,
    get_initialized_td3_actor_critic,
)
from ajax.evaluate import evaluate
from ajax.modules.pid_actor import PIDActor, PIDLaw, PIDPolicy, pid_gains
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig
from tests.environments import paid_action

from . import setpoint_env

# Two actions, two errors: obs = [e (2), I (2), D (2), raw (3)].
GAINS = {
    "kp": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    "ki": jnp.array([[0.5, 0.0], [0.0, -0.5]]),
    "kd": jnp.array([[0.0, 0.1], [-0.2, 0.0]]),
}
OBS = jax.random.normal(jax.random.PRNGKey(1), (5, 9))


def _set_gains(params, gains):
    return {"params": {**params["params"], "pid": dict(gains)}}


def _by_hand(obs, gains):
    e, i, d = obs[:, 0:2], obs[:, 2:4], obs[:, 4:6]
    return e @ gains["kp"].T + i @ gains["ki"].T + d @ gains["kd"].T


def test_the_fixed_actor_parameters_are_the_gains():
    """No bias, no hidden layer: kp, ki, kd and the log-std are all there is,
    and the mean is the PID law with them."""
    actor = PIDActor(PIDPolicy(n_errors=2), action_dim=2)
    params = actor.init(jax.random.PRNGKey(0), OBS)
    shapes = jax.tree.map(jnp.shape, params["params"])
    assert shapes == {"pid": dict.fromkeys(GAINS, (2, 2)), "log_std": (2,)}
    params = _set_gains(params, GAINS)
    np.testing.assert_allclose(
        actor.apply(params, OBS).mean(), _by_hand(OBS, GAINS), 1e-6
    )
    for name, value in pid_gains(params).items():
        np.testing.assert_array_equal(value, GAINS[name])


def test_gains_start_at_zero_and_exploration_is_state_independent():
    actor = PIDActor(PIDPolicy(n_errors=2), action_dim=2, log_std_init=-0.5)
    dist = actor.apply(actor.init(jax.random.PRNGKey(0), OBS), OBS)
    assert isinstance(dist, distrax.Normal)
    np.testing.assert_array_equal(dist.mean(), 0.0)
    np.testing.assert_allclose(dist.stddev(), jnp.exp(-0.5))


def test_the_squashed_actor_squashes_the_pid_output():
    actor = PIDActor(PIDPolicy(n_errors=2), action_dim=2, squash=True)
    params = _set_gains(actor.init(jax.random.PRNGKey(0), OBS), GAINS)
    dist = actor.apply(params, OBS)
    assert isinstance(dist, SquashedNormal)
    np.testing.assert_allclose(dist.distribution.loc, _by_hand(OBS, GAINS), 1e-6)


def test_observation_gains_with_a_constant_network_are_the_fixed_ones():
    """Zero kernel, the fixed gains as bias: the same law, the same gains."""
    policy = PIDPolicy(n_errors=2, gains="observation", architecture=("8", "relu"))
    law = PIDLaw(policy, action_dim=2)
    params = law.init(jax.random.PRNGKey(0), OBS)
    head = params["params"]["head"]
    stacked = jnp.stack([GAINS[g] for g in ("kp", "ki", "kd")])
    head = {"kernel": jnp.zeros_like(head["kernel"]), "bias": stacked.reshape(-1)}
    params = {"params": {**params["params"], "head": head}}
    np.testing.assert_allclose(law.apply(params, OBS), _by_hand(OBS, GAINS), 1e-6)
    gains = law.apply(params, OBS, method="gains")
    np.testing.assert_allclose(gains, jnp.broadcast_to(stacked, (5, 3, 2, 2)))
    with pytest.raises(ValueError, match="predicted from the observation"):
        pid_gains({"params": {"pid": params["params"]}})


def test_observation_gains_vary_with_the_observation():
    policy = PIDPolicy(gains="observation", architecture=("8", "relu"))
    law = PIDLaw(policy, action_dim=1)
    params = law.init(jax.random.PRNGKey(0), OBS)
    params = jax.tree.map(lambda x: jnp.ones_like(x) * 0.3, params)
    gains = law.apply(params, OBS, method="gains")
    assert gains.shape == (5, 3, 1, 1) and float(jnp.std(gains[:, 0])) > 0


@pytest.mark.parametrize(
    "kwargs",
    [{"n_errors": 0}, {"gains": "scheduled"}, {"architecture": ("8", "relu")}],
)
def test_bad_policies_raise(kwargs):
    with pytest.raises(ValueError):
        PIDPolicy(**kwargs)


def _build(memory=None, td3=False):
    env, params = setpoint_env.make()
    env_config = EnvironmentConfig(
        env=env, env_params=params, n_envs=2, continuous=True
    )
    net = NetworkConfig(actor_architecture=PIDPolicy(), critic_architecture=("8",))
    net = net.replace(memory=memory)
    opt = OptimizerConfig(learning_rate=1e-3)
    if td3:
        return get_initialized_td3_actor_critic(
            jax.random.PRNGKey(0), env_config, opt, opt, net
        )
    return get_initialized_actor_critic(
        jax.random.PRNGKey(0), env_config, opt, opt, net, continuous=True, squash=True
    )


def test_the_factories_build_pid_actors():
    obs = jnp.ones((3, 5))
    actor, _ = _build()
    assert isinstance(actor.apply_fn.__self__, PIDActor)
    assert isinstance(actor.apply_fn(actor.params, obs), SquashedNormal)
    actor, _ = _build(td3=True)
    assert isinstance(actor.apply_fn.__self__, DeterministicPIDActor)
    params = _set_gains(actor.params, {g: jnp.full((1, 1), 0.2) for g in GAINS})
    np.testing.assert_allclose(actor.apply_fn(params, obs).mean(), jnp.tanh(0.6))


@pytest.mark.parametrize("td3", [False, True])
def test_a_pid_actor_takes_no_memory(td3):
    with pytest.raises(NotImplementedError, match="no memory"):
        _build(memory=MemoryConfig("gru", 8), td3=td3)


def test_a_pid_actor_needs_continuous_actions():
    env, params = setpoint_env.make()
    env_config = EnvironmentConfig(
        env=env, env_params=params, n_envs=1, continuous=True
    )
    net = NetworkConfig(actor_architecture=PIDPolicy(), critic_architecture=("8",))
    opt = OptimizerConfig(learning_rate=1e-3)
    with pytest.raises(ValueError, match="continuous"):
        get_initialized_actor_critic(
            jax.random.PRNGKey(0), env_config, opt, opt, net, continuous=False
        )


def test_apg_refuses_a_pid_policy():
    """APG's PID is its output head (PIDHeadConfig), not this option."""
    from ajax.agents.APG.train_APG import build_controller

    env, params = setpoint_env.make()
    env_config = EnvironmentConfig(
        env=env, env_params=params, n_envs=1, continuous=True
    )
    net = NetworkConfig(actor_architecture=PIDPolicy(), critic_architecture=())
    with pytest.raises(TypeError, match="PIDHeadConfig"):
        build_controller(env_config, net, pid=None, squash=True)


_REPLAY = {"learning_starts": 16, "buffer_size": 256, "batch_size": 8}
_ROLLOUT = {"n_steps": 16, "batch_size": 8, "n_epochs": 1}
# id: (agent, overrides, steps); SAC updates its policy from step 2000 only.
# AVG always normalises its observations (env-side); the -norm cases
# normalise them agent-side.
AGENTS = {
    "SAC": ("SAC", _REPLAY, 2_100),
    "SAC-norm": ("SAC", _REPLAY | {"normalize_obs_running": True}, 2_100),
    "TD3": ("TD3", _REPLAY, 64),
    "REDQ": ("REDQ", _REPLAY, 64),
    "ASAC": ("ASAC", _REPLAY, 64),
    "AVG": ("AVG", {}, 64),
    "PPO": ("PPO", _ROLLOUT, 64),
    "PPO-norm": ("PPO", _ROLLOUT | {"normalize_observations": True}, 64),
    "APO": ("APO", _ROLLOUT, 64),
}


def _obs_statistics(state):
    """The observation statistics the policy is queried with, if any."""
    if state.actor_state.obs_norm_info is not None:
        return state.actor_state.obs_norm_info
    return getattr(state.collector_state.env_state, "normalization_info", None)


@pytest.mark.parametrize("case", list(AGENTS))
def test_every_continuous_agent_trains_a_pid_policy(case):
    """A short run moves the gains off zero and the actor stays the PID.
    Where observations are normalised, the terms' statistics stay pinned
    (raw terms) and the plant's are learned."""
    name, overrides, steps = AGENTS[case]
    env, params = setpoint_env.make()
    cls = getattr(importlib.import_module(f"ajax.agents.{name}.{name}"), name)
    agent = cls(
        env,
        env_params=params,
        n_envs=1,
        actor_architecture=PIDPolicy(),
        critic_architecture=("8", "relu"),
        **overrides,
    )
    state, _ = agent.train(seed=[0], n_timesteps=steps, num_episode_test=2)
    gains = pid_gains(state.actor_state.params)
    assert gains["kp"].shape == (1, 1, 1)
    assert float(jnp.abs(jnp.stack(list(gains.values()))).sum()) > 0
    stats = _obs_statistics(state)
    assert (stats is not None) == (case in ("SAC-norm", "AVG", "PPO-norm"))
    if stats is not None:
        stats = getattr(stats, "obs", stats)
        mean, var = np.asarray(stats.mean), np.asarray(stats.var)
        assert np.all(mean[..., :3] == 0.0) and np.all(var[..., :3] == 1.0)
        assert np.all(var[..., 3] != 1.0)


def test_the_gains_are_normalised_on_a_wider_box():
    """On Box(-2, 2) a SAC PID actor's deterministic control tanh(u) is
    executed as 2 tanh(u): pid_gains reads normalised gains, in env units
    twice them (small-signal)."""
    env, params = paid_action.make()
    agent = SAC(env, env_params=params, n_envs=1, actor_architecture=PIDPolicy())
    state, _ = agent.train(seed=0, n_timesteps=0)
    actor = jax.tree.map(lambda x: x[0], state.actor_state)
    gains = {
        "kp": jnp.array([[0.4]]),
        "ki": jnp.array([[0.2]]),
        "kd": jnp.array([[0.1]]),
    }
    actor = actor.replace(params=_set_gains(actor.params, gains))
    for name, value in pid_gains(actor.params).items():
        np.testing.assert_array_equal(value, gains[name])
    e, i, d = paid_action.OBS
    u = 0.4 * e + 0.2 * i + 0.1 * d
    paid = evaluate(env, actor, 2, jax.random.PRNGKey(0), params)[0]
    np.testing.assert_allclose(paid, paid_action.HIGH * jnp.tanh(u), rtol=1e-6)
