import brax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.environments.interaction import collect_experience_from_expert_policy
from ajax.environments.system_class import broadcast_env_params
from ajax.environments.utils import ExpertInAgentUnits, agent_action_to_env
from ajax.evaluate import evaluate
from ajax.extensions.expert import (
    ImitationLoss,
    ResidualPolicy,
    reads_stored_expert_actions,
)
from ajax.extensions.target_mods import IBRL
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import EnvironmentConfig, Transition
from tests.environments import paid_action


@pytest.fixture(scope="session")
def gymnax_env():
    """Provide a simple Gymnax environment and default params."""
    env, env_params = gymnax.make("Pendulum-v1")
    return env, env_params


@pytest.fixture(scope="session")
def brax_env():
    """Provide a simple Brax environment (stateless)."""
    env = brax.envs.create("fast", batch_size=1)
    return env


@pytest.fixture
def gymnax_env_args(gymnax_env):
    """Wrap Gymnax env into EnvironmentConfig."""
    env, env_params = gymnax_env
    n_envs = 2
    return EnvironmentConfig(
        env=env,
        env_params=env_params,
        n_envs=n_envs,
        continuous=True,
    )


@pytest.fixture
def brax_env_args(brax_env):
    """Wrap Brax env into EnvironmentConfig."""
    n_envs = 1
    return EnvironmentConfig(
        env=brax_env,
        env_params=None,
        n_envs=n_envs,
        continuous=True,
    )


@pytest.mark.parametrize(
    "mode,env_args_fixture",
    [
        ("gymnax", "gymnax_env_args"),
        ("brax", "brax_env_args"),
    ],
)
def test_collect_experience_shapes_and_types(request, mode, env_args_fixture):
    """Check Transition shapes and dtypes for Gymnax and Brax."""
    env_args = request.getfixturevalue(env_args_fixture)
    rng = jax.random.PRNGKey(0)
    n_steps = 5

    action_dim = getattr(env_args.env, "action_size", 1)  # gymnax Pendulum: 1

    def expert_policy(obs):
        return jnp.zeros((action_dim), dtype=jnp.float32)

    transitions = collect_experience_from_expert_policy(
        expert_policy, rng, mode, env_args, n_timesteps=n_steps
    )

    assert isinstance(transitions, Transition)
    assert transitions.obs.shape[0] == n_steps
    assert transitions.action.shape[0] == n_steps
    assert transitions.reward.shape[0] == n_steps
    assert transitions.terminated.shape[0] == n_steps
    assert transitions.truncated.shape[0] == n_steps


# ---------------------------------------------------------------------------
# The expert door: experts speak env units, Ajax reads agent units
# ---------------------------------------------------------------------------


class _GainExpert:
    """A stateful gain expert: a P controller on the first observation, in
    env units, whose integral of that observation is its state."""

    learnable_fields = ("kp",)
    anchor_gains = jnp.array([3.0])

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, state, obs):
        return self.step_with_gains(state, obs, self.anchor_gains)

    def step_with_gains(self, state, obs, gains):
        return jnp.clip(-gains[0] * obs[..., :1], -2.0, 2.0), state + obs[..., :1]


def test_a_wrapped_expert_acts_in_agent_units():
    env, params = gymnax.make("Pendulum-v1")  # torque in [-2, 2]
    obs = jnp.zeros((2, 3))
    stateless = ExpertInAgentUnits.wrap(lambda o: jnp.full((2, 1), 1.5), env, params)
    np.testing.assert_array_equal(stateless(obs), [[0.75], [0.75]])
    assert not hasattr(stateless, "init_state")
    assert not hasattr(stateless, "step_with_gains")
    assert ExpertInAgentUnits.wrap(stateless, env, params) is stateless
    assert ExpertInAgentUnits.wrap(None, env, params) is None
    gain = _GainExpert()
    wrapped = ExpertInAgentUnits.wrap(gain, env, params)
    obs = jax.random.normal(jax.random.PRNGKey(0), (4, 3))
    action, state = wrapped(gain.init_state(4), obs)
    raw, raw_state = gain(gain.init_state(4), obs)
    np.testing.assert_array_equal(state, raw_state)
    np.testing.assert_array_equal(action, raw / 2.0)
    assert wrapped.init_state == gain.init_state
    assert wrapped.learnable_fields == ("kp",)
    assert wrapped.anchor_gains is gain.anchor_gains


class _EitherWayExpert:
    """A stateful expert also callable on observations alone, as the SAC
    equivalence tests' expert is: ``expert(obs)`` or ``expert(state, obs)``."""

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, *args):
        action = jnp.full((args[-1].shape[0], 1), 1.5)
        return action if len(args) == 1 else (action, args[0] + 1.0)


def test_the_call_not_init_state_says_what_the_expert_returned():
    env, params = gymnax.make("Pendulum-v1")
    wrapped = ExpertInAgentUnits(_EitherWayExpert(), env, params)
    obs = jnp.zeros((4, 3))
    np.testing.assert_array_equal(wrapped(obs), np.full((4, 1), 0.75))
    action, state = wrapped(jnp.zeros((4, 1)), obs)
    np.testing.assert_array_equal(action, np.full((4, 1), 0.75))
    np.testing.assert_array_equal(state, np.ones((4, 1)))


@pytest.mark.parametrize("env_id", ["Pendulum-v1", "MountainCarContinuous-v0"])
def test_the_gain_policy_executes_the_experts_control(env_id):
    """Mapped back at the env door, the converted control is the expert's
    (clipped to the box): to float precision on Pendulum, bitwise on a unit
    box."""
    env, params = gymnax.make(env_id)
    gain = _GainExpert()
    obs_dim = env.observation_space(params).shape[0]
    obs = jax.random.normal(jax.random.PRNGKey(1), (8, obs_dim))
    gains = jnp.array([0.7])
    raw, _ = gain.step_with_gains(gain.init_state(8), obs, gains)
    action, _ = ExpertInAgentUnits(gain, env, params).step_with_gains(
        gain.init_state(8), obs, gains
    )
    executed = agent_action_to_env(action, env, params)
    if env_id == "Pendulum-v1":
        np.testing.assert_allclose(executed, raw, atol=1e-6)
    else:
        np.testing.assert_array_equal(executed, np.clip(raw, -1.0, 1.0))


def test_per_env_params_give_the_expert_one_box():
    """Per-env params sharing one box give unbatched (A,) bounds: a
    stateless expert vmapped over single observations keeps its shape.
    Different boxes raise."""
    env, params = gymnax.make("Pendulum-v1")
    params = broadcast_env_params(params, 3)
    wrapped = ExpertInAgentUnits(lambda o: jnp.full((1,), 1.5), env, params)
    out = jax.vmap(wrapped)(jnp.zeros((3, 3)))
    assert out.shape == (3, 1)
    np.testing.assert_array_equal(out, np.full((3, 1), 0.75))
    params = params.replace(max_torque=jnp.array([1.0, 2.0, 2.0]))
    with pytest.raises(ValueError, match="different action bounds"):
        ExpertInAgentUnits(lambda o: o, env, params)


# ---------------------------------------------------------------------------
# Behind the doors: the agent reads its expert in agent units
# ---------------------------------------------------------------------------
_SAC = {
    "n_envs": 1,
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
    "buffer_size": 64,
    "batch_size": 4,
    "expert_buffer_n_steps": 0,
    "expert_fraction": 1.0,  # the expert acts on every warm-up step
}


def _constant(obs):
    """1.5 on the paid Box(-2, 2): 0.75 in agent units."""
    return jnp.full((*obs.shape[:-1], 1), 1.5)


def _sac(**kwargs):
    env, params = paid_action.make()
    kwargs = {**_SAC, "expert_policy": _constant, **kwargs}
    return SAC(env, env_params=params, **kwargs)


def _written(state):
    """The replay buffer's written rows of the run's only seed and env."""
    buffer_state = state.collector_state.buffer_state
    n = int(np.asarray(buffer_state.current_index).reshape(-1)[0])
    return {k: np.asarray(v)[0, 0, :n] for k, v in buffer_state.experience.items()}


def test_expert_rows_are_stored_in_agent_units_and_paid_in_env_units():
    """The expert prefill and the expert's warm-up steps store action =
    a_expert = next_a_expert = 0.75 and are paid the expert's 1.5."""
    agent = _sac(expert_buffer_n_steps=4, learning_starts=64)
    state, _ = agent.train(seed=0, n_timesteps=8)
    rows = _written(state)
    assert len(rows["reward"]) == 4 + 8
    for key in ("action", "a_expert", "next_a_expert"):
        np.testing.assert_array_equal(rows[key], 0.75, err_msg=key)
    np.testing.assert_array_equal(rows["reward"], 1.5)


def test_a_zero_scale_residual_policy_executes_the_expert():
    """ResidualPolicy(scale=0) acts clip(a_expert + 0 * a, -1, 1) = 0.75,
    paid 1.5 in collection and evaluate() (1.0 when it mixed env units);
    the expert's own rollout is paid 1.5, as raw stepping."""
    agent = _sac(learning_starts=4, extensions=(ResidualPolicy(scale=0.0),))
    logging = LoggingConfig(config={}, log_frequency=4, use_wandb=False)
    state, out = agent.train(
        seed=0, n_timesteps=12, num_episode_test=2, logging_config=logging
    )
    np.testing.assert_array_equal(_written(state)["reward"][4:], 1.5)  # warmed up
    evaluated = np.asarray(out["timestep"]) >= 0
    assert evaluated.any()
    for key in ("Eval/episodic mean reward", "Eval/episodic mean expert reward"):
        np.testing.assert_allclose(np.asarray(out[key])[evaluated], 1.5, err_msg=key)


def test_the_extensions_read_the_expert_in_agent_units():
    """IBRL and ImitationLoss call their expert in agent units, IBRL
    holding the agent's own (it reads the stored actions); SAC's
    eval_expert_policy is converted too. (The augmented observation's
    column: tests/extensions/test_expert_observations.py.)"""
    env, params = paid_action.make()
    sac = _sac(
        extensions=(IBRL(expert_policy=_constant),), eval_expert_policy=_constant
    )
    td3 = TD3(env, env_params=params, extensions=(ImitationLoss(_constant),))
    obs = paid_action.OBS[None]
    for owner in (sac, td3):
        (ext,) = owner.extension_stack.extensions
        np.testing.assert_array_equal(ext.expert_policy(obs), [[0.75]])
    (ibrl,) = sac.extension_stack.extensions
    assert ibrl.expert_policy is sac.expert_policy is sac.eval_expert_policy
    assert reads_stored_expert_actions(ibrl, {"expert_policy": sac.expert_policy})


def test_evaluate_reads_an_env_unit_expert_as_a_wrapped_one():
    """evaluate()'s expert stays in env units for a direct caller: a raw and
    a wrapped expert give the same evaluation, the expert paid 1.5."""
    env, params = paid_action.make()
    state, _ = _sac().train(seed=0, n_timesteps=0)
    actor = jax.tree.map(lambda x: x[0], state.actor_state)
    key = jax.random.PRNGKey(2)
    wrapped = ExpertInAgentUnits(_constant, env, params)
    out = [
        evaluate(env, actor, 2, key, params, expert_policy=e)
        for e in (_constant, wrapped)
    ]
    for a, b in zip(*out, strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(out[0][5], 1.5)
