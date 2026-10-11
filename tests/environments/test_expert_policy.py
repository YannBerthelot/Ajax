import brax
import gymnax
import jax
import jax.numpy as jnp
import pytest

from ajax.environments.interaction import collect_experience_from_expert_policy
from ajax.state import EnvironmentConfig, Transition


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
