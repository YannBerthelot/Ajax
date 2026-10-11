"""Environment configs the agent tests share: one brax ``fast`` env, one
gymnax Pendulum env, and ``env_config``, which runs a test on each in turn.
CONTRIBUTING's Testing section describes how the tests are laid out."""

import gymnax
import pytest
from brax.envs import create as create_brax_env

from ajax.state import EnvironmentConfig


@pytest.fixture
def fast_env_config() -> EnvironmentConfig:
    """One brax ``fast`` env (continuous actions)."""
    env = create_brax_env("fast", batch_size=1)
    return EnvironmentConfig(env=env, env_params=None, n_envs=1, continuous=True)


@pytest.fixture
def gymnax_env_config() -> EnvironmentConfig:
    """One gymnax Pendulum env."""
    env, env_params = gymnax.make("Pendulum-v1")
    return EnvironmentConfig(env=env, env_params=env_params, n_envs=1, continuous=True)


@pytest.fixture(params=["fast_env_config", "gymnax_env_config"])
def env_config(request: pytest.FixtureRequest) -> EnvironmentConfig:
    """Each env mode in turn."""
    return request.getfixturevalue(request.param)
