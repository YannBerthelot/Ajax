import gymnax
import pytest
from brax.envs import create as create_brax_env

from ajax.environments.utils import (
    check_env_is_brax,
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
    get_action_dim,
    get_env_type,
    get_raw_env,
    get_state_action_shapes,
)


@pytest.mark.parametrize(
    "env,expected_continuous",
    [
        (gymnax.make("Pendulum-v1")[0], True),
    ],
)
def test_check_if_environment_has_continuous_actions(env, expected_continuous):
    assert check_if_environment_has_continuous_actions(env) == expected_continuous


def test_get_raw_env():
    env, _ = gymnax.make("CartPole-v1")
    raw_env = get_raw_env(env)
    assert raw_env is env


@pytest.mark.parametrize(
    "env,expected_is_brax",
    [
        (create_brax_env("fast"), True),
    ],
)
def test_check_env_is_brax(env, expected_is_brax):
    assert check_env_is_brax(env) == expected_is_brax


@pytest.mark.parametrize(
    "env,expected_is_gymnax",
    [
        (gymnax.make("CartPole-v1")[0], True),
    ],
)
def test_check_env_is_gymnax(env, expected_is_gymnax):
    assert check_env_is_gymnax(env) == expected_is_gymnax


@pytest.mark.parametrize(
    "env,expected_type",
    [
        (create_brax_env("fast"), "brax"),
        (gymnax.make("Pendulum-v1")[0], "gymnax"),
    ],
)
def test_get_env_type(env, expected_type):
    assert get_env_type(env) == expected_type

    with pytest.raises(ValueError):
        get_env_type("unsupported_env")


@pytest.mark.parametrize(
    "env,env_params,expected_action_dim",
    [
        (
            create_brax_env("fast"),
            None,
            1,
        ),  # Example Brax environment with action size 1
        (
            gymnax.make("CartPole-v1")[0],
            gymnax.make("CartPole-v1")[0].default_params,
            2,
        ),  # Discrete Gymnax environment with 2 actions
        (
            gymnax.make("Pendulum-v1")[0],
            gymnax.make("Pendulum-v1")[0].default_params,
            1,
        ),  # Continuous Gymnax environment with action dim 1
    ],
)
def test_action_dim(env, env_params, expected_action_dim):
    assert get_action_dim(env, env_params) == expected_action_dim


@pytest.mark.parametrize(
    "env,env_params,expected_obs_shape,expected_action_shape",
    [
        (
            gymnax.make("CartPole-v1")[0],
            gymnax.make("CartPole-v1")[0].default_params,
            (4,),  # Observation shape for CartPole-v1
            (1,),  # Discrete action space
        ),
        (
            gymnax.make("Pendulum-v1")[0],
            gymnax.make("Pendulum-v1")[0].default_params,
            (3,),  # Observation shape for Pendulum-v1
            (1,),  # Continuous action space
        ),
        (
            create_brax_env("fast"),
            None,
            (2,),  # Example observation size for Brax environment
            (1,),  # Example action size for Brax environment
        ),
    ],
)
def test_get_state_action_shapes(
    env, env_params, expected_obs_shape, expected_action_shape
):
    obs_shape, action_shape = get_state_action_shapes(env)
    assert obs_shape == expected_obs_shape
    assert action_shape == expected_action_shape
