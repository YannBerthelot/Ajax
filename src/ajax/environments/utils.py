from typing import Optional, Tuple

import jax.numpy as jnp
from ajax.types import BraxEnv, EnvType, GymnaxEnv
from gymnax import EnvParams


def check_if_environment_has_continuous_actions(
    env: EnvType, env_params: Optional[EnvParams] = None
) -> bool:
    env = get_raw_env(env)
    if check_env_is_brax(env):
        return True
    return not "discrete" in str(env.action_space(env_params)).lower()


def get_action_dim(env: EnvType, env_params: Optional[EnvParams] = None) -> int:
    """
    Get the action dimension (continuous case) or the number of action (discrete case) of the environment.
    """
    env = get_raw_env(env)
    if check_env_is_brax(env):
        return env.action_size

    return (
        env.action_space(env_params).n
        if not check_if_environment_has_continuous_actions(env)
        else env.action_space(env_params).shape[0]
    )


def get_state_action_shapes(
    env: EnvType, env_params: EnvParams = None
) -> Tuple[tuple, tuple]:
    """
    Returns the (obs_shape, action_shape) of a gymnax, brax, or gymnasium environment.

    - Discrete action spaces return shape (1,)
    - Shapes are returned as `tuple`s (not multiplied)
    - Works for wrapped environments
    """
    # Gymnax
    if check_env_is_gymnax(env):
        obs_space = env.observation_space(env_params)
        act_space = env.action_space(env_params)
        obs_shape = obs_space.shape
        # Discrete check via duck typing
        if hasattr(act_space, "n") and isinstance(act_space.n, int):
            action_shape = (1,)
        else:
            action_shape = act_space.shape
        return obs_shape, action_shape

    # Brax
    elif check_env_is_brax(env):
        obs_shape = (env.observation_size,)
        action_shape = (env.action_size,)
        return obs_shape, action_shape

    else:
        raise ValueError(f"Unsupported environment type: {type(env)}")


def get_raw_env(env: EnvType) -> EnvType:
    """
    Get the raw environment from the given environment.
    """
    if hasattr(env, "_env"):
        return env._env
    return env


def check_env_is_brax(env) -> bool:
    return isinstance(env, BraxEnv) or "brax" in str(type(env)).lower()


def check_env_is_gymnax(env) -> bool:
    return isinstance(env, GymnaxEnv) or "gymnax" in str(type(env)).lower()


def get_env_type(env: EnvType) -> str:
    """
    Get the type of the environment.
    """
    env = get_raw_env(env)
    if check_env_is_brax(env):
        return "brax"
    elif check_env_is_gymnax(env):
        return "gymnax"
    else:
        raise ValueError(f"Unsupported env type: {type(env)}")
