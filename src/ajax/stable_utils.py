from functools import partial

import jax
import jax.numpy as jnp
from target_gym.interpolator import get_interpolator


def expert_policy(obs, interpolator):
    target = obs[..., 6]
    power = interpolator(target)[..., None]
    return jnp.concatenate([power, jnp.zeros_like(power)], axis=-1)


def get_expert_policy(env, env_params):
    env_class = env.__class__
    interpolator = jax.jit(get_interpolator(env_class, env_params))

    return partial(expert_policy, interpolator=interpolator)
