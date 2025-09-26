import jax.numpy as jnp
from target_gym.interpolator import get_interpolator


def get_expert_policy(target, env, env_params):
    interpolator = get_interpolator(env, env_params)
    power = interpolator(target)

    def expert_policy(x):
        return jnp.array([power, 0])

    return expert_policy
