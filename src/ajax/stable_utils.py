import jax.numpy as jnp
from target_gym.interpolator import get_interpolator


def get_expert_policy(env, env_params):
    env_class = env.__class__
    interpolator = get_interpolator(env_class, env_params)

    def expert_policy(obs):
        target = obs[..., 6]
        power = interpolator(target)[..., None]
        return jnp.concatenate([power, jnp.zeros_like(power)], axis=-1)

    return expert_policy
