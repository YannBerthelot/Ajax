import jax.numpy as jnp
from plane_env.runner import get_interpolator


def get_expert_policy(target_alt, stick=0):
    interpolator = get_interpolator(stick=0)
    power = interpolator(target_alt)

    def expert_policy(x):
        return jnp.array([power, stick])

    return expert_policy
