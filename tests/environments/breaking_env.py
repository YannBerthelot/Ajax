"""A stand-in for a simulator that diverges (tests/environments/test_non_finite_steps.py)."""

import jax.numpy as jnp
from brax import envs as brax_envs
from brax.envs.base import Env, State

BREAKS_AT = 3  # the step of an episode that diverges
ON_POSITIVE_ACTIONS = "non_finite_on_positive_actions"
ALWAYS = "non_finite_always"


class Breaks(Env):
    """Observes its step count and rewards 1, until step BREAKS_AT of an
    episode leaves the state, observation, reward and metric NaN. The env
    itself never says done, so only the guard can end the episode."""

    def __init__(self, threshold: float = -jnp.inf):
        self.threshold = threshold

    def reset(self, rng):
        t = jnp.zeros(1)
        return State(t, t, jnp.zeros(()), jnp.zeros(()), {"m": jnp.zeros(())}, {})

    def step(self, state, action):
        t = state.pipeline_state + 1
        broken = (t[0] == BREAKS_AT) & (action[0] > self.threshold)
        t = jnp.where(broken, jnp.nan, t)
        reward = jnp.where(broken, jnp.nan, 1.0)
        return state.replace(
            pipeline_state=t, obs=t, reward=reward, metrics={"m": t[0]}
        )

    observation_size = 1
    action_size = 1
    backend = "breaks"


brax_envs.register_environment(ON_POSITIVE_ACTIONS, lambda: Breaks(threshold=0.0))
brax_envs.register_environment(ALWAYS, Breaks)
