"""A gymnax task whose rows say where they come from.

The observation is ``[t, c]``: ``t`` the step count, ``c`` drawn uniformly
in [-1, 1] at reset. An episode with ``c > 0`` terminates at ``t =
TERMINAL``, the others are truncated by the time limit at ``t = LIMIT``.
The reward is the executed action, on paid_action's Box(-2, 2) or on
Discrete(2), so a row's reward reads its own action back. With
``keep_raw`` every observation normaliser leaves [t, c] raw
(``raw_obs_columns``), so rows read the same whatever the agent normalises.
"""

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from tests.environments.paid_action import HIGH

TERMINAL, LIMIT = 3, 5


@struct.dataclass
class State(environment.EnvState):
    c: jax.Array


@struct.dataclass
class Params(environment.EnvParams):
    max_steps_in_episode: int = LIMIT


def ends(obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(terminated, truncated)`` of the step taken from ``obs``."""
    t, c = obs[..., 0] + 1, obs[..., 1]
    return (c > 0) & (t == TERMINAL), (c <= 0) & (t == LIMIT)


class IdentityEnv(environment.Environment):
    def __init__(self, discrete: bool = False, keep_raw: bool = False):
        self.discrete = discrete
        self.raw_obs_columns = 2 if keep_raw else 0

    @property
    def default_params(self) -> Params:
        return Params()

    def reset_env(self, key, params):
        state = State(time=0, c=jax.random.uniform(key, minval=-1.0, maxval=1.0))
        return self.get_obs(state), state

    def step_env(self, key, state, action, params):
        # A new state, not state.replace: the collector steps the training-
        # fraction view of the state, which replace would carry along.
        state = State(time=state.time + 1, c=state.c)
        reward = jnp.reshape(action, (-1,))[0].astype(jnp.float32)
        terminated = (state.c > 0) & (state.time >= TERMINAL)
        return self.get_obs(state), state, reward, terminated, {}

    def get_obs(self, state, params=None, key=None):
        return jnp.stack([state.time, state.c]).astype(jnp.float32)

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(-1.0, float(LIMIT), (2,), jnp.float32)

    def action_space(self, params=None):
        if self.discrete:
            return spaces.Discrete(2)
        return spaces.Box(-HIGH, HIGH, (1,), jnp.float32)
