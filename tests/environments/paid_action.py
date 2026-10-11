"""A one-step task on Box(-2, 2) that pays the action it receives.

Every episode is one step from the observation ``OBS``, and the reward is
the action the env was stepped with: a return reads the executed action
back. ``OBS`` doubles as PID error terms ``(e, I, D)`` for a
:class:`~ajax.modules.pid_actor.PIDPolicy`. Transitions keep gradients, as
APG needs.
"""

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

OBS = jnp.array([1.0, 0.5, -0.5])
HIGH = 2.0  # the box is [-HIGH, HIGH]: centre 0, half-range HIGH


@struct.dataclass
class Params(environment.EnvParams):
    max_steps_in_episode: int = 1


class PaidAction(environment.Environment):
    supports_transition_gradients = True
    transition_gradients_enabled = True

    @property
    def default_params(self) -> Params:
        return Params()

    def reset_env(self, key, params):
        return OBS, environment.EnvState(time=0)

    def step_env(self, key, state, action, params):
        reward = jnp.reshape(action, (-1,))[0]
        return OBS, state.replace(time=state.time + 1), reward, jnp.array(True), {}

    def get_obs(self, state, params=None, key=None):
        return OBS

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, OBS.shape)

    def action_space(self, params=None):
        return spaces.Box(-HIGH, HIGH, (1,))


def make() -> tuple[PaidAction, Params]:
    return PaidAction(), Params()


def mapped(a: jax.Array) -> jax.Array:
    """What the env is paid for the agent action ``a``: ``[-1, 1]`` mapped
    onto the box."""
    return HIGH * jnp.clip(a, -1.0, 1.0)
