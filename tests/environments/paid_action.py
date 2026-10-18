"""A task that pays the action it receives, on Box(-2, 2) or another box.

The reward is the action the env was stepped with, so a return reads the
executed action back. By default every episode is one terminating step from
the observation ``OBS``; with ``length``, episodes of that many steps end at
the time limit (fixed-length and non-terminating, as TD-MPC2 needs). With
``dim`` the box has that many dimensions; the first is paid.
``OBS`` doubles as PID error terms ``(e, I, D)`` for a
:class:`~ajax.modules.pid_actor.PIDPolicy`. Transitions keep gradients, as
APG needs.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from jax.typing import ArrayLike

OBS = jnp.array([1.0, 0.5, -0.5])
HIGH = 2.0  # the default box is [-HIGH, HIGH]: centre 0, half-range HIGH
BOX = (-HIGH, HIGH)


@struct.dataclass
class Params(environment.EnvParams):
    max_steps_in_episode: int = 1


class PaidAction(environment.Environment):
    supports_transition_gradients = True
    transition_gradients_enabled = True

    def __init__(
        self,
        box: tuple[float, float] = BOX,
        length: Optional[int] = None,
        dim: int = 1,
    ) -> None:
        super().__init__()
        self.box, self.length, self.dim = box, length, dim

    @property
    def default_params(self) -> Params:
        return Params(max_steps_in_episode=self.length or 1)

    def reset_env(self, key, params):
        return OBS, environment.EnvState(time=0)

    def step_env(self, key, state, action, params):
        reward = jnp.reshape(action, (-1,))[0]
        terminated = jnp.array(self.length is None)
        return OBS, state.replace(time=state.time + 1), reward, terminated, {}

    def get_obs(self, state, params=None, key=None):
        return OBS

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, OBS.shape)

    def action_space(self, params=None):
        return spaces.Box(*self.box, (self.dim,))


def make(
    box: tuple[float, float] = BOX, length: Optional[int] = None, dim: int = 1
) -> tuple[PaidAction, Params]:
    env = PaidAction(box, length, dim)
    return env, env.default_params


def mapped(a: ArrayLike, box: tuple[float, float] = BOX) -> jax.Array:
    """What the env is paid for the agent action ``a``: ``[-1, 1]`` mapped
    affinely onto ``box``."""
    low, high = box
    return (high + low) / 2 + (high - low) / 2 * jnp.clip(a, -1.0, 1.0)
