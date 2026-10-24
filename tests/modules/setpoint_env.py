"""A first-order set-point task whose best PID gains are known by hand.

The plant is an integrator, ``x_{t+1} = x_t + B u_t`` with ``u_t =
clip(a_t, -1, 1)``, from ``x_0 = 0`` toward a set-point ``r`` drawn in
``[-1, 1]`` per episode; the reward is ``-(r - x_{t+1})^2`` and episodes
last ``LENGTH`` steps. Observation ``[x, r]``; with the :data:`TERMS`
feature (``dt = 1``) it becomes ``[x, r, e, I, D]``.

The best PID is deadbeat, ``(Kp, Ki, Kd) = (1 / B, 0, 0)``: it zeroes the
error in one step and holds it there, so every reward is 0, the maximum,
and it is the only PID that does: with ``I_0 = e_0`` and ``D_0 = 0`` zero
errors force, step by step, ``Kp + Ki = 1 / B``, ``Ki = Kd``, ``Ki = 0``.
Exploration noise is independent of the past, so the deadbeat gains stay
the minimum-variance answer for a stochastic policy too. ``B = 4`` keeps
``|u| <= 1/4``, where ``tanh`` is linear to 2 % (SAC and TD3 squash).
"""

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from ajax.extensions.observe import PIDTermsInput

B = 4.0
LENGTH = 10
BEST_GAINS = {"kp": 1.0 / B, "ki": 0.0, "kd": 0.0}
#: The PID's terms of the error ``r - x``, for the agent's extensions.
TERMS = PIDTermsInput(current_idx=(0,), target_idx=(1,))


@struct.dataclass
class State(environment.EnvState):
    x: jax.Array
    r: jax.Array


@struct.dataclass
class Params(environment.EnvParams):
    max_steps_in_episode: int = LENGTH


class Setpoint(environment.Environment):
    @property
    def default_params(self) -> Params:
        return Params()

    def reset_env(self, key, params):
        r = jax.random.uniform(key, minval=-1.0, maxval=1.0)
        state = State(time=0, x=jnp.float32(0.0), r=r)
        return self.get_obs(state), state

    def step_env(self, key, state, action, params):
        u = jnp.clip(jnp.reshape(action, ()), -1.0, 1.0)
        state = state.replace(x=state.x + B * u, time=state.time + 1)
        reward = -((state.r - state.x) ** 2)
        return self.get_obs(state), state, reward, jnp.array(False), {}

    def get_obs(self, state, params=None, key=None):
        return jnp.stack([state.x, state.r])

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, (2,))

    def action_space(self, params):
        return spaces.Box(-1.0, 1.0, (1,))


def make() -> tuple[Setpoint, Params]:
    return Setpoint(), Params()
