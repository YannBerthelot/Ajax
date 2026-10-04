"""Fixed-length gymnax toy envs for the TD-MPC2 agent tests.

TD-MPC2 (paper era) needs fixed-length, non-terminating episodes of at least
``horizon + 1`` rows, which the stock probing envs (episodes of 1-2 steps)
are not (``docs/world_models/DESIGN.md`` §8). These envs are:

* :class:`CounterEnv`: deterministic, ``obs = [t, x]`` with ``x`` the sum of
  past actions and ``reward = (t + 1) a``; every quantity is recoverable from
  the action sequence, for the replay / control-flow parity tests.
* :class:`ConstantRewardEnv`: constant observation, reward 1; with the
  time-limit bootstrap the value is ``sum_t gamma^t = 1 / (1 - gamma)``.
* :class:`RewardIsActionEnv`: constant observation, reward ``a``; the best
  action is +1.
* :class:`TerminatingEnv`: terminates after ``terminate_at`` steps, before
  its time limit (refused by TD-MPC2).
* :class:`TargetEnv`: ``obs_dim``-dim observations and ``action_dim``-dim
  actions (the multi-task tests' tasks of different dims); a random initial
  state, ``x' = 0.9 x + 0.1 mean(a)`` and reward ``1 - mean((a - target)^2)``,
  so the best action is ``target`` on every dim.

Actions are in ``[-1, 1]`` (1-D except :class:`TargetEnv`); episodes last
``max_steps_in_episode`` steps.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class ToyState(environment.EnvState):
    time: int
    x: jax.Array


@struct.dataclass
class ToyParams(environment.EnvParams):
    max_steps_in_episode: int = 10


class _ToyEnv(environment.Environment[ToyState, ToyParams]):
    """Shared plumbing; subclasses define ``_obs``, ``_reward`` and
    ``_terminated``."""

    def __init__(self, length: int = 10):
        super().__init__()
        self.length = length

    @property
    def default_params(self) -> ToyParams:
        return ToyParams(max_steps_in_episode=self.length)

    def _obs(self, state: ToyState) -> jax.Array:
        return jnp.ones((1,), jnp.float32)

    def _reward(self, state: ToyState, action: jax.Array) -> jax.Array:
        raise NotImplementedError

    def _terminated(self, state: ToyState) -> jax.Array:
        return jnp.array(False)

    def step_env(
        self, key: jax.Array, state: ToyState, action: Any, params: ToyParams
    ) -> tuple[jax.Array, ToyState, jax.Array, jax.Array, dict]:
        a = jnp.clip(jnp.asarray(action, jnp.float32).reshape(()), -1.0, 1.0)
        reward = self._reward(state, a)
        state = ToyState(time=state.time + 1, x=state.x + a)
        return self._obs(state), state, reward, self._terminated(state), {}

    def reset_env(
        self, key: jax.Array, params: ToyParams
    ) -> tuple[jax.Array, ToyState]:
        state = ToyState(time=0, x=jnp.zeros((), jnp.float32))
        return self._obs(state), state

    def is_terminated(self, state: ToyState, params: ToyParams) -> jax.Array:
        return self._terminated(state)

    @property
    def num_actions(self) -> int:
        return 1

    def action_space(self, params: ToyParams | None = None) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=jnp.float32)

    def observation_space(self, params: ToyParams) -> spaces.Box:
        size = self._obs(ToyState(time=0, x=jnp.zeros(()))).shape
        return spaces.Box(-jnp.inf, jnp.inf, size, jnp.float32)


class CounterEnv(_ToyEnv):
    def _obs(self, state: ToyState) -> jax.Array:
        return jnp.stack([jnp.asarray(state.time, jnp.float32), state.x])

    def _reward(self, state: ToyState, action: jax.Array) -> jax.Array:
        return (state.time + 1) * action


class ConstantRewardEnv(_ToyEnv):
    def _reward(self, state: ToyState, action: jax.Array) -> jax.Array:
        return jnp.ones((), jnp.float32)


class RewardIsActionEnv(_ToyEnv):
    def _reward(self, state: ToyState, action: jax.Array) -> jax.Array:
        return action


class TerminatingEnv(ConstantRewardEnv):
    def __init__(self, length: int = 10, terminate_at: int = 3):
        super().__init__(length)
        self.terminate_at = terminate_at

    def _terminated(self, state: ToyState) -> jax.Array:
        return state.time >= self.terminate_at


class TargetEnv(_ToyEnv):
    """``obs_dim`` observations, ``action_dim`` actions, reward peaked at
    ``target`` (module docstring)."""

    def __init__(
        self, obs_dim: int, action_dim: int, length: int = 10, target: float = 0.5
    ):
        super().__init__(length)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.target = target

    def _obs(self, state: ToyState) -> jax.Array:
        return jnp.broadcast_to(state.x, (self.obs_dim,)).astype(jnp.float32)

    def step_env(
        self, key: jax.Array, state: ToyState, action: Any, params: ToyParams
    ) -> tuple[jax.Array, ToyState, jax.Array, jax.Array, dict]:
        a = jnp.clip(jnp.asarray(action, jnp.float32).reshape(-1), -1.0, 1.0)
        reward = 1.0 - jnp.mean(jnp.square(a - self.target))
        x = 0.9 * state.x + 0.1 * jnp.mean(a)
        state = ToyState(time=state.time + 1, x=x)
        return self._obs(state), state, reward, jnp.array(False), {}

    def reset_env(
        self, key: jax.Array, params: ToyParams
    ) -> tuple[jax.Array, ToyState]:
        x = jax.random.uniform(key, (self.obs_dim,), minval=-1.0, maxval=1.0)
        state = ToyState(time=0, x=x)
        return self._obs(state), state

    @property
    def num_actions(self) -> int:
        return self.action_dim

    def action_space(self, params: ToyParams | None = None) -> spaces.Box:
        return spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=jnp.float32
        )

    def observation_space(self, params: ToyParams) -> spaces.Box:
        return spaces.Box(-jnp.inf, jnp.inf, (self.obs_dim,), jnp.float32)
