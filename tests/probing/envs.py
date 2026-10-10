"""One environment builder: a small episodic MDP from its transition.

``Spec.transition(state, action, key)`` returns the next state, the reward
and the termination flag. ``SpecEnv`` adds gymnax's auto-reset, the time
limit (also the evaluation's scan length), the spaces and, on request, a
record of the executed actions and of one draw from each step key, kept
across auto-resets. The state is flat: the observation normaliser rebuilds
it from ``to_state_dict``. The package probe envs stay package subclasses,
so their random draws do not change.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import brax.envs
import jax
import jax.numpy as jnp
from brax.envs.base import Env as BraxEnv
from brax.envs.base import State as BraxState
from flax import struct
from gymnax.environments import environment, spaces
from probing_environments.gymnax_envs import continuous_actions as continuous

F32 = jnp.float32


@struct.dataclass
class State(environment.EnvState):
    """``s``: the hidden step or state index; ``c``: a per-episode context
    or carried value; ``flag``: set by a reset wrapper; the rest: the record."""

    s: jax.Array
    c: jax.Array
    flag: jax.Array
    clock: jax.Array
    first_a: jax.Array
    first_u: jax.Array
    last_a: jax.Array
    last_u: jax.Array


@struct.dataclass
class Params(environment.EnvParams):
    max_steps_in_episode: int = 10_000
    start: float = 0.0


def one_hot(i: Any, n: int) -> jax.Array:
    return jax.nn.one_hot(i, n, dtype=F32)


def index_obs(st: State) -> jax.Array:
    return st.s.astype(F32).reshape(1)


def at_start(key: jax.Array, params: Params) -> tuple[Any, Any]:
    return 0, 0.0


@dataclasses.dataclass(frozen=True)
class Spec:
    """``actions`` n > 0 declares Discrete(n), 0 a Box ``box`` = (low, high,
    dim); ``limit`` is the time limit; ``record`` = (first, last) keeps the
    run's first and last steps; ``gradients`` opts into gymnax's transition
    gradients."""

    name: str
    transition: Callable[[State, jax.Array, Any], tuple[State, Any, Any]]
    obs: Callable[[State], jax.Array] = index_obs
    reset: Callable[[jax.Array, Params], tuple[Any, Any]] = at_start
    obs_dim: int = 1
    obs_box: tuple[float, float] = (0.0, 1.0)
    actions: int = 0
    box: tuple[float, float, int] = (-1.0, 1.0, 1)
    limit: int = 10_000
    record: tuple[int, int] = (0, 0)
    gradients: bool = False

    def fresh(self, s: Any = 0, c: Any = 0.0) -> State:
        first, last = self.record
        zeros = (jnp.zeros(n, F32) for n in (first, first, last, last))
        flat = (jnp.asarray(s, jnp.int32), jnp.asarray(c, F32), jnp.asarray(0.0, F32))
        return State(jnp.int32(0), *flat, jnp.int32(0), *zeros)

    def make(self, **params: Any) -> tuple[SpecEnv, Params]:
        env = SpecEnv(self)
        return env, env.default_params.replace(**params)


class SpecEnv(environment.Environment):
    def __init__(self, spec: Spec) -> None:
        self.spec = spec
        self._supports_transition_gradients = spec.gradients

    @property
    def default_params(self) -> Params:
        return Params(max_steps_in_episode=self.spec.limit)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def num_actions(self) -> int:
        return self.spec.actions or self.spec.box[2]

    def action_space(self, params: Any = None) -> Any:
        if self.spec.actions:
            return spaces.Discrete(self.spec.actions)
        low, high, dim = self.spec.box
        return spaces.Box(low, high, (dim,), dtype=F32)

    def observation_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(*self.spec.obs_box, (self.spec.obs_dim,), dtype=F32)

    def get_obs(self, state: State, params: Any = None, key: Any = None) -> jax.Array:
        return self.spec.obs(state).astype(F32)

    def reset_env(self, key: jax.Array, params: Params) -> tuple[jax.Array, State]:
        state = self.spec.fresh(*self.spec.reset(key, params))
        return self.get_obs(state), state

    def step_env(self, key: jax.Array, state: State, action: Any, params: Any) -> Any:
        if self.spec.actions:
            action = jnp.reshape(action, ()).astype(jnp.int32)
        state, reward, terminated = self.spec.transition(state, action, key)
        state = state.replace(time=state.time + 1)
        obs, state = self._apply_transition_gradient_policy(self.get_obs(state), state)
        return obs, state, jnp.asarray(reward, F32), jnp.asarray(terminated), {}

    def step(self, key: Any, state: Any, action: Any, params: Any = None) -> Any:
        """gymnax's step, then the record, written from the pre-step state
        so the auto-reset cannot wipe it."""
        out = super().step(key, state, action, params)
        (first, last), c = self.spec.record, state.clock
        if not first and not last:
            return out
        u, a = jax.random.uniform(key), jnp.asarray(action, F32).reshape(-1)[0]
        rec = {"clock": c + 1}
        if first:
            i, keep = jnp.minimum(c, first - 1), c < first
            for name, x in (("first_a", a), ("first_u", u)):
                old = getattr(state, name)
                rec[name] = old.at[i].set(jnp.where(keep, x, old[i]))
        if last:
            rec |= {"last_a": state.last_a.at[c % last].set(a)}
            rec |= {"last_u": state.last_u.at[c % last].set(u)}
        return out[0], out[1].replace(**rec), *out[2:]


def register_brax(spec: Spec, env_id: str) -> str:
    """The spec as a brax env: the state rides in ``pipeline_state``, which
    Ajax's AutoResetWrapper resets; the limit is the EpisodeWrapper's."""

    class Twin(BraxEnv):
        def reset(self, rng: jax.Array) -> BraxState:
            ps = spec.fresh(*spec.reset(rng, Params()))
            zero = jnp.zeros(())
            return BraxState(ps, spec.obs(ps).astype(F32), zero, zero)

        def step(self, state: BraxState, action: jax.Array) -> BraxState:
            ps, reward, terminated = spec.transition(state.pipeline_state, action, None)
            obs, done = spec.obs(ps).astype(F32), jnp.asarray(terminated, F32)
            reward = jnp.asarray(reward, F32)
            return state.replace(pipeline_state=ps, obs=obs, reward=reward, done=done)

        observation_size = property(lambda self: spec.obs_dim)
        action_size = property(lambda self: spec.box[2])
        backend = property(lambda self: "generalized")

    brax.envs.register_environment(env_id, Twin)
    return env_id


def package(env_cls: type) -> tuple[Any, Any]:
    """A package probe env, the time-limit clause kept out (every episode
    ends by termination)."""
    env = env_cls()
    return env, env.default_params.replace(max_steps_in_episode=10_000)


def symmetric(env_cls: type) -> type:
    """Declare and enforce the [-1, 1] range the package's continuous
    policy probes reward (they declare Box(0, 1) and never clip)."""

    class Symmetric(env_cls):  # type: ignore[misc, valid-type]
        def action_space(self, params: Any = None) -> spaces.Box:
            return spaces.Box(-1.0, 1.0, (1,), dtype=F32)

        def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
            return super().step_env(key, state, jnp.clip(action, -1.0, 1.0), params)

    Symmetric.__name__ = env_cls.__name__
    return Symmetric


class SignedActionEnv(continuous.PolicyAndValueEnv):
    """s in {-1, +1}, one step, reward clip(a, -1, 1) s: unlike the
    package's sign-only reward, a gradient everywhere."""

    def action_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (1,), dtype=F32)

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
        reward = jnp.clip(jnp.squeeze(action), -1.0, 1.0) * state.x
        state = type(state)(x=state.x, time=state.time + 1)
        obs, done = self.get_obs(state), self.is_terminated(state, params)
        info = {"discount": self.discount(state, params)}
        return *jax.lax.stop_gradient((obs, state)), reward, done, info
