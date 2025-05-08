"""Wrappers for environment"""

from functools import partial
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs import Env as BraxEnv
from brax.envs.base import State
from brax.envs.base import Wrapper as BraxWrapper
from flax import struct
from flax.serialization import to_state_dict
from gymnasium import core
from gymnasium import spaces as gymnasium_spaces
from gymnax.environments import environment, spaces

from ajax.utils import online_normalize


class GymnaxWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env: environment.Environment):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def observation_space(self, params) -> spaces.Box:
        """Get the observation space from a gymnax env given its params"""
        assert isinstance(
            self._env.observation_space(params),
            spaces.Box,
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState]:
        """Reset the environment and flatten the observation"""
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: float,
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        """Step the environment and flatten the observation"""
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    """Logging buffer"""

    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState]:
        """Reset the environment and log the state of the env"""
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)  # type: ignore[call-arg]
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: float,
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        """Step the environment and log the env state, episode return, episode length and timestep"""
        obs, env_state, reward, done, info = self._env.step(
            key,
            state.env_state,
            action,
            params,
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(  # type: ignore[call-arg]
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class ClipAction(GymnaxWrapper):
    """Continus action clipping wrapper"""

    def __init__(self, env, low=-1.0, high=1.0):
        """Set the high and low bounds"""
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """Step the environment while clipping the action first"""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class ClipActionBrax(BraxWrapper):
    """Continus action clipping wrapper"""

    def __init__(self, env, low=-1.0, high=1.0):
        """Set the high and low bounds"""
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, state, action):
        """Step the environment while clipping the action first"""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self.env.step(state, action)


class TransformObservation(GymnaxWrapper):
    """Observation modifying wrapper"""

    def __init__(self, env, transform_obs):
        """Set the observation transformation"""
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        """Reset the env and return the transformed obs"""
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        """Step the env and return the transformed obs"""
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    """Reward modifying wrapper"""

    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        """Step the env and return the transformed reward"""
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    """Vectorized an environment by vectorizing step and reset"""

    def __init__(self, env):
        """Override reset and step"""
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


# @struct.dataclass
# class NormalizeVecObsEnvState:
#     """Carry variables necessary for online normalization"""

#     mean: jnp.ndarray
#     var: jnp.ndarray
#     count: float
#     env_state: State


# @struct.dataclass
# class NormalizeVecObsEnvStateBrax:
#     """Carry variables necessary for online normalization"""

#     mean: jnp.ndarray
#     var: jnp.ndarray
#     count: float
#     pipeline_state: Optional[State]
#     obs: jax.Array
#     reward: jax.Array
#     done: jax.Array
#     metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
#     info: Dict[str, Any] = struct.field(default_factory=dict)


class NormalizeVecObservationBrax(BraxWrapper):
    """Wrapper for online normalization of observations"""

    def __init__(self, env: BraxEnv):
        super().__init__(env)

    def reset(self, key):
        state = self.env.reset(key)

        # Initialize normalization stats
        count = jnp.zeros((state.obs.shape[0], 1))
        mean = jnp.zeros_like(state.obs)
        mean_2 = jnp.zeros_like(state.obs)

        # Normalize
        obs, count, mean, mean_2, std = online_normalize(state.obs, count, mean, mean_2)

        # Replace state immutably
        state = state.replace(
            obs=obs,
            info={
                "count": count,
                "mean": mean,
                "mean_2": mean_2,
                "std": std,
                **state.info,
            },
        )
        return state

    def step(self, state, action):
        count = state.info["count"]
        mean = state.info["mean"]
        mean_2 = state.info["mean_2"]

        state = self.env.step(state, action)

        obs, count, mean, mean_2, std = online_normalize(state.obs, count, mean, mean_2)

        # Replace the whole info dict immutably
        state = state.replace(
            obs=obs,
            info={
                "count": count,
                "mean": mean,
                "mean_2": mean_2,
                "std": std,
                **state.info,
            },
        )
        return state


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        """Set gamma"""
        super().__init__(env)
        BaseState = env.reset(key=jax.random.PRNGKey(0))[1].__class__

        @struct.dataclass
        class NormalizedEnvState(BaseState):  # Inherit from the actual env_state class
            count: jnp.ndarray
            mean: jnp.ndarray
            mean_2: jnp.ndarray
            std: jnp.ndarray

        self.state_class = NormalizedEnvState

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)

        # Initialize normalization state
        count = jnp.array(0)
        mean = jnp.zeros_like(obs)
        mean_2 = jnp.zeros_like(obs)
        std = jnp.ones_like(obs)

        # Normalize obs
        obs, count, mean, mean_2, std = online_normalize(obs, count, mean, mean_2)
        state = self.state_class(
            **to_state_dict(env_state),
            count=count,
            mean=mean,
            mean_2=mean_2,
            std=std,
        )

        return obs, state

    def step(self, key, state, action, params=None):
        # Extract normalization state
        count, mean, mean_2 = state.count, state.mean, state.mean_2

        # Step through env
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)

        # Normalize observation
        obs, count, mean, mean_2, std = online_normalize(obs, count, mean, mean_2)

        # Repack new state
        state = self.state_class(
            **to_state_dict(env_state), count=count, mean=mean, mean_2=mean_2, std=std
        )
        return obs, state, reward, done, info


@struct.dataclass
class NormalizeVecRewEnvState:
    """Carry variables necessary for online normalization"""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: State


@struct.dataclass
class NormalizeVecRewEnvStateBrax:
    """Carry variables necessary for online normalization"""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    pipeline_state: Optional[State]
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict)


@jax.jit
def _return_original_reward(reward):
    return reward


@jax.jit
def _normalize_reward(reward, var):
    return reward / jnp.sqrt(var + 1e-8)


@jax.jit
def normalize_reward(reward, var):
    return jax.lax.cond(
        jnp.abs(var) < 1e-3,
        _return_original_reward,
        lambda x: _normalize_reward(x, var),
        operand=reward,
    )


class NormalizeVecReward(GymnaxWrapper):
    """Wrapper for online normalization of rewards"""

    def __init__(self, env, gamma):
        """Set gamma"""
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        """Reset the environment and return the normalized reward"""
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        """Step the environment and return the normalized reward"""
        obs, env_state, reward, done, info = self._env.step(
            key,
            state.env_state,
            action,
            params,
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        new_reward = normalize_reward(reward, state.var)
        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, new_reward, done, info


class NormalizeVecRewardBrax(BraxWrapper):
    """Wrapper for online normalization of rewards"""

    def __init__(self, env, gamma):
        """Set gamma"""
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key):
        """Reset the environment and return the normalized reward"""
        env_state = self.env.reset(key)
        batch_count = env_state.obs.shape[0]

        state = NormalizeVecRewEnvStateBrax(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            reward=env_state.reward,
            done=env_state.done,
            metrics=env_state.metrics,
            pipeline_state=env_state.pipeline_state,
            info=env_state.info,
            obs=env_state.obs,
        )
        return state

    def step(self, wrapped_state, action):
        """Step the environment and return the normalized reward"""
        unwrapped_state = State(
            wrapped_state.pipeline_state,
            wrapped_state.obs,
            wrapped_state.reward,
            wrapped_state.done,
            wrapped_state.metrics,
            wrapped_state.info,
        )
        env_state = self.env.step(unwrapped_state, action)
        obs, reward, done = (
            env_state.obs,
            env_state.reward,
            env_state.done,
        )
        return_val = wrapped_state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - wrapped_state.mean
        tot_count = wrapped_state.count + batch_count

        new_mean = wrapped_state.mean + delta * batch_count / tot_count
        m_a = wrapped_state.var * wrapped_state.count
        m_b = batch_var * batch_count
        M2 = (
            m_a
            + m_b
            + jnp.square(delta) * wrapped_state.count * batch_count / tot_count
        )
        new_var = M2 / tot_count
        new_count = tot_count
        new_reward = normalize_reward(reward, wrapped_state.var)
        new_wrapped_state = NormalizeVecRewEnvStateBrax(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            reward=new_reward,
            done=env_state.done,
            metrics=env_state.metrics,
            pipeline_state=env_state.pipeline_state,
            info=env_state.info,
            obs=env_state.obs,
        )
        return new_wrapped_state


def get_wrappers(mode: str = "gymnax"):
    if mode == "gymnax":
        return ClipAction, NormalizeVecObservation, NormalizeVecReward
    return ClipActionBrax, NormalizeVecObservationBrax, NormalizeVecRewardBrax


def check_wrapped_env_has_autoreset(wrapped_env: BraxWrapper):
    if "AutoResetWrapper" in wrapped_env.__repr__():
        return True
    while "env" in dir(wrapped_env):
        return check_wrapped_env_has_autoreset(wrapped_env.env)
    return False


class BraxToGymnasium(BraxWrapper):
    def __init__(self, env: BraxEnv, seed: Optional[int] = None):
        super().__init__(env)
        assert not check_wrapped_env_has_autoreset(
            env
        ), "Environment should not autoreset"
        self.env = env
        env_name = str(env.unwrapped.__class__).split(".")[-1][:-2]
        self.metadata = {
            "name": env_name,
            "render_modes": ["human", "rgb_array"] if hasattr(env, "render") else [],
        }

        self.rng: chex.PRNGKey = jax.random.PRNGKey(0)  # Placeholder
        self._seed(seed)

    @property
    def action_space(self):
        """Dynamically adjust action space depending on params."""
        return gymnasium_spaces.Box(
            low=-1,
            high=1,
            shape=(self.env.action_size,),
        )

    @property
    def observation_space(self):
        """Dynamically adjust state space depending on params."""
        return gymnasium_spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(self.env.observation_size,)
        )

    def _seed(self, seed: Optional[int] = None):
        """Set RNG seed (or use 0)."""
        self.rng = jax.random.PRNGKey(seed or 0)

    def step(
        self, action: core.ActType
    ) -> Tuple[core.ObsType, float, bool, bool, Dict[Any, Any]]:
        """Step environment, follow new step API."""
        self.env_state = self.env.step(self.env_state, action)  # type: ignore[has-type]
        obsv, reward, done, info = (
            self.env_state.obs,
            self.env_state.reward,
            self.env_state.done,
            self.env_state.info,
        )
        return (
            obsv,
            float(reward.item()),
            bool(done.item()),
            bool(done.item()),
            info,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[Any] = None,  # dict
    ) -> Tuple[core.ObsType, Any]:  # dict]:
        """Reset environment, update parameters and seed if provided."""
        if seed is not None:
            self._seed(seed)
        self.rng, reset_key = jax.random.split(self.rng)
        self.env_state = self.env.reset(reset_key)
        return self.env_state.obs, {}

    def render(self, mode="human") -> None:
        """use underlying environment rendering if it exists, otherwise return None."""
        raise NotImplementedError


def identity(x):
    return x


def split(x):
    return jax.random.split(x)[0]


class AutoResetWrapper(BraxWrapper):
    """Automatically resets Brax envs that are done, sampling a new random seed for initialization at each reset. This seed is propagated through info["rng"]"""

    def __init__(self, env: BraxWrapper):
        super().__init__(env)
        self.n_envs = env.reset(jax.random.PRNGKey(0)).obs.shape[0]
        self.single_env = self.n_envs == 1

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["rng"] = (
            rng.reshape(1, -1) if self.single_env else jnp.tile(rng, (self.n_envs, 1))
        )
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        rng = state.info["rng"][0]
        new_rng = jax.lax.cond(
            state.done.any(), split, identity, rng
        )  # Only generate a new seed when at least one env is done
        new_init_state = self.reset(
            new_rng
        )  # If I am correct, as long as the seed is the same jax will use the cached result and not recompute reset, only recomputing for a new seed.
        state.info["rng"] = (
            new_rng.reshape(1, -1)
            if self.single_env
            else jnp.tile(new_rng, (self.n_envs, 1))
        )  # shape shenanigans to adapt to parallel environments, they are suboptimal at the moment as the seed is only copied to match the batch size, but only one is really used. TODO : Check if it works correcly on parallel environments : check that each one has a proper different reset.

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done,
            new_init_state.info["first_pipeline_state"],
            state.pipeline_state,
        )
        obs = where_done(new_init_state.info["first_obs"], state.obs)
        state = state.replace(pipeline_state=pipeline_state, obs=obs)
        return state
