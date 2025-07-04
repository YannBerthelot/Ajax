"""Wrappers for environment"""

# ruff: noqa: C901
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

from ajax.environments.utils import check_env_is_brax, get_state_action_shapes
from ajax.types import EnvNormalizationInfo, NormalizationInfo
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


def init_norm_info(
    batch_size: int, obs_shape: tuple, returns: bool = False
) -> NormalizationInfo:
    count = jnp.zeros((batch_size, 1))
    mean = jnp.zeros((batch_size, *obs_shape))
    mean_2 = jnp.zeros((batch_size, *obs_shape))
    var = jnp.zeros((batch_size, *obs_shape))
    return NormalizationInfo(
        var,
        count,
        mean,
        mean_2,
        returns=jnp.zeros((batch_size, 1)) if returns else None,
    )


@partial(jax.jit, static_argnames="mode")
def get_obs_from_state(state: State | Tuple, mode: str) -> jax.Array:
    if mode == "gymnax" and isinstance(state, tuple):
        match state:
            case (obs, _):
                return obs
            case (obs, _, _, _, _):
                return obs
    elif mode == "brax" and isinstance(state, State):
        return state.obs


@partial(jax.jit, static_argnames="mode")
def get_obs_and_reward_and_done_from_state(
    state: State | Tuple, mode: str
) -> jax.Array:
    if mode == "gymnax" and isinstance(state, tuple):
        return state[0], state[2], state[3]
    elif mode == "brax" and isinstance(state, State):
        return state.obs, state.reward, state.done


def normalize_wrapper_factory(
    mode,
):
    Base = BraxWrapper if mode == "brax" else GymnaxWrapper

    class NormalizeVecObservation(Base):
        """Wrapper for online normalization of observations and rewards"""

        def __init__(
            self,
            env: BraxEnv,
            train: bool = True,
            norm_info: Optional[EnvNormalizationInfo] = None,
            normalize_obs: bool = True,
            normalize_reward: bool = True,
            gamma: Optional[float] = None,
        ):
            self.mode = "brax" if check_env_is_brax(env) else "gymnax"
            super().__init__(env)

            self.obs_shape, _ = get_state_action_shapes(env)

            self.train = train
            self.normalize_obs = normalize_obs
            self.normalize_reward = normalize_reward
            if self.normalize_reward:
                assert (
                    gamma is not None
                ), "Gamma must be provided for reward normalization."
            self.norm_info = norm_info
            self.gamma = gamma
            rng = jax.random.PRNGKey(0)
            dummy_obs = get_obs_from_state(env.reset(rng), mode=self.mode)

            self.batch_size = dummy_obs.shape[0] if jnp.ndim(dummy_obs) > 1 else 1

            if mode == "gymnax":
                # if self.mode == "gymnax":
                BaseState = env.reset(key=jax.random.PRNGKey(0))[1].__class__

                @struct.dataclass
                class NormalizedEnvState(BaseState):  # type: ignore[valid-type]
                    # Inherit from the actual env_state class
                    normalization_info: NormalizationInfo

                self.state_class = NormalizedEnvState

        @partial(jax.jit, static_argnames=("self", "mode"))
        def update_state_reset(
            self, state: State | Tuple, obs: jax.Array, norm_info, mode: str
        ):
            """Helper function to update state immutably"""
            if mode == "brax" and isinstance(state, State):
                return state.replace(
                    obs=obs,
                    info={
                        **state.info,
                        "normalization_info": norm_info,
                    },
                )
            else:
                _, env_state = state
                env_state = self.state_class(
                    **to_state_dict(env_state), normalization_info=norm_info
                )
                return obs, env_state

        @partial(jax.jit, static_argnames=("mode", "self"))
        def update_state_step(
            self,
            state: State | Tuple,
            obs: jax.Array,
            reward: jax.Array,
            norm_info,
            mode: str,
        ):
            """Helper function to update state immutably"""
            if mode == "brax" and isinstance(state, State):
                return state.replace(
                    obs=obs,
                    reward=reward,
                    info={
                        **state.info,
                        "normalization_info": norm_info,
                    },
                )
            else:
                _, env_state, _, done, info = state
                env_state = self.state_class(
                    **to_state_dict(env_state), normalization_info=norm_info
                )
                return obs, env_state, reward, done, info

        def reset(self, key, params=None):
            state = (
                self.env.reset(key)
                if self.mode == "brax"
                else self._env.reset(key, params)
            )
            rew_norm_info = None
            obs_norm_info = None
            if self.norm_info is None:
                if self.normalize_reward:
                    rew_norm_info = init_norm_info(
                        self.batch_size, (1,), returns=self.normalize_reward
                    )
                if self.normalize_obs:
                    obs_norm_info = init_norm_info(self.batch_size, self.obs_shape)
            else:
                obs_info = self.norm_info.obs
                reward_info = self.norm_info.reward
                if self.normalize_obs:
                    obs_norm_info = NormalizationInfo(
                        count=obs_info.count,
                        mean=obs_info.mean,
                        mean_2=obs_info.mean_2,
                        var=obs_info.var,
                    )
                if self.normalize_reward:
                    rew_norm_info = NormalizationInfo(
                        count=reward_info.count,
                        mean=reward_info.mean,
                        mean_2=reward_info.mean_2,
                        var=reward_info.var,
                        returns=reward_info.returns if self.normalize_reward else None,
                    )
            obs = get_obs_from_state(state, self.mode)

            if self.normalize_obs:
                obs, obs_count, obs_mean, obs_mean_2, obs_var = online_normalize(
                    obs,
                    obs_norm_info.count,
                    obs_norm_info.mean,
                    obs_norm_info.mean_2,
                    train=self.train,
                )
                obs_norm_info = NormalizationInfo(
                    count=obs_count,
                    mean=obs_mean,
                    mean_2=obs_mean_2,
                    var=obs_var,
                )

            norm_info = EnvNormalizationInfo(reward=rew_norm_info, obs=obs_norm_info)
            state = self.update_state_reset(state, obs, norm_info, self.mode)

            return state

        def unnormalize_reward(self, reward: jax.Array, norm_info: NormalizationInfo):
            """Unnormalize the reward using the normalization info."""
            if norm_info is None or norm_info.var is None:
                return reward
            return reward * jnp.sqrt(norm_info.var + 1e-8)

        def normalize_observation(
            self, obs: jax.Array, norm_info: NormalizationInfo
        ) -> jax.Array:
            """Normalize the observation using the normalization info."""
            if norm_info is None or norm_info.var is None:
                return obs
            return (obs - norm_info.mean) / jnp.sqrt(norm_info.var + 1e-8)

        def step(self, *, state, action, params=None, key=None):
            obs_norm_info = (
                state.info["normalization_info"].obs
                if self.mode == "brax"
                else state.normalization_info.obs
            )
            reward_norm_info = (
                state.info["normalization_info"].reward
                if self.mode == "brax"
                else state.normalization_info.reward
            )

            state = (
                self.env.step(state, action)
                if self.mode == "brax"
                else self._env.step(key, state, action, params=params)
            )

            obs, reward, done = get_obs_and_reward_and_done_from_state(
                state, mode=self.mode
            )
            if self.normalize_obs:
                obs, obs_count, obs_mean, obs_mean_2, obs_var = online_normalize(
                    obs,
                    obs_norm_info.count,
                    obs_norm_info.mean,
                    obs_norm_info.mean_2,
                    train=self.train,
                )
                obs_norm_info = NormalizationInfo(
                    count=obs_count,
                    mean=obs_mean,
                    mean_2=obs_mean_2,
                    var=obs_var,
                )

            if self.normalize_reward:
                returns = reward.reshape(
                    -1, 1
                ) + reward_norm_info.returns * self.gamma * (1 - done.reshape(-1, 1))

                normed_reward, rew_count, rew_mean, rew_mean_2, rew_var = (
                    online_normalize(
                        reward,
                        reward_norm_info.count,
                        reward_norm_info.mean,
                        reward_norm_info.mean_2,
                        train=self.train,
                        shift=False,  # Important: rewards shouldn't be mean-shifted
                        returns=returns,
                    )
                )
                normed_reward = (
                    normed_reward.squeeze(-1)
                    if jnp.ndim(normed_reward) > 1 and normed_reward.shape[-1] == 1
                    else (
                        normed_reward.squeeze(0)
                        if np.ndim(normed_reward) > 1 and normed_reward.shape[0] == 1
                        else normed_reward
                    )
                )
                reward_norm_info = NormalizationInfo(
                    count=rew_count,
                    mean=rew_mean,
                    mean_2=rew_mean_2,
                    var=rew_var,
                    returns=returns if self.normalize_reward else None,
                )
                reward = normed_reward

            norm_info = EnvNormalizationInfo(reward=reward_norm_info, obs=obs_norm_info)

            state = self.update_state_step(state, obs, reward, norm_info, self.mode)

            return state

    return NormalizeVecObservation


NormalizeVecObservationBrax = normalize_wrapper_factory("brax")
NormalizeVecObservationGymnax = normalize_wrapper_factory("gymnax")


def clean_to_state_dict(struct_obj):
    raw_state_dict = to_state_dict(struct_obj)
    for key in raw_state_dict.keys():
        if "__dataclass_fields__" in dir(struct_obj.__dataclass_fields__[key].type):
            raw_state_dict[key] = struct_obj.__dataclass_fields__[key].type(
                **raw_state_dict[key]
            )

    return raw_state_dict


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


def get_wrappers(mode: str = "gymnax"):
    if mode == "gymnax":
        return ClipAction, NormalizeVecObservationGymnax
    return ClipActionBrax, NormalizeVecObservationBrax


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
