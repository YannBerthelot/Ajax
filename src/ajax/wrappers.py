"""Wrappers for environment"""

# ruff: noqa: C901
from functools import cached_property, partial
from typing import Any, Callable, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs import Env as BraxEnv
from brax.envs.base import State
from brax.envs.base import Wrapper as BraxWrapper

try:
    from mujoco_playground._src.mjx_env import State as _PlaygroundState

    _StateClasses: tuple = (State, _PlaygroundState)
except ImportError:
    _PlaygroundState = None
    _StateClasses = (State,)
from flax import struct
from flax.serialization import to_state_dict
from gymnasium import core
from gymnasium import spaces as gymnasium_spaces
from gymnax.environments import environment, spaces

from ajax.environments.utils import get_state_action_shapes
from ajax.types import EnvNormalizationInfo, NormalizationInfo
from ajax.utils import online_normalize

# Return signature of ``Environment.step`` since gymnax 1.0: the single
# pre-1.0 ``done`` flag was split into Gymnasium-style ``terminated`` (the
# episode reached a natural terminal state) and ``truncated`` (the episode
# hit its time limit). Ajax needs the two apart -- a truncated episode must
# still bootstrap on V(s_T), a terminated one must not.
StepReturn = Tuple[
    chex.Array, environment.EnvState, jax.Array, jax.Array, jax.Array, dict
]


class GymnaxWrapper:
    """Base class for Gymnax wrappers."""

    def __init__(self, env: environment.Environment):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> environment.Environment:
        if "unwrapped" not in dir(self._env):
            return self._env
        return self._env.unwrapped


_FINAL_OBS_KEYS = ("final_obs", "final_observation", "obs_st")


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
    ) -> StepReturn:
        """Step the environment and flatten the observation"""
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        obs = jnp.reshape(obs, (-1,))
        # The pre-reset observation is written to `info` by the *inner* env, so
        # it escapes the flattening above. Ajax reads it back for the
        # truncation value-bootstrap, where an unflattened array would clash
        # with the flat `last_obs` carried through the rollout scan.
        info = {
            k: (jnp.reshape(v, (-1,)) if k in _FINAL_OBS_KEYS else v)
            for k, v in info.items()
        }
        return obs, state, reward, terminated, truncated, info

    def get_obs(
        self,
        state: environment.EnvState,
        params: Optional[environment.EnvParams] = None,
        key: Optional[chex.PRNGKey] = None,  # noqa: ARG002 -- gymnax's get_obs signature
    ) -> chex.Array:
        """Recompute the observation from state, flattened to match `reset`."""
        return jnp.reshape(self._env.get_obs(state, params), (-1,))


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
    ) -> StepReturn:
        """Step the environment and log the env state, episode return, episode length and timestep"""
        obs, env_state, reward, terminated, truncated, info = self._env.step(
            key,
            state.env_state,
            action,
            params,
        )
        # Episode bookkeeping closes on either terminal flag: a truncated
        # episode is just as finished as a terminated one for logging.
        done = jnp.logical_or(terminated, truncated)
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
        return obs, state, reward, terminated, truncated, info


class ClipAction(GymnaxWrapper):
    """Clip a continuous action to the env's action space, or to ``[low,
    high]`` when given: gym's ``ClipAction``. PureJaxRL's, which this one
    descends from, clips to ``[-1, 1]`` and leaves reading the space as a
    TODO; ``[-1, 1]`` would make the top of a ``Box(0, 2)`` unreachable."""

    def __init__(self, env, low=None, high=None):
        super().__init__(env)
        self.low = None if low is None else jnp.array(low)
        self.high = None if high is None else jnp.array(high)

    def step(self, key, state, action, params=None):
        """Step the environment while clipping the action first"""
        space = self._env.action_space(
            self.default_params if params is None else params
        )
        low = space.low if self.low is None else self.low
        high = space.high if self.high is None else self.high
        action = jnp.clip(action, low, high)
        return self._env.step(key=key, state=state, action=action, params=params)


class ClipActionBrax(BraxWrapper):
    """Clip an action to brax's action space, ``[-1, 1]`` (or ``[low, high]``)."""

    def __init__(self, env, low=-1.0, high=1.0):
        """Set the high and low bounds"""
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, state, action):
        """Step the environment while clipping the action first"""
        action = jnp.clip(action, self.low, self.high)
        return self.env.step(state=state, action=action)


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
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        return self.transform_obs(obs), state, reward, terminated, truncated, info


class TransformReward(GymnaxWrapper):
    """Reward modifying wrapper"""

    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        """Step the env and return the transformed reward"""
        obs, state, reward, terminated, truncated, info = self._env.step(
            key, state, action, params
        )
        return obs, state, self.transform_reward(reward), terminated, truncated, info


class InitialStateWrapper(GymnaxWrapper):
    """Override the plant's state at reset (an initial-condition distribution).

    ``init_state_fn(key, state, params) -> state`` receives the env's own
    reset state and returns the state to start from; the observation is
    recomputed with ``env.get_obs``. Use it to narrow (or fix) the initial
    conditions of a benchmark, e.g. the ``p(O)`` of a control meta-dataset
    (Busetto et al. 2024 keep it fixed at the nominal steady state).
    """

    def __init__(self, env, init_state_fn):
        super().__init__(env)
        self.init_state_fn = init_state_fn

    def reset(self, key, params=None):
        """Reset the inner env, then replace its state."""
        key_env, key_init = jax.random.split(key)
        _, state = self._env.reset(key_env, params)
        state = self.init_state_fn(key_init, state, params)
        return self._env.get_obs(state, params), state


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
    """Initialise running stats with leading axis ``batch_size``.

    The leading axis is *redundant* (online_normalize collapses it on
    every update, so rows carry identical values) but it is load-bearing
    for env-side callers: env-side stats live inside the env state
    pytree, which Brax's ``VmapWrapper`` vmaps over the n_envs axis.
    Shrinking the stats to ``(1, *)`` would break that vmap with a
    mismatched-axis error. Agent-side callers should pass
    ``batch_size=1`` instead (see ``init_agent_obs_norm``).
    """
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
    """Observation out of a gymnax reset/step return, or a brax state.

    Both gymnax returns -- ``(obs, state)`` from reset and the six-value
    ``(obs, state, reward, terminated, truncated, info)`` from step -- carry
    the observation first, so a single index covers them.
    """
    if mode == "gymnax" and isinstance(state, tuple):
        return state[0]
    elif mode == "brax" and isinstance(state, _StateClasses):
        return state.obs


@partial(jax.jit, static_argnames="mode")
def get_obs_and_reward_and_done_from_state(
    state: State | Tuple, mode: str
) -> jax.Array:
    """Observation, reward and end-of-episode flag out of a step return.

    Gymnax >= 1.0 reports ``terminated`` and ``truncated`` separately; the
    reward normaliser only needs to know that the episode ended (to stop the
    discounted-return accumulator), so the two are folded back together here.
    """
    if mode == "gymnax" and isinstance(state, tuple):
        _, _, reward, terminated, truncated, _ = state
        return state[0], reward, jnp.logical_or(terminated, truncated)
    elif mode == "brax" and isinstance(state, _StateClasses):
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
            apply_normalization: bool = True,
        ):
            self.mode = mode
            super().__init__(env)

            self.obs_shape, _ = get_state_action_shapes(env)

            self.train = train
            self.normalize_obs = normalize_obs
            self.normalize_reward = normalize_reward
            self.norm_info = norm_info
            self.gamma = gamma
            # Brax-faithful mode: track running obs stats at every env step
            # (so info["normalization_info"] reflects the latest stats) but
            # do NOT apply the normalisation to ``state.obs`` -- leave obs
            # raw and let the agent normalise inside its loss with the
            # freshest stats. Matches brax PPO's normalise-at-forward
            # pattern (single up-to-date normalizer per training step).
            self.apply_normalization = apply_normalization
            rng = jax.random.PRNGKey(0)
            dummy_obs = get_obs_from_state(
                (
                    env.reset(rng)
                    if self.mode == "brax"
                    else env.reset(rng, params=env.default_params)
                ),
                mode=self.mode,
            )

            self.batch_size = dummy_obs.shape[0] if jnp.ndim(dummy_obs) > 1 else 1

            if mode == "gymnax":
                # if self.mode == "gymnax":
                BaseState = env.reset(
                    key=jax.random.PRNGKey(0), params=env.default_params
                )[1].__class__
                self._raw_state = BaseState

                @struct.dataclass
                class NormalizedEnvState(BaseState):  # type: ignore[valid-type]
                    # Inherit from the actual env_state class
                    normalization_info: Optional[NormalizationInfo] = None

                self.state_class = NormalizedEnvState

        @partial(jax.jit, static_argnames=("self", "mode"))
        def update_state_reset(
            self, state: State | Tuple, obs: jax.Array, norm_info, mode: str
        ):
            """Helper function to update state immutably"""
            if mode == "brax" and isinstance(state, _StateClasses):
                return state.replace(
                    obs=obs,
                    info={
                        **state.info,
                        "normalization_info": norm_info,
                    },
                )
            else:
                _, env_state = state
                state_dict = to_state_dict(env_state)
                state_dict["normalization_info"] = norm_info
                env_state = self.state_class(**state_dict)
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
            if mode == "brax" and isinstance(state, _StateClasses):
                return state.replace(
                    obs=obs,
                    reward=reward,
                    info={
                        **state.info,
                        "normalization_info": norm_info,
                    },
                )
            else:
                _, env_state, _, terminated, truncated, info = state
                state_dict = to_state_dict(env_state)
                state_dict["normalization_info"] = norm_info
                env_state = self.state_class(**state_dict)
                return obs, env_state, reward, terminated, truncated, info

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
            raw_obs = get_obs_from_state(state, self.mode)

            obs = raw_obs
            if self.normalize_obs:
                norm_obs, obs_count, obs_mean, obs_mean_2, obs_var = online_normalize(
                    raw_obs,
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
                obs = norm_obs if self.apply_normalization else raw_obs

            norm_info = EnvNormalizationInfo(reward=rew_norm_info, obs=obs_norm_info)
            state = self.update_state_reset(state, obs, norm_info, self.mode)

            return state

        def unnormalize_reward(self, reward: jax.Array, norm_info: NormalizationInfo):
            """Unnormalize the reward using the normalization info."""
            if norm_info is None or norm_info.var is None:
                return reward
            return reward * jnp.sqrt(norm_info.var.squeeze() + 1e-8)

        def step(self, *, state, action, params=None, key=None):
            if params is None and self.mode == "gymnax":
                params = self._env.default_params
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
            if mode == "gymnax":
                raw_state_dict = to_state_dict(state)
                if "normalization_info" in raw_state_dict:
                    del raw_state_dict["normalization_info"]
                raw_state = self._raw_state(**raw_state_dict)
            else:
                raw_state = state

            raw_state = (
                self.env.step(raw_state, action)
                if self.mode == "brax"
                else self._env.step(
                    key=key, state=raw_state, action=action, params=params
                )
            )

            raw_obs, reward, done = get_obs_and_reward_and_done_from_state(
                raw_state, mode=self.mode
            )
            obs = raw_obs
            if self.normalize_obs:
                norm_obs, obs_count, obs_mean, obs_mean_2, obs_var = online_normalize(
                    raw_obs,
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
                obs = norm_obs if self.apply_normalization else raw_obs

            if self.normalize_reward:
                if self.gamma is None:
                    returns = reward.reshape(-1, 1)
                else:
                    returns = reward.reshape(
                        -1, 1
                    ) + reward_norm_info.returns * self.gamma * (
                        1 - done.reshape(-1, 1)
                    )

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

            state = self.update_state_step(raw_state, obs, reward, norm_info, self.mode)

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
        # gymnasium's reset signature; neither applies to a jax env.
        return_info: bool = False,  # noqa: ARG002
        options: Optional[Any] = None,  # noqa: ARG002
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


def split(x):
    return jax.random.split(x)[0]


class AutoResetWrapper(BraxWrapper):
    """Automatically resets Brax envs that are done, sampling a new random seed for initialization at each reset. This seed is propagated through info["rng"]

    By default the reset is evaluated only on steps where some env is done,
    inside a ``lax.while_loop`` (see :func:`_call_if_any`), and skipped on
    the others. Rollouts stay differentiable with respect to the actions,
    the policy parameters and the env state, none of which the reset depends
    on, and forward mode (``jax.jvp``) works with respect to anything. What
    reverse mode cannot do is differentiate *through the reset itself*, with
    respect to something it depends on (e.g. physics parameters of the env):
    ``jax.grad`` then raises "Reverse-mode differentiation does not work for
    lax.while_loop" at trace time. ``differentiable_reset=True`` evaluates
    the reset on every step instead and keeps the result only for done envs
    (the behaviour before the gating), which reverse mode can cross, at the
    cost of a reset per step.
    """

    def __init__(self, env: BraxWrapper, differentiable_reset: bool = False):
        super().__init__(env)
        self.differentiable_reset = differentiable_reset
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

    def _initial_state(self, rng: jax.Array):
        """Pipeline state and observation of every env after a reset keyed by
        ``rng``."""
        state = self.env.reset(rng)
        return state.pipeline_state, state.obs

    @staticmethod
    def _advance(rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """``_call_if_any``'s key advance: the reset is keyed by the seed's
        next value itself, as ``step`` stores it."""
        rng = split(rng)
        return rng, rng

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)

        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        # The seed advances only on steps where at least one env is done. The
        # done envs restart from a batched reset keyed by the new seed, in
        # which VmapWrapper gives every env its own sub-key.
        rng = state.info["rng"][0]
        new_rng = jnp.where(state.done.any(), split(rng), rng)
        if self.differentiable_reset:
            first_pipeline_state, first_obs = self._initial_state(new_rng)
        else:
            # The same reset, keyed by new_rng too: the loop advances rng to
            # it (see _advance and _call_if_any on why the loop must).
            first_pipeline_state, first_obs = _call_if_any(
                state.done, self._initial_state, rng, advance=self._advance
            )
        # The seed is tiled to the batch size so that info entries keep a
        # leading env axis; only row 0 is read.
        state.info["rng"] = (
            new_rng.reshape(1, -1)
            if self.single_env
            else jnp.tile(new_rng, (self.n_envs, 1))
        )

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, first_pipeline_state, state.pipeline_state
        )
        obs = where_done(first_obs, state.obs)
        info = state.info
        state = state.replace(pipeline_state=pipeline_state, obs=obs, info=info)
        return state


def add_gaussian_noise(x, key, scale: float):
    return x + jax.random.normal(key, x.shape) * scale


class NoiseWrapper(BraxWrapper):
    """
    Add gaussian noise to observations and rewards during transitions.
    """

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale
        self.n_envs = env.reset(jax.random.PRNGKey(0)).obs.shape[0]
        self.single_env = self.n_envs == 1

    def step(self, state: State, action: jax.Array) -> State:
        """Step environment, follow new step API."""
        key = state.info["rng"][0]
        state = self.env.step(state, action)  # type: ignore[has-type]
        obsv, reward, _, _ = (
            state.obs,
            state.reward,
            state.done,
            state.info,
        )

        obs_key, reward_key = jax.random.split(key, 2)
        noisy_obs = add_gaussian_noise(obsv, obs_key, scale=self.scale)
        noisy_reward = add_gaussian_noise(reward, reward_key, scale=self.scale)
        # TODO : find how to infer done from the noisy obs?
        state = state.replace(obs=noisy_obs, reward=noisy_reward)
        print("noising output")

        return state

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["rng"] = (
            rng.reshape(1, -1) if self.single_env else jnp.tile(rng, (self.n_envs, 1))
        )
        info = state.info
        info["rng"] = (
            rng.reshape(1, -1) if self.single_env else jnp.tile(rng, (self.n_envs, 1))
        )
        state = state.replace(info=info)
        return state


class BatchRngWrapper:
    """Splits an unbatched PRNG key into n_envs keys on reset.

    Playground's `BraxAutoResetWrapper.reset` (and `FreshAutoResetWrapper.reset`)
    assumes `rng` is already shape (n_envs, 2) so it can
    `jax.vmap(jax.random.split)(rng)`. Ajax callers pass a single unbatched
    key, so this adapter bridges the two conventions.
    """

    def __init__(self, env, n_envs: int):
        self.env = env
        self.n_envs = n_envs

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def reset(self, rng):
        if rng.ndim == 1:
            rng = jax.random.split(rng, self.n_envs)
        return self.env.reset(rng)

    def step(self, state, action):
        return self.env.step(state, action)


class FinalObsWrapper:
    """Stashes state.obs in state.info['final_obs'] on every reset/step.

    Place this directly below an auto-reset wrapper in the stack. The auto-reset
    wrapper overwrites state.obs with the reset observation on `done`, so
    without this the pre-reset observation is lost — breaking correct value
    bootstrapping on truncation (PPO/SAC need V(s_T) at truncation, not
    V(s_reset)). With this wrapper, downstream code can read
    state.info['final_obs'] to recover the terminal observation.

    Backend-agnostic: duck-typed passthrough that works for both brax State
    and mujoco_playground State (both expose `.obs` and `.info`).
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["final_obs"] = state.obs
        return state

    def step(self, state, action):
        state = self.env.step(state, action)
        state.info["final_obs"] = state.obs
        return state


def _split_each(keys: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Split every key of an ``(n, 2)`` batch; returns two ``(n, 2)`` batches."""
    pairs = jax.vmap(jax.random.split)(keys)
    return pairs[:, 0], pairs[:, 1]


def _call_if_any(
    pred: jax.Array,
    fn: Callable[[jax.Array], Any],
    keys: jax.Array,
    advance: Callable[[jax.Array], Tuple[jax.Array, jax.Array]] = _split_each,
):
    """Return ``fn(subkeys)`` if any element of ``pred`` is set, else zeros of
    the same structure -- evaluating ``fn`` at most once, and not at all when
    no element is set, including under ``jax.vmap``.

    ``advance(keys)`` returns ``(next_keys, subkeys)``. By default ``keys`` is
    an ``(n, 2)`` batch of PRNG keys and ``advance`` splits each of them.

    Why a ``while_loop`` and not a ``lax.cond``: Ajax always vmaps training
    over seeds, so a predicate computed from the env state is batched, and a
    ``cond`` with a batched predicate lowers to ``select``, which evaluates
    the branch on every call. A batched ``while_loop`` instead keeps
    iterating while *any* element's predicate holds, so here it runs its
    body once when some element needs it and skips it otherwise; elements
    whose own predicate is false keep the zeros.

    The keys travel in the loop carry and the body advances them. This is
    load-bearing: were ``fn`` to close over constant keys, its whole
    computation would be loop-invariant and XLA's while-loop invariant code
    motion would hoist it out of the loop, evaluating it unconditionally
    again (measured on CheetahRun: the same cost as resetting every step).
    So ``next_keys`` must differ from ``keys``, as any split's output does.
    """
    zeros = jax.tree.map(
        lambda s: jnp.zeros(s.shape, s.dtype),
        jax.eval_shape(lambda k: fn(advance(k)[1]), keys),
    )

    def body(carry):
        _, loop_keys, _ = carry
        loop_keys, subkeys = advance(loop_keys)
        return jnp.zeros((), dtype=bool), loop_keys, fn(subkeys)

    _, _, out = jax.lax.while_loop(
        lambda carry: carry[0], body, (jnp.any(pred), keys, zeros)
    )
    return out


class FreshAutoResetWrapper:
    """Auto-resets a batched mujoco_playground env to a *fresh* initial state.

    Drop-in replacement for playground's ``BraxAutoResetWrapper`` in Ajax's
    playground stack (EpisodeWrapper, VmapWrapper, FinalObsWrapper, this,
    BatchRngWrapper; see ``ajax.environments.create``). With its default
    ``full_reset=False``, upstream caches each env's first reset and restarts
    every later episode from it, so a whole run sees only ``n_envs`` initial
    conditions, whereas dm_control draws a new one per episode. Upstream's
    ``full_reset=True`` does draw new ones, but it (a) evaluates a full reset
    on every step (on CheetahRun, whose reset runs a 200-step stabilisation,
    ~180x the cost of a step) and (b) replaces the whole ``info`` of done
    envs with the reset's, which zeroes ``truncation``, ``episode_done`` and
    ``episode_metrics`` and overwrites ``final_obs`` with the reset
    observation, breaking the V(s_T) bootstrap at truncation.

    On a step where some env is done, this wrapper draws a fresh batched
    reset (computed only on such steps, see :func:`_call_if_any`) and gives
    each done env, from it:

    * ``data`` and ``obs``;
    * the ``info`` entries produced by the base environment's own ``reset``,
      i.e. its per-episode state (its rng, task targets, ...).

    Everything else describes the transition just taken and passes through:
    ``reward``, ``done``, ``metrics``, and the bookkeeping that the wrappers
    below write into ``info`` on every step (EpisodeWrapper's ``steps``,
    ``truncation`` and ``episode_*``; FinalObsWrapper's ``final_obs``).

    ``differentiable_reset=True`` evaluates the reset on every step instead,
    for reverse-mode gradients through the reset itself; see
    :class:`AutoResetWrapper`.
    """

    _RNG_KEY = "fresh_auto_reset_rng"

    def __init__(self, env, differentiable_reset: bool = False):
        self.env = env
        self.differentiable_reset = differentiable_reset

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    @cached_property
    def _episode_info_keys(self) -> Tuple[str, ...]:
        """``info`` keys of the base environment's own reset (shape-only)."""
        base_reset = jax.eval_shape(self.unwrapped.reset, jax.random.PRNGKey(0))
        return tuple(base_reset.info)

    def reset(self, rng: jax.Array):
        """``rng``: one key per env, shape ``(n_envs, 2)``."""
        rng, key = _split_each(rng)
        state = self.env.reset(key)
        state.info[self._RNG_KEY] = rng
        return state

    def step(self, state, action: jax.Array):
        if "steps" in state.info:
            # EpisodeWrapper's step counter restarts with the new episode.
            steps = jnp.where(state.done, 0, state.info["steps"])
            state = state.replace(info={**state.info, "steps": steps})
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        rng, key = _split_each(state.info[self._RNG_KEY])
        done = state.done.astype(bool)
        if self.differentiable_reset:
            fresh = self.env.reset(_split_each(key)[1])
        else:
            fresh = _call_if_any(done, self.env.reset, key)

        def where_done(new, old):
            mask = jnp.reshape(done, done.shape + (1,) * (old.ndim - done.ndim))
            return jnp.where(mask, new, old)

        info = dict(state.info)
        for name in self._episode_info_keys:
            info[name] = jax.tree.map(where_done, fresh.info[name], info[name])
        info[self._RNG_KEY] = rng
        return state.replace(
            data=jax.tree.map(where_done, fresh.data, state.data),
            obs=jax.tree.map(where_done, fresh.obs, state.obs),
            info=info,
        )


class TerminatedTruncatedWrapper(GymnaxWrapper):
    """Split a time-limit-inclusive terminal flag back into terminated/truncated.

    Stock gymnax >= 1.0 environments already return the two flags apart, so
    they do **not** need this wrapper. It is for third-party environments that
    subclass gymnax's ``Environment`` but whose ``step_env`` still reports the
    pre-1.0 ``done = terminated | truncated`` in the ``terminated`` slot (or
    that return the five-value tuple outright). Left unwrapped, such an env
    makes every time-limit truncation look like a natural termination, so
    PPO/SAC drop the ``V(s_T)`` bootstrap and silently underestimate the
    value of long-running states.

    The time limit is recovered the same way gymnax's own
    ``Environment.is_truncated`` does -- ``state.time >=
    params.max_steps_in_episode`` -- and removed from ``terminated``.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(
        self,
        key,
        state: environment.EnvState,
        action,
        params: environment.EnvParams = None,
    ) -> StepReturn:
        """Step the env and return Gymnasium-style terminal flags"""
        if params is None:
            params = self._env.default_params
        out = self._env.step(key, state, action, params)
        # Accept both the pre-1.0 five-value tuple and the 1.0 six-value one;
        # in either case the terminal flag we get still folds in the time
        # limit, which is exactly what this wrapper undoes.
        obs, state, reward, done = out[0], out[1], out[2], out[3]
        info = out[-1]
        truncated = state.time >= params.max_steps_in_episode
        terminated = jnp.logical_and(
            jnp.asarray(done, dtype=bool), jnp.logical_not(truncated)
        )
        return obs, state, reward, terminated, truncated, info
