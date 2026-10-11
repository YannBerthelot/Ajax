"""Wrappers for environment"""

import dataclasses
from functools import cached_property, partial
from typing import Any, Callable, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import State
from brax.envs.base import Wrapper as BraxWrapper
from flax import struct
from gymnax.environments import environment, spaces

from ajax.environments.utils import get_state_action_shapes, raw_obs_columns
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
        # copy and pickle build the instance before ``_env`` is set and then
        # probe protocol dunders (``__setstate__``, ``__deepcopy__``):
        # forwarding ``_env`` would recurse forever, and forwarding a dunder
        # would run the inner env's protocol on the wrapper.
        if name == "_env" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> environment.Environment:
        if "unwrapped" not in dir(self._env):
            return self._env
        return self._env.unwrapped


_FINAL_OBS_KEYS = ("final_obs", "final_observation")
# Where an observation normaliser keeps the pre-reset observation in the
# env's own units once it has normalised the final-observation keys.
RAW_FINAL_OBS_KEY = "raw_final_obs"
# Where an observation normaliser keeps a brax state's observation in the
# env's own units, the state's ``obs`` being the normalised one.
RAW_OBS_KEY = "raw_obs"
# Where a reward normaliser keeps the step's reward in the env's own units.
RAW_REWARD_KEY = "raw_reward"


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


class ClipAction(GymnaxWrapper):
    """Clip a continuous action to the env's action space, or to ``[low,
    high]`` when given: gym's ``ClipAction``. PureJaxRL's, which this one
    descends from, clips to ``[-1, 1]`` and leaves reading the space as a
    TODO; ``[-1, 1]`` would make the top of a ``Box(0, 2)`` unreachable.
    Ajax's own steps map agent actions onto the box first
    (:func:`ajax.environments.utils.agent_action_to_env`), so here it is a
    guard for raw stepping (``interaction.step``)."""

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


class InitialStateWrapper(GymnaxWrapper):
    """Override the plant's state at reset (an initial-condition distribution).

    ``init_state_fn(key, state, params) -> state`` receives the env's own
    reset state and returns the state to start from; the observation is
    recomputed with ``env.get_obs``. Use it to narrow (or fix) the initial
    conditions of a benchmark, e.g. the ``p(O)`` of a control meta-dataset
    (Busetto et al. 2024 keep it fixed at the nominal steady state).

    Every episode starts this way, the auto-reset ones included: gymnax's
    ``step`` auto-resets with the inner env's ``reset_env``, so ``step``
    applies ``init_state_fn`` to that reset state too (fields the inner step
    carries across its auto-reset stay as it left them).
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

    def step(self, key, state, action, params=None):
        """Step the inner env; where it auto-reset, replace the reset state."""
        key_step, key_init = jax.random.split(key)
        obs, state, reward, terminated, truncated, info = self._env.step(
            key_step, state, action, params
        )
        done = jnp.logical_or(terminated, truncated)
        start = self.init_state_fn(key_init, state, params)
        state = jax.tree.map(lambda s, x: jnp.where(done, s, x), start, state)
        obs = jnp.where(done, self._env.get_obs(start, params), obs)
        return obs, state, reward, terminated, truncated, info


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


def _fields(state: Any) -> dict:
    """A state's fields one level deep. A wrapper's nested env state stays
    a state; ``to_state_dict`` would turn it into a dict the inner env
    cannot step."""
    return {f.name: getattr(state, f.name) for f in dataclasses.fields(state)}


class _OnlineNormalization:
    """Online normalisation of an env's observations and rewards, the
    arithmetic the brax and gymnax normalisers below share; each says where
    its backend's reset and step keep the observation, the reward, the info
    and the statistics (``normalization_info``)."""

    mode: str

    def __init__(
        self,
        env: Any,
        train: bool = True,
        norm_info: Optional[EnvNormalizationInfo] = None,
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        gamma: Optional[float] = None,
        apply_normalization: bool = True,
    ):
        super().__init__(env)  # type: ignore[call-arg]
        self.obs_shape, _ = get_state_action_shapes(env)
        self.raw_columns = raw_obs_columns(env)
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
        obs_shape = self._trace_reset().shape
        self.batch_size = obs_shape[0] if len(obs_shape) > 1 else 1

    def _trace_reset(self) -> Any:
        """The inner reset's observation, traced, not run (shapes only)."""
        raise NotImplementedError

    def _initial_stats(self) -> EnvNormalizationInfo:
        """The statistics a reset starts from: the given ones (an
        observation's carry no discounted return), else zeros."""
        given: Any = self.norm_info
        if given is not None:
            return EnvNormalizationInfo(
                reward=given.reward if self.normalize_reward else None,
                obs=given.obs.replace(returns=None) if self.normalize_obs else None,
            )
        return EnvNormalizationInfo(
            reward=(
                init_norm_info(self.batch_size, (1,), returns=True)
                if self.normalize_reward
                else None
            ),
            obs=(
                init_norm_info(self.batch_size, self.obs_shape)
                if self.normalize_obs
                else None
            ),
        )

    def _normalize_obs(
        self, raw_obs: jax.Array, stats: Optional[NormalizationInfo], info: dict
    ) -> Tuple[jax.Array, Optional[NormalizationInfo], dict]:
        """The observation the agent sees, the updated statistics and the
        info entries normalising adds (see :meth:`_normalized_info`)."""
        if not self.normalize_obs:
            return raw_obs, stats, {}
        assert stats is not None
        obs, count, mean, mean_2, var = online_normalize(
            raw_obs,
            stats.count,
            stats.mean,
            stats.mean_2,
            train=self.train,
            raw_columns=self.raw_columns,
        )
        stats = NormalizationInfo(count=count, mean=mean, mean_2=mean_2, var=var)
        if not self.apply_normalization:
            return raw_obs, stats, {}
        return obs, stats, self._normalized_info(info, stats)

    def _normalized_info(self, info: dict, stats: NormalizationInfo) -> dict:
        """The pre-reset observations an auto-reset stack leaves in its
        info, normalised as the step's observation was (they are the
        transition's next observation), and the raw one under
        ``RAW_FINAL_OBS_KEY`` for readers of the env's own units (a
        stateful expert)."""
        keys = [k for k in _FINAL_OBS_KEYS if k in info]
        out = {
            k: online_normalize(
                info[k],
                stats.count,
                stats.mean,
                stats.mean_2,
                train=False,
                raw_columns=self.raw_columns,
            )[0]
            for k in keys
        }
        return out | ({RAW_FINAL_OBS_KEY: info[keys[0]]} if keys else {})

    def _normalize_reward(
        self, reward: jax.Array, done: jax.Array, stats: NormalizationInfo
    ) -> Tuple[jax.Array, NormalizationInfo]:
        """The reward divided by the running statistics, and those updated:
        the single-step reward's (gamma None), or SB3 VecNormalize's
        discounted return of the episode so far, R = r + gamma R, restarted
        after the episode's end (gymnasium's NormalizeReward restarts it at
        the end step instead)."""
        returns = reward.reshape(-1, 1)
        if self.gamma is not None:
            assert stats.returns is not None  # init_norm_info(returns=True)
            returns = returns + self.gamma * stats.returns
        normed, count, mean, mean_2, var = online_normalize(
            reward,
            stats.count,
            stats.mean,
            stats.mean_2,
            train=self.train,
            shift=False,  # Important: rewards shouldn't be mean-shifted
            returns=returns,
        )
        if jnp.ndim(normed) > 1 and normed.shape[-1] == 1:
            normed = normed.squeeze(-1)
        elif np.ndim(normed) > 1 and normed.shape[0] == 1:
            normed = normed.squeeze(0)
        stats = NormalizationInfo(
            count=count,
            mean=mean,
            mean_2=mean_2,
            var=var,
            returns=returns * (1 - done.reshape(-1, 1)),
        )
        return normed, stats

    def _normalize(
        self,
        raw_obs: jax.Array,
        info: dict,
        stats: EnvNormalizationInfo,
        reward: Optional[jax.Array] = None,
        done: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, Optional[jax.Array], EnvNormalizationInfo, dict]:
        """A reset's (no ``reward``) or a step's observation and reward
        normalised, the statistics updated and the info entries added."""
        obs, obs_stats, extra = self._normalize_obs(raw_obs, stats.obs, info)
        reward_stats = stats.reward
        if self.normalize_reward and reward is not None:
            extra |= {RAW_REWARD_KEY: reward}
            assert reward_stats is not None and done is not None
            reward, reward_stats = self._normalize_reward(reward, done, reward_stats)
        return (
            obs,
            reward,
            EnvNormalizationInfo(reward=reward_stats, obs=obs_stats),
            extra,
        )


class NormalizeVecObservationBrax(_OnlineNormalization, BraxWrapper):
    """Online normalisation of a brax / playground stack's observations and
    rewards. The statistics live in ``state.info["normalization_info"]``,
    and the info keeps the raw observation and reward (``RAW_OBS_KEY``,
    ``RAW_REWARD_KEY``): the state's own are the normalised ones."""

    mode = "brax"

    def _trace_reset(self) -> Any:
        return jax.eval_shape(self.env.reset, jax.random.PRNGKey(0)).obs

    def _with_stats(self, state: State, norm_info, extra: dict, **fields) -> State:
        info = {**state.info, **extra, "normalization_info": norm_info}
        return state.replace(info=info, **fields)

    def _normalize_obs(self, raw_obs, stats, info):
        obs, stats, extra = super()._normalize_obs(raw_obs, stats, info)
        if self.normalize_obs and self.apply_normalization:
            # The state's obs is the normalised one, and an expert reads the
            # env's own units (interaction.get_raw_obs). A gymnax state
            # recomputes its observation from the env instead.
            extra = extra | {RAW_OBS_KEY: raw_obs}
        return obs, stats, extra

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        obs, _, norm_info, extra = self._normalize(
            state.obs, state.info, self._initial_stats()
        )
        if self.normalize_reward:
            # A brax state carries its info: the key exists from the reset.
            extra |= {RAW_REWARD_KEY: state.reward}
        return self._with_stats(state, norm_info, extra, obs=obs)

    def step(self, state: State, action: jax.Array) -> State:
        stats = state.info["normalization_info"]
        state = self.env.step(state, action)
        obs, reward, norm_info, extra = self._normalize(
            state.obs, state.info, stats, state.reward, state.done
        )
        return self._with_stats(state, norm_info, extra, obs=obs, reward=reward)


class NormalizeVecObservationGymnax(_OnlineNormalization, GymnaxWrapper):
    """Online normalisation of a gymnax env's observations and rewards. The
    statistics live in the env state, a subclass of the inner env's state
    class with a ``normalization_info`` field."""

    mode = "gymnax"

    def _trace_reset(self) -> Any:
        """Also the state class the statistics extend."""
        obs, state = jax.eval_shape(
            lambda key: self._env.reset(key, params=self._env.default_params),
            jax.random.PRNGKey(0),
        )
        self._raw_state = type(state)

        @struct.dataclass
        class NormalizedEnvState(self._raw_state):  # type: ignore[name-defined]
            normalization_info: Optional[NormalizationInfo] = None

        self.state_class = NormalizedEnvState
        return obs

    def _with_stats(self, env_state: Any, norm_info) -> Any:
        return self.state_class(**_fields(env_state), normalization_info=norm_info)

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        obs, _, norm_info, _ = self._normalize(obs, {}, self._initial_stats())
        return obs, self._with_stats(env_state, norm_info)

    def step(self, *, state, action, params=None, key=None):
        if params is None:
            params = self._env.default_params
        raw_fields = dataclasses.fields(self._raw_state)
        raw_state = self._raw_state(
            **{f.name: getattr(state, f.name) for f in raw_fields}
        )
        obs, env_state, reward, terminated, truncated, info = self._env.step(
            key=key, state=raw_state, action=action, params=params
        )
        obs, reward, norm_info, extra = self._normalize(
            obs,
            info,
            state.normalization_info,
            reward,
            jnp.logical_or(terminated, truncated),
        )
        env_state = self._with_stats(env_state, norm_info)
        return obs, env_state, reward, terminated, truncated, {**info, **extra}


def get_wrappers(mode: str = "gymnax"):
    if mode == "gymnax":
        return ClipAction, NormalizeVecObservationGymnax
    return ClipActionBrax, NormalizeVecObservationBrax


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
        self.n_envs = jax.eval_shape(env.reset, jax.random.PRNGKey(0)).obs.shape[0]
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


class BatchRngWrapper(BraxWrapper):
    """Splits an unbatched PRNG key into n_envs keys on reset.

    Playground's `BraxAutoResetWrapper.reset` (and `FreshAutoResetWrapper.reset`)
    assumes `rng` is already shape (n_envs, 2) so it can
    `jax.vmap(jax.random.split)(rng)`. Ajax callers pass a single unbatched
    key, so this adapter bridges the two conventions.
    """

    def __init__(self, env, n_envs: int):
        super().__init__(env)
        self.n_envs = n_envs

    def reset(self, rng):
        if rng.ndim == 1:
            rng = jax.random.split(rng, self.n_envs)
        return self.env.reset(rng)


class FinalObsWrapper(BraxWrapper):
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


class FreshAutoResetWrapper(BraxWrapper):
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
        super().__init__(env)
        self.differentiable_reset = differentiable_reset

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
    pre-1.0 ``done = terminated | truncated``, which gymnax's base ``step``
    hands back in the ``terminated`` slot. Left unwrapped, such an env makes
    every time-limit truncation look like a natural termination, so PPO/SAC
    drop the ``V(s_T)`` bootstrap and silently underestimate the value of
    long-running states.

    The time limit is the ``truncated`` flag gymnax's base ``step`` reports:
    it is computed from the pre-reset state, whereas the returned state has
    already been auto-reset (``time == 0``) at the limit.
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
        obs, state, reward, done, truncated, info = self._env.step(
            key, state, action, params
        )
        terminated = jnp.logical_and(
            jnp.asarray(done, dtype=bool), jnp.logical_not(truncated)
        )
        return obs, state, reward, terminated, truncated, info
