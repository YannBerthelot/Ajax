"""The error terms a PID controller acts on, carried in the observation.

A PID controller computes ``u_t = Kp e_t + Ki I_t + Kd D_t`` from the
tracking error ``e_t = target_t - current_t``, its integral ``I_t`` and
its difference ``D_t``. :class:`PIDTermsWrapper` puts ``(e_t, I_t, D_t)``
in front of a gymnax env's observation, so a policy reading them stays a
function of the observation: every agent (replay or on-policy, with or
without memory) trains on the wrapped env unchanged, and training and
evaluation, which step the same wrapped env, see the same terms. The PID
policy (:class:`ajax.modules.pid_actor.PIDPolicy`) reads them there.

The terms are the discrete-time PID's with sampling period ``dt``
(backward-difference form, as in Astrom & Hagglund, *Advanced PID
Control*, 2006, cited from memory, unverified)::

    I_t = clip(I_{t-1} + dt * e_t, -integral_limit, integral_limit),  I_{-1} = 0
    D_t = (e_t - e_{t-1}) / dt,                                       D_0 = 0

The clip is clamping anti-windup (``integral_limit=None``: no clip).
Every episode starts afresh, the auto-reset ones included: the integral
restarts from ``dt * e_0`` and the difference from 0, so no term carries
over from the previous episode.

The wrapper declares the terms raw (``raw_obs_columns``): Ajax's
observation normalisers leave them in the env's units, so gains acting on
them are a PID's, and normalise the rest of the observation as usual.

The wrapper is deprecated: the terms are the controller's, which computes
them from the observation it is handed, so they are an observation feature
(:class:`ajax.extensions.observe.PIDTermsInput`, which reads the functions
below). The PID policy reads them wherever the observation holds them.
"""

from __future__ import annotations

import warnings
from math import prod
from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from ajax.wrappers import _FINAL_OBS_KEYS, GymnaxWrapper


def split_pid_terms(obs: jax.Array, n_errors: int, at: int = 0) -> jax.Array:
    """The ``(e, I, D)`` held in ``obs`` from column ``at`` (a
    :class:`PIDTermsWrapper` puts them first), stacked as
    ``(..., 3, n_errors)``."""
    terms = obs[..., at : at + 3 * n_errors]
    return terms.reshape(*obs.shape[:-1], 3, n_errors)


def check_pid_terms(
    current_idx: Sequence[int],
    target_idx: Sequence[int],
    dt: float,
    integral_limit: Optional[float],
) -> None:
    """Refuse terms without one set-point per controlled value, or with a
    period or clamp that is not positive."""
    if not current_idx or len(current_idx) != len(target_idx):
        raise ValueError(
            "current_idx and target_idx need one entry per error, got"
            f" {tuple(current_idx)} and {tuple(target_idx)}."
        )
    if dt <= 0 or (integral_limit is not None and integral_limit <= 0):
        raise ValueError(
            f"dt and integral_limit must be positive, got {dt} and {integral_limit}."
        )


def tracking_error(
    flat_obs: jax.Array, current_idx: Sequence[int], target_idx: Sequence[int]
) -> jax.Array:
    """``e = target - current`` from a flat observation's last axis."""
    return flat_obs[..., jnp.array(target_idx)] - flat_obs[..., jnp.array(current_idx)]


def clamp_integral(integral: jax.Array, integral_limit: Optional[float]) -> jax.Array:
    """Clamping anti-windup; ``integral_limit`` None: no clamp."""
    if integral_limit is None:
        return integral
    return jnp.clip(integral, -integral_limit, integral_limit)


def start_terms(
    error: jax.Array, dt: float, integral_limit: Optional[float]
) -> tuple[jax.Array, jax.Array]:
    """``(I_0, D_0)`` at an episode's first error ``e_0``."""
    return clamp_integral(dt * error, integral_limit), jnp.zeros_like(error)


def advance_terms(
    error: jax.Array,
    last_error: jax.Array,
    integral: jax.Array,
    dt: float,
    integral_limit: Optional[float],
) -> tuple[jax.Array, jax.Array]:
    """``(I_t, D_t)`` from ``e_t``, ``e_{t-1}`` and ``I_{t-1}``."""
    integral = clamp_integral(integral + dt * error, integral_limit)
    return integral, (error - last_error) / dt


@struct.dataclass
class PIDTermsState:
    env_state: Any
    error: jax.Array  # e_t, (n_errors,)
    integral: jax.Array  # I_t
    difference: jax.Array  # D_t

    @property
    def time(self) -> jax.Array:
        """The inner env's clock, which Ajax's gymnax ``step`` reads."""
        inner = self.env_state
        return inner.time if hasattr(inner, "time") else inner.t


class PIDTermsWrapper(GymnaxWrapper):
    """Observation ``[e, I, D, raw obs]`` for a gymnax env. Deprecated: pass
    :class:`ajax.extensions.observe.PIDTermsInput` with the same arguments
    in the agent's extensions.

    Args:
        env: gymnax env; its observation is flattened, and the indices
            below index the flat observation.
        current_idx: observation dims of the controlled values.
        target_idx: observation dims of their set-points, one per value.
        dt: sampling period: the integral sums ``dt * e``, the difference
            is divided by it, so the gains keep their continuous-time units.
        integral_limit: clamp of the integral (anti-windup); None: no clamp.
    """

    def __init__(
        self,
        env: Any,
        current_idx: Sequence[int],
        target_idx: Sequence[int],
        dt: float = 1.0,
        integral_limit: Optional[float] = None,
    ):
        super().__init__(env)
        warnings.warn(
            "PIDTermsWrapper is deprecated: pass PIDTermsInput(current_idx,"
            " target_idx, dt, integral_limit) (ajax.extensions.observe) in the"
            " agent's extensions instead.",
            FutureWarning,
            stacklevel=2,
        )
        check_pid_terms(current_idx, target_idx, dt, integral_limit)
        self.current_idx = tuple(current_idx)
        self.target_idx = tuple(target_idx)
        self.dt = dt
        self.integral_limit = integral_limit

    @property
    def n_errors(self) -> int:
        return len(self.current_idx)

    @property
    def raw_obs_columns(self) -> int:
        """The terms stay raw under every observation normaliser
        (:func:`ajax.environments.utils.raw_obs_columns`): a PID's gains act
        on them in the env's units."""
        return 3 * self.n_errors

    def observation_space(self, params) -> spaces.Box:
        raw_dim = prod(self._env.observation_space(params).shape)
        return spaces.Box(-jnp.inf, jnp.inf, (3 * self.n_errors + raw_dim,))

    def _error(self, raw_obs: jax.Array) -> jax.Array:
        flat = jnp.reshape(raw_obs, (-1,))
        return tracking_error(flat, self.current_idx, self.target_idx)

    def _start(self, env_state: Any, raw_obs: jax.Array) -> PIDTermsState:
        error = self._error(raw_obs)
        return PIDTermsState(
            env_state, error, *start_terms(error, self.dt, self.integral_limit)
        )

    def _advance(
        self, state: PIDTermsState, env_state: Any, raw_obs: jax.Array
    ) -> PIDTermsState:
        error = self._error(raw_obs)
        terms = advance_terms(
            error, state.error, state.integral, self.dt, self.integral_limit
        )
        return PIDTermsState(env_state, error, *terms)

    @staticmethod
    def _observation(state: PIDTermsState, raw_obs: jax.Array) -> jax.Array:
        terms = (state.error, state.integral, state.difference)
        return jnp.concatenate([*terms, jnp.reshape(raw_obs, (-1,))]).astype(
            jnp.float32
        )

    def reset(self, key, params=None):
        raw_obs, env_state = self._env.reset(key, params)
        state = self._start(env_state, raw_obs)
        return self._observation(state, raw_obs), state

    def step(self, key, state: PIDTermsState, action, params=None):
        raw_obs, env_state, reward, terminated, truncated, info = self._env.step(
            key, state.env_state, action, params
        )
        # gymnax auto-resets inside `step`: on an end `raw_obs` is the next
        # episode's first observation, the pre-reset one sits in `info`.
        # Without that key the env did not report one: continue on raw_obs.
        final_raw = next((info[k] for k in _FINAL_OBS_KEYS if k in info), raw_obs)
        ended = self._advance(state, env_state, final_raw)
        fresh = self._start(env_state, raw_obs)
        done = jnp.logical_or(terminated, truncated)
        state = PIDTermsState(
            env_state,
            jnp.where(done, fresh.error, ended.error),
            jnp.where(done, fresh.integral, ended.integral),
            jnp.where(done, fresh.difference, ended.difference),
        )
        info = {
            k: self._observation(ended, v) if k in _FINAL_OBS_KEYS else v
            for k, v in info.items()
        }
        return (
            self._observation(state, raw_obs),
            state,
            reward,
            terminated,
            truncated,
            info,
        )

    def get_obs(self, state: PIDTermsState, params=None, key=None):
        """Recompute the observation from ``state``, as ``reset``/``step``."""
        del key
        return self._observation(state, self._env.get_obs(state.env_state, params))


__all__ = ["PIDTermsState", "PIDTermsWrapper", "split_pid_terms"]
