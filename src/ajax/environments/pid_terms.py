"""The error terms a PID controller acts on, which the observation feature
ajax.extensions.observe.PIDTermsInput appends to the observation and the PID
policy (ajax.modules.pid_actor.PIDPolicy) reads there.

A PID controller computes u_t = Kp e_t + Ki I_t + Kd D_t from the tracking
error e_t = target_t - current_t, its integral I_t and its difference D_t.
The terms are the discrete-time PID's with sampling period dt
(backward-difference form, as in Astrom & Hagglund, Advanced PID Control,
2006, cited from memory, unverified)::

    I_t = clip(I_{t-1} + dt * e_t, -integral_limit, integral_limit),  I_{-1} = 0
    D_t = (e_t - e_{t-1}) / dt,                                       D_0 = 0

The clip is clamping anti-windup (integral_limit=None: no clip). Every
episode starts afresh: the integral restarts from dt * e_0 and the
difference from 0, so no term carries over from the previous episode.
"""

from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp


def split_pid_terms(obs: jax.Array, n_errors: int, at: int = 0) -> jax.Array:
    """The (e, I, D) held in obs from column at, stacked as
    (..., 3, n_errors)."""
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


__all__ = ["split_pid_terms"]
