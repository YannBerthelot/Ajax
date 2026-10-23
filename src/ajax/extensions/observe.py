"""Observation features: columns an agent appends to every observation.

Each is an :class:`~ajax.extensions.base.Extension` implementing the
``observe`` phase (:mod:`ajax.controller`): its columns join the
observation whenever one arrives, in collection and evaluation alike, and
are stored with it, so the actor, the critics and the replay all read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp
from flax import struct

from ajax.controller import (
    ObserveInputs,
    call_expert,
    flatten_expert_state,
    observation_layout,
)
from ajax.environments.pid_terms import (
    advance_terms,
    check_pid_terms,
    start_terms,
    tracking_error,
)
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.extensions.base import Extension
from ajax.schedule import progress


@dataclass(frozen=True)
class ExpertActionInput(Extension):
    """The agent's expert's action at the observation, called from the
    state it holds there, in agent units: raw, since it is an action. An
    agent learning from the expert's labels alone (DAgger) would read the
    label it learns: it refuses it (``reward_learning``)."""

    name: str = "expert_action"
    observes_normalised: ClassVar[bool] = False
    requires: ClassVar[frozenset[str]] = frozenset({"agent_expert", "reward_learning"})

    def observation_width(self, env_args: Any) -> int:
        return get_action_dim(env_args.env, env_args.env_params)

    def observe(self, inputs: ObserveInputs, state: Any) -> tuple[jax.Array, Any]:
        action, _ = call_expert(inputs.expert, inputs.expert_state, inputs.raw_obs)
        return jax.lax.stop_gradient(action), state


@dataclass(frozen=True)
class ExpertStateInput(Extension):
    """The agent's stateful expert's state before its call at the
    observation (a PID's integrator), flattened to ``width`` columns
    (:meth:`for_expert` reads it off the expert)."""

    width: int = 0
    name: str = "expert_state"
    requires: ClassVar[frozenset[str]] = frozenset({"agent_expert"})

    @classmethod
    def for_expert(cls, expert: Any) -> "ExpertStateInput":
        """The input for ``expert``'s state, as wide as it flattens."""
        if not hasattr(expert, "init_state"):
            raise ValueError("ExpertStateInput reads a stateful expert's state.")
        flat = flatten_expert_state(expert.init_state(1))
        return cls(width=0 if flat is None else int(flat.shape[-1]))

    def observation_width(self, env_args: Any) -> int:
        del env_args
        return self.width

    def observe(self, inputs: ObserveInputs, state: Any) -> tuple[jax.Array, Any]:
        flat = flatten_expert_state(inputs.expert_state)
        if flat is None:
            raise ValueError("ExpertStateInput reads a stateful expert's state.")
        return jax.lax.stop_gradient(flat), state


@struct.dataclass
class PIDMemory:
    """A PID's memory per env: the last error and integral, and whether the
    episode has had an observation (its first starts the terms afresh)."""

    error: jax.Array  # (n_envs, n_errors)
    integral: jax.Array
    started: jax.Array  # (n_envs,)


@dataclass(frozen=True)
class PIDTermsInput(Extension):
    """A PID controller's terms ``(e, I, D)`` for the tracking errors
    ``e = obs[target_idx] - obs[current_idx]`` of the env's flat observation,
    with sampling period ``dt`` and clamping anti-windup ``integral_limit``
    (:mod:`ajax.environments.pid_terms` gives the discrete-time terms). Raw,
    so gains acting on them are a PID's; a :class:`PIDPolicy` actor reads
    them where the observation holds them (:func:`pid_terms_at`). Every
    episode starts afresh: no term carries over."""

    current_idx: tuple[int, ...]
    target_idx: tuple[int, ...]
    dt: float = 1.0
    integral_limit: Optional[float] = None
    name: str = "pid_terms"
    observes_normalised: ClassVar[bool] = False

    def __post_init__(self) -> None:
        # Tuples: the extension is a static, hashed argument.
        object.__setattr__(self, "current_idx", tuple(self.current_idx))
        object.__setattr__(self, "target_idx", tuple(self.target_idx))
        check_pid_terms(self.current_idx, self.target_idx, self.dt, self.integral_limit)

    @property
    def n_errors(self) -> int:
        return len(self.current_idx)

    def observation_width(self, env_args: Any) -> int:
        del env_args
        return 3 * self.n_errors

    def init_observer(self, n_envs: int) -> PIDMemory:
        zeros = jnp.zeros((n_envs, self.n_errors))
        return PIDMemory(zeros, zeros, jnp.zeros((n_envs,), jnp.bool_))

    def observe(self, inputs: ObserveInputs, state: PIDMemory) -> tuple[jax.Array, Any]:
        flat = inputs.raw_obs.reshape(inputs.raw_obs.shape[0], -1)
        error = tracking_error(flat, self.current_idx, self.target_idx)
        fresh = start_terms(error, self.dt, self.integral_limit)
        going = advance_terms(
            error, state.error, state.integral, self.dt, self.integral_limit
        )
        started = state.started[:, None]
        integral, difference = (
            jnp.where(started, g, f) for g, f in zip(going, fresh, strict=False)
        )
        columns = jnp.concatenate([error, integral, difference], axis=-1)
        memory = PIDMemory(error, integral, jnp.ones_like(state.started))
        return columns.astype(jnp.float32), memory


def pid_terms_at(env_args: Any, n_errors: int) -> int:
    """The first column of the ``n_errors`` PID terms a PID actor reads: a
    :class:`PIDTermsInput`'s among the observation features, else the
    observation's first (a deprecated PIDTermsWrapper's)."""
    found = [e for e in env_args.observers if isinstance(e, PIDTermsInput)]
    if not found:
        return 0
    if len(found) > 1 or found[0].n_errors != n_errors:
        raise ValueError(
            f"A PIDPolicy with n_errors={n_errors} reads one PIDTermsInput of as"
            f" many errors, got {[e.n_errors for e in found]}."
        )
    env_width = get_state_action_shapes(env_args.env)[0][-1]
    layout = observation_layout(env_width, env_args.observers, env_args)
    return layout.slice(found[0].name).start


@dataclass(frozen=True)
class TrainingProgressInput(Extension):
    """The share of the run done (:func:`ajax.schedule.progress`), the same
    for every env: evaluation holds the collector's, as the deployed policy
    holds the value at the end of training."""

    name: str = "training_progress"
    #: It reads the run's horizon, which the collector then keeps.
    reads_progress: ClassVar[bool] = True

    def observation_width(self, env_args: Any) -> int:
        del env_args
        return 1

    def observe(self, inputs: ObserveInputs, state: Any) -> tuple[jax.Array, Any]:
        done = inputs.progress
        if done is None:
            done = progress(inputs.timestep, inputs.horizon)
        return jnp.full((inputs.raw_obs.shape[0], 1), done), state
