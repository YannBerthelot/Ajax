"""Observation features: columns an agent appends to every observation.

Each is an :class:`~ajax.extensions.base.Extension` implementing the
``observe`` phase (:mod:`ajax.controller`): its columns join the
observation whenever one arrives, in collection and evaluation alike, and
are stored with it, so the actor, the critics and the replay all read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from ajax.controller import ObserveInputs, call_expert, flatten_expert_state
from ajax.environments.utils import get_action_dim
from ajax.extensions.base import Extension
from ajax.schedule import progress


@dataclass(frozen=True)
class ExpertActionInput(Extension):
    """The agent's expert's action at the observation, called from the
    state it holds there, in agent units: raw, since it is an action."""

    name: str = "expert_action"
    observes_normalised: ClassVar[bool] = False
    requires: ClassVar[frozenset[str]] = frozenset({"agent_expert"})

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
