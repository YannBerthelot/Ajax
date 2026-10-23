"""What a policy acts on, the same in collection and evaluation.

The controller is everything a policy does when it acts, the part that would
ship with it (side-session design note ``controller-design.md``): its
features, normaliser, expert, carry, choice of action and action transforms.
A step acts on an :class:`ActingState`. Collection builds it from the
collector and the actor (:func:`acting_state`), and evaluation from its own
rollout, so the action pipelines read one state whichever runs them.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp


class ActingState(NamedTuple):
    """What one step acts on, per env: the observation the actor reads
    (before any normalisation), where the last step ended an episode (the
    carry and the expert restart there), the global timestep, the tracked
    expert's state (None: no stateful expert), the actor's carry (None: a
    feedforward actor) and each env's step in its episode (None unless an
    extension reads it)."""

    obs: jax.Array
    done: jax.Array
    timestep: jax.Array
    expert_state: Any = None
    carry: Any = None
    step_in_episode: Optional[jax.Array] = None


def acting_state(agent_state: Any) -> ActingState:
    """A collecting agent's acting state, its collector's and its actor's."""
    collector = agent_state.collector_state
    return ActingState(
        obs=collector.last_obs,
        done=jnp.logical_or(collector.last_terminated, collector.last_truncated),
        timestep=collector.timestep,
        expert_state=collector.expert_state,
        carry=agent_state.actor_state.hidden_state,
        step_in_episode=collector.step_in_episode,
    )
