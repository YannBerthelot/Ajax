"""What a policy acts on, the same in collection and evaluation.

The controller is everything a policy does when it acts, the part that would
ship with it (side-session design note ``controller-design.md``): its
features, normaliser, expert, carry, choice of action and action transforms.
A step acts on an :class:`ActingState`. Collection builds it from the
collector and the actor (:func:`acting_state`), and evaluation from its own
rollout, so the action pipelines read one state whichever runs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from ajax.networks.memory import flatten_carry


class ActingState(NamedTuple):
    """What one step acts on, per env: the observation the actor reads
    (before any normalisation), where the last step ended an episode (the
    carry and the expert restart there), the global timestep, the tracked
    expert's state (None: no stateful expert), the actor's carry (None: a
    feedforward actor) and each env's step in its episode (None unless an
    extension reads it). Evaluation acts on neither a timestep nor a step
    in an episode: None there."""

    obs: jax.Array
    done: jax.Array
    timestep: Optional[jax.Array]
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


def flatten_expert_state(expert_state: Any) -> Optional[jax.Array]:
    """A stateful expert's state as one ``(n_envs, D)`` array
    (:func:`~ajax.networks.memory.flatten_carry`): the Markov component an
    observation augmented with it carries (a PID's integrator). None for a
    stateless expert (no state, or an empty one)."""
    if expert_state is None or not jax.tree.leaves(expert_state):
        return None
    return flatten_carry(expert_state)


def augment_obs(
    obs: jax.Array,
    expert_state: Optional[jax.Array] = None,
    expert_action: Optional[jax.Array] = None,
) -> jax.Array:
    """The observation of an agent augmented with its expert, in the one
    layout every site builds (collection, the replay buffer, SAC's update,
    the pretraining and cloning, the evaluation): the agent's observation
    (its training-fraction column last when it has one), the expert's
    flattened state (:func:`flatten_expert_state`), then the expert's
    action. A part given as None is left out. Appending keeps the agent's
    own observation leading, as the observation normaliser reads it."""
    extra = [x for x in (expert_state, expert_action) if x is not None]
    return jnp.concatenate([obs, *map(jax.lax.stop_gradient, extra)], axis=-1)


def call_expert(expert: Any, state: Any, obs: Optional[jax.Array]) -> tuple[Any, Any]:
    """``(action, state)``: a stateful expert (one with ``init_state``)
    called as ``expert(state, obs)``, a stateless one as ``expert(obs)``,
    its state kept."""
    if hasattr(expert, "init_state"):
        return expert(state, obs)
    return expert(obs), state


@dataclass(frozen=True)
class Controller:
    """How an agent acts beyond its actor's forward pass, which evaluation
    runs (collection still runs the agent's pipeline): the expert in the
    actor's input (SAC's ``observe_expert_action`` and
    ``observe_expert_state``), and the executed action made from the
    actor's: gain mode's (``gains``: the actor's action sets an expert's
    gains, ``anchor * 10**a``, and the expert acts), or ``transform``'s
    ``(actions, expert_actions, obs, agent_state) -> actions`` (residual
    RL's). The default executes the actor's action on its observation. The
    expert is the caller's, in agent units. Hashable: a static argument."""

    observe_expert_action: bool = False
    observe_expert_state: bool = False
    transform: Optional[Callable] = None
    gains: bool = False

    def observe(
        self, acting: ActingState, expert: Any, expert_obs: Optional[jax.Array]
    ) -> tuple[jax.Array, Any]:
        """The actor's input, and the expert's state after the call made
        for it (None when the input does not read the expert): ``acting``'s
        observation, then the expert's state before this step's call and its
        action where the controller observes them (:func:`augment_obs`)."""
        expert_action, observed_state = None, None
        if (self.observe_expert_action or self.observe_expert_state) and (
            expert is not None
        ):
            expert_action, observed_state = call_expert(
                expert, acting.expert_state, expert_obs
            )
        stateful = expert is not None and hasattr(expert, "init_state")
        actor_obs = augment_obs(
            acting.obs,
            (
                flatten_expert_state(acting.expert_state)
                if self.observe_expert_state and stateful
                else None
            ),
            expert_action if self.observe_expert_action else None,
        )
        return actor_obs, observed_state

    def act(
        self,
        actions: jax.Array,
        acting: ActingState,
        expert: Any,
        expert_obs: Optional[jax.Array],
        agent_state: Any,
        observed_state: Any,
    ) -> tuple[jax.Array, Any]:
        """The executed action from the actor's ``actions``, and the
        expert's state after the step: gain mode's expert acting on the
        gains, the transform's action (the expert called for it, its action
        0 without one), else the actor's own (the expert's state as
        :meth:`observe` left it)."""
        if self.gains:
            gains = expert.anchor_gains * jnp.exp(jnp.log(10.0) * actions)
            return expert.step_with_gains(acting.expert_state, expert_obs, gains)
        if self.transform is not None:
            expert_actions: Any = 0.0
            expert_state = acting.expert_state
            if expert is not None:
                expert_actions, expert_state = call_expert(
                    expert, acting.expert_state, expert_obs
                )
            return (
                self.transform(actions, expert_actions, acting.obs, agent_state),
                expert_state,
            )
        return actions, (
            acting.expert_state if observed_state is None else observed_state
        )
