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

from ajax.environments.obs_norm import normalized_obs
from ajax.networks.memory import flatten_carry
from ajax.types import NormalizationInfo


class ActingState(NamedTuple):
    """What one step acts on, per env: the observation the actor reads
    (before any normalisation), where the last step ended an episode (the
    carry and the expert restart there), the global timestep, the tracked
    expert's state (None: no stateful expert), the actor's carry (None: a
    feedforward actor), each env's step in its episode (None unless an
    extension reads it) and the observation normaliser's statistics (None:
    none). Evaluation acts on neither a timestep nor a step in an episode:
    None there."""

    obs: jax.Array
    done: jax.Array
    timestep: Optional[jax.Array]
    expert_state: Any = None
    carry: Any = None
    step_in_episode: Optional[jax.Array] = None
    obs_norm: Optional[NormalizationInfo] = None


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
        obs_norm=collector.obs_norm_info,
    )


class ObserveInputs(NamedTuple):
    """What an observation feature reads, per env: the env's own observation
    the features are for (``raw_obs``, before any normalisation), the agent's
    tracked expert (in agent units; None without one) and the state its call
    there starts from, the collector's timestep (None in evaluation, which
    passes its frozen ``progress``), the run's horizon (None where the
    collector keeps none) and, in evaluation, the training progress it
    holds."""

    raw_obs: jax.Array
    expert: Any = None
    expert_state: Any = None
    timestep: Any = None
    horizon: Any = None
    progress: Any = None


@dataclass(frozen=True)
class ObservationLayout:
    """The columns of an observation with features: the env's own
    (``env_width``), then each feature's in ``columns`` order, ``(name,
    width, normalised)``: the normalised ones first, then the raw ones, so
    the normaliser's statistics cover the leading ``normalised_width``."""

    env_width: int
    columns: tuple[tuple[str, int, bool], ...] = ()

    @property
    def normalised_width(self) -> int:
        return self.env_width + sum(w for _, w, norm in self.columns if norm)

    @property
    def feature_width(self) -> int:
        return sum(w for _, w, _ in self.columns)

    @property
    def width(self) -> int:
        return self.env_width + self.feature_width

    def slice(self, name: str) -> slice:
        """The columns of the feature ``name``."""
        start = self.env_width
        for feature, width, _ in self.columns:
            if feature == name:
                return slice(start, start + width)
            start += width
        raise KeyError(name)


def observers_of(stack: Any) -> tuple[Any, ...]:
    """The extensions of ``stack`` that append observation features, in
    column order: those the normaliser covers first, then the raw ones, each
    in stack order."""
    found = [e for e in stack if "observe" in e.implemented_phases()]
    return tuple(e for e in found if e.observes_normalised) + tuple(
        e for e in found if not e.observes_normalised
    )


def observation_layout(
    env_width: int, observers: Any, env_args: Any
) -> ObservationLayout:
    """The layout of an env's ``env_width`` columns followed by
    ``observers``' (:func:`observers_of`)."""
    return ObservationLayout(
        env_width,
        tuple(
            (e.name, e.observation_width(env_args), e.observes_normalised)
            for e in observers
        ),
    )


def init_observers(observers: Any, n_envs: int) -> tuple:
    """Each observer's per-env state at an episode start."""
    return tuple(e.init_observer(n_envs) for e in observers)


def feature_columns(
    inputs: ObserveInputs, observers: Any, states: tuple
) -> tuple[jax.Array, tuple]:
    """``observers``' columns at the observation ``inputs`` describes, in
    column order, and their states after it."""
    columns, new_states = [], []
    for observer, state in zip(observers, states, strict=True):
        column, state = observer.observe(inputs, state)
        columns.append(column)
        new_states.append(state)
    return jnp.concatenate(columns, axis=-1), tuple(new_states)


def observe_features(
    raw_obs: jax.Array, inputs: ObserveInputs, observers: Any, states: tuple
) -> tuple[jax.Array, tuple]:
    """``raw_obs`` with ``observers``' columns appended, and their states
    after this observation; without observers, ``raw_obs`` itself."""
    if not observers:
        return raw_obs, states
    columns, states = feature_columns(inputs, observers, states)
    return jnp.concatenate([raw_obs, columns], axis=-1), states


def feature_width(env_args: Any, normalised: Optional[bool] = None) -> int:
    """The columns ``env_args.observers`` append (those the normaliser
    covers with ``normalised`` True, the raw ones with False)."""
    return sum(
        e.observation_width(env_args)
        for e in env_args.observers
        if normalised is None or e.observes_normalised == normalised
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
    runs (collection still runs the agent's pipeline): the observation
    normalised (``normalize_obs``: an agent storing its observations
    normalised, which collection stores so already; the leading
    ``raw_columns`` stay raw), the expert in the actor's input (SAC's
    ``observe_expert_action`` and ``observe_expert_state``), and the
    executed action made from the actor's: gain mode's (``gains``: the
    actor's action sets an expert's gains, ``anchor * 10**a``, and the
    expert acts), or ``transform``'s ``(actions, expert_actions) ->
    actions`` (residual RL's ``ResidualPolicy.transform_action``, the one
    collection applies). The default executes the actor's action on its
    observation. The expert is the caller's, in agent units. Hashable: a
    static argument."""

    normalize_obs: bool = False
    raw_columns: int = 0
    # The observation features' extensions (EnvironmentConfig.observers),
    # which evaluation appends to every observation as collection does.
    observers: tuple = ()
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
        obs = acting.obs
        if self.normalize_obs:
            obs = normalized_obs(obs, acting.obs_norm, self.raw_columns)
        actor_obs = augment_obs(
            obs,
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
            return self.transform(actions, expert_actions), expert_state
        return actions, (
            acting.expert_state if observed_state is None else observed_state
        )
