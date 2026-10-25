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
from functools import partial
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from ajax.environments.obs_norm import normalized_obs
from ajax.environments.utils import critic_action_units as env_critic_action_units
from ajax.environments.utils import get_action_dim
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
    # The tracked expert's action this step, which collection computes once
    # before the action is chosen (None: no expert, or not computed yet).
    expert_action: Optional[jax.Array] = None


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


class TransformInputs(NamedTuple):
    """What an action transform reads, per env: the agent's tracked expert
    (agent units), the state its call this step starts from, the env's own
    observation and the expert's action there. The learner, scoring stored
    rows, passes the stored action alone."""

    expert: Any = None
    expert_state: Any = None
    raw_obs: Optional[jax.Array] = None
    expert_action: Optional[jax.Array] = None


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


def transforms_of(stack: Any) -> tuple[Any, ...]:
    """The extensions of ``stack`` that transform the action (residual RL's
    ``transform_action``, gain mode's ``actuate``)."""
    phases = {"transform_action", "actuate"}
    return tuple(e for e in stack if e.implemented_phases().intersection(phases))


class WarmupInputs(NamedTuple):
    """What a warm-up chooser reads in one step: the expert's action in the
    space the warm-up draws (expert_action_for_warmup), and the key the
    step's uniform actions were drawn on."""

    expert_action: jax.Array
    key: jax.Array


def warmups_of(extensions: Any) -> tuple[Any, ...]:
    """The extensions choosing what the warm-up acts with (ExpertWarmup)."""
    return tuple(
        extension
        for extension in extensions
        if "choose_warmup_action" in extension.implemented_phases()
    )


def choose_warmup_actions(
    warmups: Any, uniform_actions: jax.Array, inputs: WarmupInputs
) -> tuple[jax.Array, jax.Array]:
    """The warm-up's actions and whether the expert took them (one boolean
    for the step): the run's chooser's (an agent takes one), else the
    uniform ones."""
    if not warmups:
        return uniform_actions, jnp.zeros((), dtype=jnp.bool_)
    return warmups[0].choose_warmup_action(uniform_actions, inputs)


def expert_action_for_warmup(
    transform: Any, after_choice: bool, expert_action: jax.Array, policy_actions: Any
) -> jax.Array:
    """The expert's action in the space the warm-up draws, which the replay
    stores: the executed one, unless the transform acts after the choice
    (gain mode, or a residual applied after it), where it is the policy
    action that executes the expert's own (policy_action_for_expert)."""
    acts_after_choice = transform is not None and (
        after_choice or "actuate" in transform.implemented_phases()
    )
    if not acts_after_choice:
        return expert_action
    return transform.policy_action_for_expert(policy_actions)


def warmup_expert_flags(expert_took: jax.Array, policy_actions: jax.Array) -> jax.Array:
    """A warm-up step's is_expert column: 1 in every env where the expert
    took it."""
    flags = jnp.ones_like(policy_actions[..., :1], dtype=jnp.float32)
    return flags * expert_took.astype(jnp.float32)


def executed_actions(env_args: Any, actions: jax.Array, a_expert: Any) -> jax.Array:
    """The actions the critics score: the policy's ``actions`` through the
    run's ``transform_action`` (residual RL), from the expert's actions
    ``a_expert`` at the same rows; the policy's own without one. Gain mode
    stores the policy's actions, so its critics score them as they are."""
    for transform in env_args.transforms:
        if "transform_action" in transform.implemented_phases():
            inputs = TransformInputs(expert_action=a_expert)
            return transform.transform_action(actions, inputs)
    return actions


def executed_fn(env_args: Any) -> Optional[Callable[[jax.Array, Any], jax.Array]]:
    """``(actions, a_expert) -> actions`` the critics score, where the run has
    a residual policy (:func:`executed_actions`); None otherwise, so the
    learner traces nothing extra."""
    if any("transform_action" in t.implemented_phases() for t in env_args.transforms):
        return partial(executed_actions, env_args)
    return None


def policy_action_dim(env_args: Any) -> int:
    """The dimension of the actions the policy chooses: gain mode's, the
    expert's learnable gains (``env_args.policy_action_dim``), else the
    env's."""
    if env_args.policy_action_dim is not None:
        return env_args.policy_action_dim
    return get_action_dim(env_args.env, env_args.env_params)


def critic_action_units(env_args: Any) -> Any:
    """The critic's map of its action input to the env's units
    (:func:`ajax.environments.utils.critic_action_units`); none under gain
    mode, whose critics read the gains."""
    if env_args.policy_action_dim is not None:
        return None
    return env_critic_action_units(env_args.env, env_args.env_params)


def reads_progress(env_args: Any) -> bool:
    """Whether a feature of ``env_args.observers`` reads the training
    progress, whose horizon the collector then keeps (``max_timesteps``)."""
    return any(getattr(e, "reads_progress", False) for e in env_args.observers)


def reads_agent_expert(env_args: Any) -> bool:
    """Whether a feature of ``env_args.observers`` reads the agent's expert."""
    return any("agent_expert" in e.requires for e in env_args.observers)


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
    runs (collection still runs the agent's pipeline): its observation
    features (``observers``, :attr:`EnvironmentConfig.observers`), the
    observation normalised (``normalize_obs``: an agent storing its
    observations normalised, which collection stores so already; the leading
    ``raw_columns`` stay raw), and the executed action made from the actor's
    by ``transforms`` (:attr:`EnvironmentConfig.transforms`): residual RL's
    ``transform_action``, or gain mode's ``actuate``, where the actor's action
    sets the expert's gains and the expert acts. The default executes the
    actor's action on its observation. The expert is the caller's, in agent
    units. Hashable: a static argument."""

    normalize_obs: bool = False
    raw_columns: int = 0
    # The observation features' extensions (EnvironmentConfig.observers),
    # which evaluation appends to every observation as collection does.
    observers: tuple = ()
    # The action transforms' extensions (EnvironmentConfig.transforms).
    transforms: tuple = ()

    @property
    def actuates(self) -> bool:
        """Whether the actor's action is actuated through the expert (gain
        mode): the critics score the actor's action, not the executed one."""
        return any("actuate" in t.implemented_phases() for t in self.transforms)

    def observe(self, acting: ActingState) -> jax.Array:
        """The actor's input: ``acting``'s observation, its features
        included, normalised with its statistics where the agent stores its
        observations so."""
        if self.normalize_obs:
            return normalized_obs(acting.obs, acting.obs_norm, self.raw_columns)
        return acting.obs

    def act(
        self,
        actions: jax.Array,
        acting: ActingState,
        expert: Any,
        expert_obs: Optional[jax.Array],
    ) -> tuple[jax.Array, Any]:
        """The executed action from the actor's ``actions``, and the
        expert's state after the step: the transforms' (the expert called
        for them, its action 0 without one; gain mode's actuation steps the
        expert itself), else the actor's own. The expert is called every
        step its state is read, as collection calls it: for a transform, or
        for the features; otherwise its state is left as it is."""
        if self.transforms:
            expert_actions: Any = 0.0
            expert_state = acting.expert_state
            if expert is not None:
                expert_actions, expert_state = call_expert(
                    expert, acting.expert_state, expert_obs
                )
            inputs = TransformInputs(
                expert, acting.expert_state, expert_obs, expert_actions
            )
            for transform in self.transforms:
                if "actuate" in transform.implemented_phases():
                    return transform.actuate(actions, inputs)
                actions = transform.transform_action(actions, inputs)
            return actions, expert_state
        if self.observers and expert is not None:
            _, expert_state = call_expert(expert, acting.expert_state, expert_obs)
            return actions, expert_state
        return actions, acting.expert_state
