"""The Extension framework — composable research features for any agent.

An :class:`Extension` is a research feature (expert guidance, observation
augmentation, online behavioural cloning, target modifiers, exploration
schemes, instrumentation, …) that attaches to *any* agent without
bloating the agent's own file. It implements only the lifecycle *phases*
it touches; an agent's training loop folds every extension through each
phase, in list order.

Design contract
---------------
* **Static.** Extensions are plain Python objects passed as static
  arguments to the jitted training step, so the fold unrolls at trace
  time (a Python ``for`` over a fixed tuple — no dynamic dispatch). Every
  extension must therefore be *hashable*: the default id-based hash is
  fine for a one-off instance; a ``@dataclass(frozen=True)`` is preferred
  for instances carrying config, so the JIT cache is reused across
  equivalent runs.
* **State lives in the pytree.** An extension never stores mutable state
  on ``self`` — only frozen config. Anything that changes during
  training is a pytree returned by :meth:`Extension.init_state` and
  carried in ``BaseAgentState.ext_state`` (one entry per extension; the
  default ``()`` means stateless, zero overhead).
* **No-op by default.** Every phase method has a default that is the
  identity / ``0.0`` / ``{}`` / unchanged-state, so an extension is
  exactly as invasive as the phases it overrides, and an empty
  :class:`ExtensionStack` is genuinely zero-cost.

Phases
------
``init_state``   build the extension's pytree state (once, on fresh init)
``pretrain``     one-shot, before the training loop (MC/BC pre-training)
``on_target``    transform the TD / value target
``critic_loss``  extra additive critic-loss term (summed over extensions)
``actor_loss``   extra additive actor-loss term (summed over extensions)
``action``       override the collection-time action (``None`` = defer;
                 the last non-None proposal in the stack wins)
``eval_action``  override the evaluation-time action (same rule)
``transform_action`` make the executed action a function of the policy's
                 and the expert's (residual RL), in collection, the
                 losses and evaluation
``post_update``  hook after the update step (φ-refresh, schedules, …)
``eval_metrics`` extra metrics, merged into the eval log

Phase batches
-------------
The ``batch`` dict of a phase holds the keys :data:`PHASE_BATCH_KEYS`
lists on every agent that folds it; an agent may add its own (read its
fold call), and an extension raises on a key it needs and does not find,
rather than return a no-op. ``gamma`` is ``None`` for an average-reward
agent, ``dones`` is where the target cut the bootstrap and ``targets`` the
target, held constant; the critic (DQN's and PQN's ``q_state``) and the
actor come at the differentiated parameters. The replay agents'
``raw_observations`` is None in a run without an expert, whose replay
does not keep them. ``a_expert`` is the agent's ``expert_policy``'s action
at each row, stored at collection from the expert's state along the
trajectory; None without one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ExtensionContext:
    """Shared context handed to every extension phase.

    ``step`` and ``rng`` are traced; ``total_steps`` is static so it can
    drive Python-level decisions (e.g. schedule lengths) at trace time.
    """

    step: jax.Array  # current global timestep (traced)
    rng: jax.Array  # a PRNG key for this phase call (traced)
    total_steps: int = struct.field(pytree_node=False, default=0)


# Phase methods an Extension may override. Used for the construction-time
# applicability check (an extension implementing a phase an agent never
# folds is almost certainly a user error).
PHASES: tuple[str, ...] = (
    "pretrain",
    "on_target",
    "critic_loss",
    "actor_loss",
    "action",
    "eval_action",
    "transform_action",
    "post_update",
    "eval_metrics",
)

#: The keys of each phase's ``batch`` ("Phase batches" above): per phase,
#: the agents passing them ("*": every agent folding it) -> the keys.
PHASE_BATCH_KEYS: dict[str, dict[str, str]] = {
    "on_target": {
        "*": "observations next_observations rewards gamma",
        "SAC TD3 REDQ ASAC AVG DQN PQN": "actions dones reward_scale",
        "SAC TD3 REDQ ASAC AVG": "raw_next_observations next_a_expert",
        "PQN PPO APO": "terminated truncated",
        "PPO APO": "gae values",
    },
    "critic_loss": {
        "*": "observations targets",
        "SAC TD3 REDQ ASAC AVG PPO APO": "critic_params critic_state",
        "SAC TD3 REDQ ASAC AVG DQN PQN": "actions",
        "DQN PQN": "q_state",
    },
    "actor_loss": {
        "*": "actor_params",
        "SAC TD3 REDQ ASAC AVG PPO APO UDRL APG": "actor_state observations",
        "SAC TD3 REDQ ASAC AVG PPO APO APG": "raw_observations pi_mean",
        "SAC TD3 REDQ ASAC AVG APO": "a_expert",
        "SAC": "critic_state",
        "UDRL": "actions",
        "APG": "rollout returns",
    },
}


def check_extension_phases(
    agent_name: str,
    stack: Iterable["Extension"],
    supported: frozenset,
    supported_fields: frozenset = frozenset(),
    agent_expert: Any = None,
    stores_expert_actions: bool = False,
) -> None:
    """Reject the extensions of ``stack`` that implement phases outside
    ``supported`` (the phases ``agent_name``'s training loop folds), read
    buffer fields outside ``supported_fields`` (those the agent stores on
    an extension's declaration, ``Extension.buffer_fields``), or call a
    stateful expert without its state (:func:`check_extension_experts`).

    Raises ``ValueError`` naming every offending extension and phase: an
    extension whose phase an agent never folds would otherwise be silently
    ignored; also on a reader of phi* without MCPretrain
    (:func:`check_phi_star_source`). Agents call it at construction
    (``ActorCritic.__init__``).
    """
    stack = tuple(stack)
    check_extension_experts(agent_name, stack, agent_expert, stores_expert_actions)
    missing = frozenset().union(*(e.buffer_fields for e in stack)) - supported_fields
    if missing:
        raise ValueError(
            f"{agent_name} does not store the buffer fields {sorted(missing)}"
            " its extensions read."
        )
    unsupported = [(ext, sorted(ext.implemented_phases() - supported)) for ext in stack]
    details = "; ".join(
        f"extension {ext.name!r} ({type(ext).__name__}) implements {phases}"
        for ext, phases in unsupported
        if phases
    )
    if details:
        raise ValueError(
            f"{agent_name} does not support these extension phases: {details}."
            f" Supported phases: {sorted(supported)}."
        )
    check_phi_star_source(agent_name, stack)


def check_phi_star_source(agent_name: str, stack: Sequence["Extension"]) -> None:
    """Reject extensions reading the frozen expert critic phi* in a stack
    without one that fits it (MCPretrain): without phi* they would add
    nothing all run (OnlineBC, CriticBlend, MCVarianceCorrection) or fail at
    trace time (ValueBox, PhiRefresh)."""
    readers = [type(ext).__name__ for ext in stack if ext.reads_phi_star()]
    if readers and not any(ext.fits_phi_star for ext in stack):
        raise ValueError(
            f"{', '.join(readers)} read the frozen expert critic phi*, which"
            f" MCPretrain fits, and {agent_name}'s extensions hold no"
            " MCPretrain: add MCPretrain(expert_policy=...) to them."
        )


def holds_expert(agent_expert: Any, stack: Iterable["Extension"]) -> bool:
    """Whether a run has an expert: the agent's ``expert_policy``, or one an
    extension holds (its ``expert_fields``). Replay keeps the raw
    observations and the expert flags, which only experts read, for such
    a run (init_buffer's include_raw_obs, include_is_expert)."""
    return agent_expert is not None or any(
        getattr(ext, field, None) is not None
        for ext in stack
        for field in ext.expert_fields
    )


def check_extension_experts(
    agent_name: str,
    stack: Sequence["Extension"],
    agent_expert: Any = None,
    stores_expert_actions: bool = False,
) -> None:
    """Reject an extension calling a stateful expert (one with
    ``init_state``, called ``expert(state, obs)``) without its state, and
    one reading the actions of the agent's expert (``needs_agent_expert``)
    on an agent without an ``expert_policy``, where it would add nothing.

    An extension calls the experts it declares (:meth:`Extension.experts`)
    on the observations it is given, where no state of the expert exists:
    a stateful one is allowed only as the agent's ``expert_policy`` of an
    agent that stores that expert's actions from its state
    (``stores_expert_actions``), for an extension that reads them
    (``reads_agent_expert_actions``). Stateless experts are always allowed.
    """
    needing = [type(ext).__name__ for ext in stack if ext.needs_agent_expert]
    if needing and agent_expert is None:
        raise ValueError(
            f"{', '.join(needing)} read the actions of the agent's"
            f" expert_policy, and {agent_name} has none: without it they would"
            " add nothing all run. Give the agent an expert_policy."
        )
    for ext in stack:
        for expert in ext.experts():
            if not hasattr(expert, "init_state"):
                continue
            stored = expert is agent_expert and stores_expert_actions
            if stored and ext.reads_agent_expert_actions:
                continue
            raise ValueError(
                f"{type(ext).__name__} calls its expert without a state, and this"
                " expert is stateful (it has init_state). An extension uses a"
                " stateful expert only as the agent's expert_policy, whose"
                " actions the agent stores from the expert's state, and only"
                " if the extension reads them; here"
                + (
                    f" {type(ext).__name__} does not read them."
                    if stored
                    else f" {agent_name} does not store this expert's actions."
                )
                + " To keep only the extension's use of the expert, give the"
                " agent the same expert as its expert_policy, on an agent that"
                " stores its actions (stores_expert_actions; SAC with"
                " expert_buffer_n_steps=0 (no expert prefill) and"
                " expert_fraction=0.0 (no warm-up mixing)); or give the"
                " extension a stateless expert."
            )


class Extension:
    """Base class for a composable research feature.

    Subclass and override only the phases the feature touches. Keep
    subclasses hashable (``@dataclass(frozen=True)`` is the recommended
    form) and stateless on ``self`` — see the module docstring.
    """

    #: Human-readable label, used in logs and the applicability check.
    name: str = "extension"
    #: Replay-buffer fields beyond the agent's own that this extension reads
    #: from the batches it is given (``raw_next_obs``: the environment's own
    #: next observation). An agent stores a field only when an extension of
    #: its stack declares it, and rejects one it cannot provide.
    buffer_fields: ClassVar[frozenset[str]] = frozenset()
    #: Whether the extension reads the agent's stored actions of its expert
    #: when that expert is the agent's (check_extension_experts).
    reads_agent_expert_actions: ClassVar[bool] = False
    #: Whether the extension reads the actions of the agent's own
    #: ``expert_policy``, refused on an agent without one.
    needs_agent_expert: ClassVar[bool] = False
    #: The fields holding an expert (raw observations in, env-unit actions
    #: out), which :meth:`ExtensionStack.with_experts` converts.
    expert_fields: ClassVar[frozenset[str]] = frozenset()
    #: Whether the extension fits the frozen expert critic phi* that the
    #: extensions which :meth:`reads_phi_star` need (MCPretrain does).
    fits_phi_star: ClassVar[bool] = False
    #: Whether the extension reads each env's step within its episode
    #: (``CollectorState.step_in_episode``), which an agent counts only then.
    reads_step_in_episode: ClassVar[bool] = False

    # -- lifecycle -------------------------------------------------------
    def init_state(self, agent_state: Any, rng: jax.Array) -> Any:
        """Return this extension's initial pytree state.

        Called once, on fresh initialisation only (skipped on resume).
        The default ``()`` marks a stateless extension — no pytree
        overhead in ``BaseAgentState.ext_state``.
        """
        del agent_state, rng
        return ()

    def pretrain(
        self, agent_state: Any, ext_state: Any, ctx: ExtensionContext
    ) -> tuple[Any, Any]:
        """One-shot step before the training loop (e.g. MC/BC pre-train).

        Returns the (possibly updated) ``(agent_state, ext_state)``.
        """
        del ctx
        return agent_state, ext_state

    # -- per-iteration transforms (state-read-only) ----------------------
    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: Any,
        target: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array:
        """Transform the TD / value target."""
        del agent_state, ext_state, batch, ctx
        return target

    def critic_loss(
        self, agent_state: Any, ext_state: Any, batch: Any, ctx: ExtensionContext
    ) -> jax.Array | float:
        """Extra additive critic-loss term (summed across extensions)."""
        del agent_state, ext_state, batch, ctx
        return 0.0

    def actor_loss(
        self, agent_state: Any, ext_state: Any, batch: Any, ctx: ExtensionContext
    ) -> jax.Array | float:
        """Extra additive actor-loss term (summed across extensions)."""
        del agent_state, ext_state, batch, ctx
        return 0.0

    # -- action overrides (across a stack, the last non-None wins) -------
    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: jax.Array,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        """Override the collection-time action; ``None`` defers."""
        del agent_state, ext_state, obs, rng, ctx
        return None

    def eval_action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: jax.Array,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        """Override the evaluation-time action; ``None`` defers."""
        del agent_state, ext_state, obs, rng, ctx
        return None

    def transform_action(self, actions: jax.Array, a_expert: jax.Array) -> jax.Array:
        """The executed action from the policy's ``actions`` and the
        expert's ``a_expert``; the policy's by default."""
        del a_expert
        return actions

    # -- state-threading + logging --------------------------------------
    def post_update(
        self, agent_state: Any, ext_state: Any, ctx: ExtensionContext
    ) -> tuple[Any, Any]:
        """Hook after the update step. Returns ``(agent_state, ext_state)``."""
        del ctx
        return agent_state, ext_state

    def eval_metrics(
        self, agent_state: Any, ext_state: Any, rng: jax.Array, ctx: ExtensionContext
    ) -> dict:
        """Extra metrics, merged into the eval log."""
        del agent_state, ext_state, rng, ctx
        return {}

    # -- introspection ---------------------------------------------------
    def experts(self) -> tuple:
        """The experts this extension calls itself, on the observations it
        is given (without a state); none by default."""
        return ()

    def reads_phi_star(self) -> bool:
        """Whether the extension reads phi*: a stack holding one and none
        that ``fits_phi_star`` is refused (:func:`check_phi_star_source`)."""
        return False

    def implemented_phases(self) -> frozenset[str]:
        """The phases this extension actually overrides (vs the no-op base)."""
        return frozenset(
            phase
            for phase in PHASES
            if getattr(type(self), phase) is not getattr(Extension, phase)
        )

    # -- agent-context binding ------------------------------------------
    def bind_to_agent(self, **agent_context: Any) -> "Extension":
        """Return ``self``, optionally populating closed-over agent context.

        The default is the identity: an extension ignores every kwarg it
        does not need. One that closes over values the agent factory
        resolves (env_args, buffer, …; e.g. :class:`MCPretrain`,
        :class:`PhiRefresh`) overrides this to return a new frozen instance
        with them filled in, so a whole stack binds without per-extension
        type checks (:meth:`ExtensionStack.bind_to_agent`).
        """
        del agent_context
        return self


class ExtensionStack:
    """A folded, immutable collection of extensions.

    Holds the static tuple of :class:`Extension` objects and folds them
    through each phase in list order. An empty stack is a true no-op.
    Hashable (so it can be a JIT static argument) iff its extensions are.
    """

    __slots__ = ("extensions",)

    def __init__(self, extensions: Sequence[Extension] = ()):
        self.extensions: tuple[Extension, ...] = tuple(extensions)

    def __len__(self) -> int:
        return len(self.extensions)

    def __bool__(self) -> bool:
        return bool(self.extensions)

    def __iter__(self):
        return iter(self.extensions)

    def __hash__(self) -> int:
        return hash(self.extensions)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExtensionStack) and self.extensions == other.extensions

    def __repr__(self) -> str:
        names = ", ".join(e.name for e in self.extensions)
        return f"ExtensionStack({names})"

    def implemented_phases(self) -> frozenset[str]:
        """The phases at least one extension of the stack implements."""
        return frozenset().union(*(e.implemented_phases() for e in self.extensions))

    def buffer_fields(self) -> frozenset[str]:
        """The replay-buffer fields the stack's extensions declare they read."""
        return frozenset().union(*(e.buffer_fields for e in self.extensions))

    def _keys(self, rng: jax.Array) -> tuple[jax.Array, ...]:
        """One sub-key per extension (a fold never splits for an empty stack)."""
        return tuple(jax.random.split(rng, len(self.extensions)))

    # ``fold_<phase>``: each fold builds the phase's ExtensionContext and
    # reads (and, for state-threading phases, writes)
    # ``agent_state.ext_state``, in list order. On an empty stack it returns
    # its input unchanged without building a context, so an agent without
    # extensions traces nothing extra and never needs to guard a fold.

    def fold_init(self, agent_state: Any, rng: jax.Array, total_steps: int) -> Any:
        """A fresh run's extension setup: every extension's initial state,
        then its one-shot ``pretrain`` at step 0 (``rng`` split between the
        two). Skipped on resume, where the state already carries both."""
        if not self.extensions:
            return agent_state
        init_key, pretrain_key = jax.random.split(rng)
        agent_state = self.fold_init_states(agent_state, init_key)
        return self.fold_pretrain(
            agent_state, jnp.asarray(0), pretrain_key, total_steps
        )

    def fold_init_states(self, agent_state: Any, rng: jax.Array) -> Any:
        """``agent_state`` with every extension's initial state (one key
        each) in its ``ext_state``."""
        if not self.extensions:
            return agent_state
        keys = self._keys(rng)
        return agent_state.replace(
            ext_state=tuple(
                ext.init_state(agent_state, keys[i])
                for i, ext in enumerate(self.extensions)
            )
        )

    def fold_pretrain(
        self,
        agent_state: Any,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> Any:
        """One-shot pretrain fold; returns the updated ``agent_state``."""
        if not self.extensions:
            return agent_state
        ctx = ExtensionContext(step=step, rng=rng, total_steps=total_steps)
        ext_states = list(agent_state.ext_state)
        for i, ext in enumerate(self.extensions):
            agent_state, ext_states[i] = ext.pretrain(agent_state, ext_states[i], ctx)
        return agent_state.replace(ext_state=tuple(ext_states))

    def fold_post_update(
        self,
        agent_state: Any,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> Any:
        """Post-update fold; returns the updated ``agent_state``."""
        if not self.extensions:
            return agent_state
        ctx = ExtensionContext(step=step, rng=rng, total_steps=total_steps)
        ext_states = list(agent_state.ext_state)
        for i, ext in enumerate(self.extensions):
            agent_state, ext_states[i] = ext.post_update(
                agent_state, ext_states[i], ctx
            )
        return agent_state.replace(ext_state=tuple(ext_states))

    def fold_on_target(
        self,
        agent_state: Any,
        batch: Any,
        target: jax.Array,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> jax.Array:
        """Target-modifier fold; returns the (possibly reshaped) target."""
        if not self.extensions:
            return target
        ctx = ExtensionContext(step=step, rng=rng, total_steps=total_steps)
        ext_states = agent_state.ext_state
        for i, ext in enumerate(self.extensions):
            target = ext.on_target(agent_state, ext_states[i], batch, target, ctx)
        return target

    def fold_critic_loss(
        self,
        agent_state: Any,
        batch: Any,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> jax.Array | float:
        """Critic-loss fold; returns the summed additive critic-loss term."""
        if not self.extensions:
            return 0.0
        ctx = ExtensionContext(step=step, rng=rng, total_steps=total_steps)
        total: jax.Array | float = 0.0
        ext_states = agent_state.ext_state
        for i, ext in enumerate(self.extensions):
            total = total + ext.critic_loss(agent_state, ext_states[i], batch, ctx)
        return total

    def fold_actor_loss(
        self,
        agent_state: Any,
        batch: Any,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> jax.Array | float:
        """Actor-loss fold; returns the summed additive actor-loss term."""
        if not self.extensions:
            return 0.0
        ctx = ExtensionContext(step=step, rng=rng, total_steps=total_steps)
        total: jax.Array | float = 0.0
        ext_states = agent_state.ext_state
        for i, ext in enumerate(self.extensions):
            total = total + ext.actor_loss(agent_state, ext_states[i], batch, ctx)
        return total

    def fold_eval_metrics(
        self,
        agent_state: Any,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> dict:
        """Eval-metrics fold; returns the extensions' merged metrics (each
        extension gets its own key; the context carries ``rng``)."""
        if not self.extensions:
            return {}
        ctx = ExtensionContext(step=step, rng=rng, total_steps=total_steps)
        out: dict = {}
        ext_states, keys = agent_state.ext_state, self._keys(rng)
        for i, ext in enumerate(self.extensions):
            out.update(ext.eval_metrics(agent_state, ext_states[i], keys[i], ctx))
        return out

    def bind_to_agent(self, **agent_context: Any) -> "ExtensionStack":
        """Return a new stack with every extension bound to ``agent_context``
        (each :meth:`Extension.bind_to_agent` picks the kwargs it needs);
        each agent factory passes the context it owns. An empty stack
        returns ``self``.
        """
        if not self.extensions:
            return self
        return ExtensionStack(
            tuple(ext.bind_to_agent(**agent_context) for ext in self.extensions)
        )

    def with_experts(self, convert: Callable[[Any], Any]) -> "ExtensionStack":
        """The stack with ``convert`` applied to every expert its extensions
        hold (their ``expert_fields``; one still None is left). A
        ``convert`` memoised by identity keeps one expert one object, which
        :func:`ajax.extensions.expert.reads_stored_expert_actions` compares."""

        def converted(ext: Extension) -> Extension:
            experts = {f: getattr(ext, f) for f in ext.expert_fields}
            changes = {f: convert(e) for f, e in experts.items() if e is not None}
            if not changes:
                return ext
            # An extension holding an expert is a dataclass (the recommended
            # frozen form, which the field test enforces); the base is not.
            return dataclasses.replace(ext, **changes)  # type: ignore[type-var]

        return ExtensionStack(tuple(converted(ext) for ext in self.extensions))
