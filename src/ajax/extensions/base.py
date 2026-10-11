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
``post_update``  hook after the update step (φ-refresh, schedules, …)
``eval_metrics`` extra metrics, merged into the eval log

Phase batches
-------------
The ``batch`` dict of a phase holds these keys on every agent that folds
it; an agent may add its own (read its fold call), and an extension
raises on a key it needs and does not find, rather than return a no-op.

``on_target``    ``observations``, ``next_observations``, ``rewards`` and
                 ``gamma`` (``None`` for an average-reward agent); the
                 replay agents and PQN add ``actions``, ``dones`` (where
                 the target cut the bootstrap) and ``reward_scale``, PQN
                 also ``terminated`` and ``truncated``, PPO and APO
                 ``terminated``, ``truncated``, ``gae`` and ``values``
``critic_loss``  ``observations`` and ``targets`` (the target, held
                 constant), with the critic at the differentiated
                 parameters: ``critic_params`` and ``critic_state`` (and
                 ``actions`` for a Q-critic), or DQN's and PQN's
                 ``q_state`` and ``actions``
``actor_loss``   ``actor_params`` (differentiated) and ``actor_state``,
                 plus ``observations``, ``raw_observations`` and
                 ``pi_mean`` on SAC, ASAC, REDQ, TD3 and APO (SAC adds
                 the expert's action ``a_expert``), but on APG (its
                 ``rollout``, ``returns`` and ``actor_params``)
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
    "post_update",
    "eval_metrics",
)


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
    ignored. Agents call it at construction (``ActorCritic.__init__``).
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


def check_extension_experts(
    agent_name: str,
    stack: Sequence["Extension"],
    agent_expert: Any = None,
    stores_expert_actions: bool = False,
) -> None:
    """Reject an extension calling a stateful expert (one with
    ``init_state``, called ``expert(state, obs)``) without its state.

    An extension calls the experts it declares (:meth:`Extension.experts`)
    on the observations it is given, where no state of the expert exists:
    a stateful one is allowed only as the agent's ``expert_policy`` of an
    agent that stores that expert's actions from its state
    (``stores_expert_actions``), for an extension that reads them
    (``reads_agent_expert_actions``). Stateless experts are always allowed.
    """
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
                + " To keep only the extension's use of the expert, give SAC the"
                " same expert with expert_buffer_n_steps=0 (no expert prefill)"
                " and expert_fraction=0.0 (no warm-up mixing); or give the"
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
    #: The fields holding an expert (raw observations in, env-unit actions
    #: out), which :meth:`ExtensionStack.with_experts` converts.
    expert_fields: ClassVar[frozenset[str]] = frozenset()

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

        The default is the identity: extensions that don't need agent-
        factory context (env_args, network_args, buffer, …) ignore every
        kwarg and stay as-is. Extensions that DO close over factory-
        resolved values (:class:`MCPretrain`, :class:`PhiRefresh`,
        :class:`ResidualPolicy`) override
        this to return a new (frozen) instance with the relevant fields
        filled in.

        Callers may invoke ``ext.bind_to_agent(env_args=..., buffer=...,
        ...)`` uniformly across an entire stack without per-extension
        type checks — see :meth:`ExtensionStack.bind_to_agent`.
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

    # ------------------------------------------------------------------
    # ``fold_<phase>``: the phase folded on an agent state
    # ------------------------------------------------------------------
    # Each fold builds the phase's :class:`ExtensionContext` and reads
    # (and, for state-threading phases, writes) ``agent_state.ext_state``,
    # in list order. On an empty stack it returns its input unchanged
    # without building a context, so an agent without extensions traces
    # nothing extra and an agent never needs to guard a fold itself.

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

    # ------------------------------------------------------------------
    # ``bind_to_agent`` — push factory context onto every extension.
    # ------------------------------------------------------------------
    def bind_to_agent(self, **agent_context: Any) -> "ExtensionStack":
        """Return a new stack with every extension bound to ``agent_context``.

        Each extension's :meth:`Extension.bind_to_agent` decides which
        kwargs (if any) it cares about; the base implementation is the
        identity. This is what lets the SAC factory (and any other
        agent factory) populate closed-over context on extensions like
        :class:`MCPretrain` / :class:`PhiRefresh` /
        :class:`ResidualPolicy` without
        per-extension type checks.

        Empty stack: returns ``self`` unchanged.
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
