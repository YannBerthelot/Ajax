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
``on_obs``       transform an observation before the network sees it
``on_batch``     transform a sampled batch before the update
``on_target``    transform the TD / value target
``critic_loss``  extra additive critic-loss term (summed over extensions)
``actor_loss``   extra additive actor-loss term (summed over extensions)
``action``       override the collection-time action (``None`` = defer)
``eval_action``  override the evaluation-time action (``None`` = defer)
``post_update``  hook after the update step (φ-refresh, schedules, …)
``eval_metrics`` extra metrics, merged into the eval log
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
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
    "on_obs",
    "on_batch",
    "on_target",
    "critic_loss",
    "actor_loss",
    "action",
    "eval_action",
    "post_update",
    "eval_metrics",
)


class Extension:
    """Base class for a composable research feature.

    Subclass and override only the phases the feature touches. Keep
    subclasses hashable (``@dataclass(frozen=True)`` is the recommended
    form) and stateless on ``self`` — see the module docstring.
    """

    #: Human-readable label, used in logs and the applicability check.
    name: str = "extension"

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
    def on_obs(
        self, obs: jax.Array, ext_state: Any, ctx: ExtensionContext
    ) -> jax.Array:
        """Transform an observation before the network consumes it."""
        del ext_state, ctx
        return obs

    def on_batch(self, batch: Any, ext_state: Any, ctx: ExtensionContext) -> Any:
        """Transform a sampled batch before the update step."""
        del ext_state, ctx
        return batch

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

    # -- action overrides (first non-None wins) --------------------------
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
    def implemented_phases(self) -> frozenset[str]:
        """The phases this extension actually overrides (vs the no-op base)."""
        return frozenset(
            phase
            for phase in PHASES
            if getattr(type(self), phase) is not getattr(Extension, phase)
        )


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

    def _keys(self, rng: jax.Array) -> tuple[jax.Array, ...]:
        """One sub-key per extension (≥1 split so an empty stack is safe)."""
        return tuple(jax.random.split(rng, max(1, len(self.extensions))))

    # -- init / pretrain -------------------------------------------------
    def init_states(self, agent_state: Any, rng: jax.Array) -> tuple:
        """Build the per-extension state tuple for ``BaseAgentState.ext_state``."""
        keys = self._keys(rng)
        return tuple(
            ext.init_state(agent_state, keys[i])
            for i, ext in enumerate(self.extensions)
        )

    def pretrain(
        self, agent_state: Any, ext_states: tuple, ctx: ExtensionContext
    ) -> tuple[Any, tuple]:
        new = list(ext_states)
        for i, ext in enumerate(self.extensions):
            agent_state, new[i] = ext.pretrain(agent_state, new[i], ctx)
        return agent_state, tuple(new)

    # -- transforms ------------------------------------------------------
    def on_obs(
        self, obs: jax.Array, ext_states: tuple, ctx: ExtensionContext
    ) -> jax.Array:
        for i, ext in enumerate(self.extensions):
            obs = ext.on_obs(obs, ext_states[i], ctx)
        return obs

    def on_batch(self, batch: Any, ext_states: tuple, ctx: ExtensionContext) -> Any:
        for i, ext in enumerate(self.extensions):
            batch = ext.on_batch(batch, ext_states[i], ctx)
        return batch

    def on_target(
        self,
        agent_state: Any,
        ext_states: tuple,
        batch: Any,
        target: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array:
        for i, ext in enumerate(self.extensions):
            target = ext.on_target(agent_state, ext_states[i], batch, target, ctx)
        return target

    # -- additive loss terms --------------------------------------------
    def critic_loss(
        self, agent_state: Any, ext_states: tuple, batch: Any, ctx: ExtensionContext
    ) -> jax.Array | float:
        total: jax.Array | float = 0.0
        for i, ext in enumerate(self.extensions):
            total = total + ext.critic_loss(agent_state, ext_states[i], batch, ctx)
        return total

    def actor_loss(
        self, agent_state: Any, ext_states: tuple, batch: Any, ctx: ExtensionContext
    ) -> jax.Array | float:
        total: jax.Array | float = 0.0
        for i, ext in enumerate(self.extensions):
            total = total + ext.actor_loss(agent_state, ext_states[i], batch, ctx)
        return total

    # -- action overrides (last non-None wins; None -> agent default) ----
    def action(
        self,
        agent_state: Any,
        ext_states: tuple,
        obs: jax.Array,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        keys = self._keys(rng)
        chosen: jax.Array | None = None
        for i, ext in enumerate(self.extensions):
            proposed = ext.action(agent_state, ext_states[i], obs, keys[i], ctx)
            if proposed is not None:
                chosen = proposed
        return chosen

    def eval_action(
        self,
        agent_state: Any,
        ext_states: tuple,
        obs: jax.Array,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> jax.Array | None:
        keys = self._keys(rng)
        chosen: jax.Array | None = None
        for i, ext in enumerate(self.extensions):
            proposed = ext.eval_action(agent_state, ext_states[i], obs, keys[i], ctx)
            if proposed is not None:
                chosen = proposed
        return chosen

    # -- state-threading + logging --------------------------------------
    def post_update(
        self, agent_state: Any, ext_states: tuple, ctx: ExtensionContext
    ) -> tuple[Any, tuple]:
        new = list(ext_states)
        for i, ext in enumerate(self.extensions):
            agent_state, new[i] = ext.post_update(agent_state, new[i], ctx)
        return agent_state, tuple(new)

    def eval_metrics(
        self, agent_state: Any, ext_states: tuple, rng: jax.Array, ctx: ExtensionContext
    ) -> dict:
        out: dict = {}
        keys = self._keys(rng)
        for i, ext in enumerate(self.extensions):
            out.update(ext.eval_metrics(agent_state, ext_states[i], keys[i], ctx))
        return out

    # -- introspection ---------------------------------------------------
    def implemented_phases(self) -> frozenset[str]:
        """Union of the phases used by any extension in the stack."""
        used: frozenset[str] = frozenset()
        for ext in self.extensions:
            used = used | ext.implemented_phases()
        return used
