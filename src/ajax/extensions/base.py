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

``on_target``, ``critic_loss`` and ``actor_loss`` may return ``(value,
metrics)`` instead of their value: ``metrics``, a dict of scalars named
``<extension>/<metric>``, is logged with the update's own metrics (the last
update's, or the mean over the update's epochs and minibatches, as the
agent logs its own). Two extensions logging one name raise.

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
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp
from flax import struct

from ajax.extensions.capabilities import PHASE_NEEDS, check_capabilities


@struct.dataclass
class ExtensionContext:
    """Shared context handed to every extension phase.

    ``step`` and ``rng`` are traced; ``total_steps`` is static so it can
    drive Python-level decisions (e.g. schedule lengths) at trace time.
    The ``pretrain`` phase of a run on ``ajax.agents.loop.TrainLoop`` also
    gets its static ``env_args`` and, off-policy, its ``update(agent_state,
    experience) -> (agent_state, aux)`` and replay ``buffer``, so an
    extension can run the agent's own update; each is None where the run
    has none.
    """

    step: jax.Array  # current global timestep (traced)
    rng: jax.Array  # a PRNG key for this phase call (traced)
    total_steps: int = struct.field(pytree_node=False, default=0)
    env_args: Any = struct.field(pytree_node=False, default=None)
    update: Optional[Callable] = struct.field(pytree_node=False, default=None)
    buffer: Any = struct.field(pytree_node=False, default=None)


# Phase methods an Extension may override. Used for the construction-time
# applicability check (an extension implementing a phase an agent never
# folds is almost certainly a user error).
PHASES: tuple[str, ...] = (
    "pretrain",
    "reward",
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
    "reward": {"PPO": "observations raw_observations actions"},
    "on_target": {
        "*": "observations next_observations rewards gamma",
        "SAC TD3 REDQ ASAC AVG DQN PQN IQL CrossQ TQC": "actions dones reward_scale",
        "SAC TD3 REDQ ASAC AVG IQL CrossQ TQC": "raw_next_observations next_a_expert",
        "PQN PPO APO": "terminated truncated",
        "PPO APO": "gae values",
        "SAC": "rng_key q_preds recurrent",
        "TD3 REDQ ASAC AVG IQL CrossQ TQC": "next_action",
        "TD3": "next_actions",
    },
    "critic_loss": {
        "*": "observations targets",
        "SAC TD3 REDQ ASAC AVG PPO APO IQL CrossQ TQC": "critic_params critic_state",
        "SAC TD3 REDQ ASAC AVG DQN PQN IQL CrossQ TQC": "actions",
        "SAC TD3 REDQ ASAC AVG IQL CrossQ TQC": "next_observations actor_state",
        "SAC TD3 REDQ IQL CrossQ TQC": "return_to_go",
        "DQN PQN": "q_state",
    },
    "actor_loss": {
        "*": "actor_params",
        "SAC TD3 REDQ ASAC AVG PPO APO UDRL APG DAgger IQL CrossQ TQC": "actor_state observations",
        "SAC TD3 REDQ ASAC AVG PPO APO APG UDRL DAgger IQL CrossQ TQC": "raw_observations pi_mean",
        "SAC TD3 REDQ ASAC AVG APO DAgger CrossQ TQC": "a_expert",
        "SAC TD3 REDQ ASAC AVG IQL CrossQ TQC": "critic_state",
        "SAC TD3 REDQ ASAC AVG IQL DAgger CrossQ TQC": "agent_loss",
        "SAC TD3 REDQ ASAC UDRL IQL CrossQ TQC": "actions",
        "APG": "rollout returns",
    },
}


def check_extensions(
    agent_name: str,
    stack: Iterable["Extension"],
    capabilities: Iterable[str],
    phases: frozenset,
    *,
    context: Iterable[str] = (),
    pretrain_context: Iterable[str] = (),
    buffer_fields: Iterable[str] = (),
    agent_expert: Any = None,
    takes_expert_policy: bool = False,
    stores_expert_actions: bool = False,
) -> None:
    """Reject the extensions of ``stack`` that cannot run on ``agent_name``,
    whose run has ``capabilities`` and whose code folds ``phases``, binds the
    ``context`` keys, fills the ``pretrain_context`` fields and stores the
    ``buffer_fields``
    (:func:`~ajax.extensions.capabilities.check_capabilities`), or that call
    a stateful expert without its state (:func:`check_extension_experts`).

    Raises a ``ValueError`` naming every offending extension: one whose phase
    an agent never folds would otherwise be silently ignored. Agents call it
    at construction (``ActorCritic.__init__``).
    """
    stack = tuple(stack)
    check_capabilities(
        agent_name,
        stack,
        capabilities,
        phases=phases,
        context=context,
        pretrain_context=pretrain_context,
        buffer_fields=buffer_fields,
        batch_table=PHASE_BATCH_KEYS,
        takes_expert_policy=takes_expert_policy,
    )
    check_extension_experts(agent_name, stack, agent_expert, stores_expert_actions)


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
    #: The capabilities it needs beyond those its phases imply (:meth:`needs`,
    #: :mod:`ajax.extensions.capabilities`).
    requires: ClassVar[frozenset[str]] = frozenset()
    #: The capabilities it gives the other extensions of its stack (MCPretrain:
    #: ``phi_star``).
    provides: ClassVar[frozenset[str]] = frozenset()
    #: The agent-context keys it cannot work without (:meth:`bind_to_agent`).
    context: ClassVar[frozenset[str]] = frozenset()
    #: The :class:`ExtensionContext` fields its ``pretrain`` reads (``env_args``,
    #: ``update``, ``buffer``), which the agent's loop fills.
    pretrain_context: ClassVar[frozenset[str]] = frozenset()
    #: Per phase, the batch keys it reads beyond those every agent folding
    #: the phase passes (``PHASE_BATCH_KEYS``).
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {}
    #: Whether the extension reads the agent's stored actions of its expert
    #: when that expert is the agent's (check_extension_experts).
    reads_agent_expert_actions: ClassVar[bool] = False
    #: The fields holding an expert (raw observations in, env-unit actions
    #: out), which :meth:`ExtensionStack.with_experts` converts.
    expert_fields: ClassVar[frozenset[str]] = frozenset()
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
    ) -> Any:
        """Transform the TD / value target; or return ``(target, metrics)``
        (the module docstring)."""
        del agent_state, ext_state, batch, ctx
        return target

    def critic_loss(
        self, agent_state: Any, ext_state: Any, batch: Any, ctx: ExtensionContext
    ) -> Any:
        """Extra additive critic-loss term (summed across extensions); or
        ``(term, metrics)``."""
        del agent_state, ext_state, batch, ctx
        return 0.0

    def actor_loss(
        self, agent_state: Any, ext_state: Any, batch: Any, ctx: ExtensionContext
    ) -> Any:
        """Extra additive actor-loss term (summed across extensions); or
        ``(term, metrics)``."""
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

    def reward(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: Any,
        rewards: jax.Array,
        ctx: ExtensionContext,
    ) -> tuple[jax.Array, Any, dict]:
        """Replace a rollout's ``rewards`` before the agent's targets read
        them. Returns ``(rewards, ext_state, metrics)``."""
        del agent_state, batch, ctx
        return rewards, ext_state, {}

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

    def needs(self) -> frozenset[str]:
        """The capabilities it needs: those its phases imply and those it
        ``requires``."""
        return self.requires.union(
            *(PHASE_NEEDS.get(phase, ()) for phase in self.implemented_phases())
        )

    def reads(self) -> Mapping[str, frozenset[str]]:
        """Per phase, the batch keys it reads (``batch_keys``)."""
        return self.batch_keys

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


def phase_output(out: Any) -> tuple[Any, dict]:
    """A training phase's value and metrics: it returns its value, or
    ``(value, metrics)`` (the module docstring)."""
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out
    return out, {}


def merge_metrics(*metrics: dict) -> dict:
    """The extensions' training metrics in one dict; a name two of them
    log raises (two extensions of one class report the same names)."""
    merged: dict = {}
    for named in metrics:
        clash = sorted(merged.keys() & named.keys())
        if clash:
            raise ValueError(
                f"Two extensions in the stack log the training metrics {clash};"
                " each name is logged once, so a stack holds one extension"
                " reporting it."
            )
        merged |= named
    return merged


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

    def fold_init(
        self, agent_state: Any, rng: jax.Array, total_steps: int, **context: Any
    ) -> Any:
        """A fresh run's extension setup: every extension's initial state,
        then its one-shot ``pretrain`` at step 0 (``rng`` split between the
        two; ``context``: :meth:`fold_pretrain`'s). Skipped on resume, where
        the state already carries both."""
        if not self.extensions:
            return agent_state
        init_key, pretrain_key = jax.random.split(rng)
        agent_state = self.fold_init_states(agent_state, init_key)
        return self.fold_pretrain(
            agent_state, jnp.asarray(0), pretrain_key, total_steps, **context
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
        **context: Any,
    ) -> Any:
        """One-shot pretrain fold; returns the updated ``agent_state``.
        ``context``: the run's static :class:`ExtensionContext` fields
        (``env_args``, ``update``, ``buffer``)."""
        if not self.extensions:
            return agent_state
        ctx = ExtensionContext(step=step, rng=rng, total_steps=total_steps, **context)
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

    def fold_reward(
        self,
        agent_state: Any,
        batch: Any,
        rewards: jax.Array,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> tuple[Any, jax.Array, dict]:
        """Reward fold, in list order, each seeing the previous rewards;
        returns the state (the extensions' states updated), the rewards and
        the metrics."""
        if not self.extensions:
            return agent_state, rewards, {}
        ext_states, metrics = list(agent_state.ext_state), dict[str, Any]()
        for i, (ext, key) in enumerate(
            zip(self.extensions, self._keys(rng), strict=True)
        ):
            ctx = ExtensionContext(step=step, rng=key, total_steps=total_steps)
            rewards, ext_states[i], ext_metrics = ext.reward(
                agent_state, ext_states[i], batch, rewards, ctx
            )
            metrics = merge_metrics(metrics, ext_metrics)
        return agent_state.replace(ext_state=tuple(ext_states)), rewards, metrics

    def fold_on_target(
        self,
        agent_state: Any,
        batch: Any,
        target: jax.Array,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> tuple[jax.Array, dict]:
        """Target-modifier fold; returns the (possibly reshaped) target and
        the extensions' metrics."""
        if not self.extensions:
            return target, {}
        ctx = ExtensionContext(step=step, rng=rng, total_steps=total_steps)
        metrics: dict = {}
        ext_states = agent_state.ext_state
        for i, ext in enumerate(self.extensions):
            out = ext.on_target(agent_state, ext_states[i], batch, target, ctx)
            target, ext_metrics = phase_output(out)
            metrics = merge_metrics(metrics, ext_metrics)
        return target, metrics

    def fold_critic_loss(
        self,
        agent_state: Any,
        batch: Any,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> tuple[jax.Array | float, dict]:
        """Critic-loss fold; returns the summed additive critic-loss term
        and the extensions' metrics."""
        return self._fold_loss(
            "critic_loss", agent_state, batch, step, rng, total_steps
        )

    def fold_actor_loss(
        self,
        agent_state: Any,
        batch: Any,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> tuple[jax.Array | float, dict]:
        """Actor-loss fold; returns the summed additive actor-loss term and
        the extensions' metrics."""
        return self._fold_loss("actor_loss", agent_state, batch, step, rng, total_steps)

    def _fold_loss(
        self,
        phase: str,
        agent_state: Any,
        batch: Any,
        step: Any,
        rng: jax.Array,
        total_steps: int,
    ) -> tuple[jax.Array | float, dict]:
        if not self.extensions:
            return 0.0, {}
        ctx = ExtensionContext(step=step, rng=rng, total_steps=total_steps)
        total: jax.Array | float = 0.0
        metrics: dict = {}
        ext_states = agent_state.ext_state
        for i, ext in enumerate(self.extensions):
            out = getattr(ext, phase)(agent_state, ext_states[i], batch, ctx)
            term, ext_metrics = phase_output(out)
            total = total + term
            metrics = merge_metrics(metrics, ext_metrics)
        return total, metrics

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
