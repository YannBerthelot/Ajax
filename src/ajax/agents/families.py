"""The agents' families. An agent declares its hyperparameters once, as the
fields of a dataclass, and writes what its paper defines: its networks, its
``init`` (on its own key layout), its ``update`` and its equations. Its
family builds everything else from those fields, the same for every member:
the constructor, the logged run config, the ``agent_config`` view, the
optimisers' settings, the replay, the extensions' binding, the collector's
storage, the training loop's shape and the evaluation's value estimate::

    @dataclass(kw_only=True, eq=False, repr=False)
    class IQL(ObservationNormalization, ExpertGuided, ReplayAgent):
        expectile: float = 0.7
        ...

        def init(self, key, ctx): ...
        def update(self, state, ctx): ...

The shared knobs an algorithm may take beyond the ones every agent takes
(gradient clipping, normalisation, an expert) are field mixins; a field
named like an :class:`ActorCritic` argument reaches it
(:meth:`DeclaredAgent.__post_init__`). Every agent, family and mixin carries
the literal ``@dataclass`` decorator, which the API docs read the
constructor from; the contract an agent declares for the extensions and the
loop (``name``, ``capabilities``, ...) stays unannotated, so it is no field.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import partial
from typing import Any, ClassVar, Optional, Union

import jax
from flax import struct
from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.loop import LOOP_PHASES, TrainLoop
from ajax.agents.recurrent import (
    make_replay_buffer,
    stored_actor_carry_dim,
    unsupported_recurrent_options,
)
from ajax.agents.steps import REPLAY_BATCH_KEYS
from ajax.buffers.utils import get_buffer
from ajax.controller import executed_fn
from ajax.environments.interaction import init_collector_state
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import Extension, ExtensionStack, holds_expert
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import (
    action_value,
    get_initialized_actor_critic,
    state_value,
)
from ajax.state import (
    BaseAgentConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import BufferType, EnvType

#: The arguments of ActorCritic's initialiser a same-named field reaches.
SHARED = frozenset(inspect.signature(ActorCritic.__init__).parameters) - {"self"}


@dataclass(frozen=True, eq=False)
class TrainContext:
    """What one train function's ``init`` and ``update`` read beyond the
    agent's fields: the run's configs, replay, bound extensions and horizon
    (CLAUDE.md rule 7: passed in, never global)."""

    env_args: EnvironmentConfig
    network_args: NetworkConfig
    actor_optimizer_args: OptimizerConfig
    critic_optimizer_args: OptimizerConfig
    stack: ExtensionStack
    total_timesteps: int
    #: ``collector(key, **options)``: the collector state, its replay storing
    #: what the run needs (``options``: init_collector_state's, an agent's).
    collector: Callable[..., Any]
    buffer: Optional[BufferType] = None
    expert_policy: Any = None
    recurrent: bool = False
    #: ``(actions, a_expert) -> actions`` the critics score under a residual
    #: policy (controller.executed_fn); None otherwise.
    executed: Optional[Callable] = None

    def actor_critic(
        self,
        key: jax.Array,
        build: Callable = get_initialized_actor_critic,
        critic_optimizer_args: Optional[OptimizerConfig] = None,
        **options: Any,
    ) -> tuple[Any, Any]:
        """``build``'s actor and critic train states on this run's configs
        (another critic optimiser's with ``critic_optimizer_args``)."""
        critic_args = critic_optimizer_args or self.critic_optimizer_args
        return build(
            key,
            self.env_args,
            self.actor_optimizer_args,
            critic_args,
            self.network_args,
            **options,
        )


# -- Shared knobs an algorithm may take beyond the ones every agent takes ---
@dataclass(kw_only=True, eq=False, repr=False)
class GradientClipping:
    #: Clip each network's gradients to this global norm (OptimizerConfig).
    max_grad_norm: Optional[float] = None


@dataclass(kw_only=True, eq=False, repr=False)
class ObservationNormalization:
    #: Normalise the observations the networks read by running statistics.
    normalize_observations: bool = False


@dataclass(kw_only=True, eq=False, repr=False)
class RewardNormalization:
    #: Divide the rewards learned from by a running standard deviation (the
    #: logged returns stay in the env's units): the single step's, or with
    #: ``reward_normalization_gamma`` the discounted return's (SB3's
    #: VecNormalize).
    normalize_rewards: bool = False
    reward_normalization_gamma: Optional[float] = None


@dataclass(kw_only=True, eq=False, repr=False)
class ExpertGuided:
    #: An expert the collector tracks: its actions, from its state along each
    #: trajectory, feed the observation features, a residual policy and the
    #: extensions reading them; each evaluation logs its return (it speaks
    #: the env's units; the agent reads it in its own).
    expert_policy: Optional[Callable] = None


@dataclass(eq=False, repr=False)
class DeclaredAgent(ActorCritic):
    """An agent whose constructor is its fields' (keyword-only but
    ``env_id``): the knobs every agent takes, its family's, its mixins' and
    its own."""

    env_id: Union[str, EnvType]
    _: KW_ONLY
    n_envs: int = 1
    env_params: Optional[EnvParams] = None
    #: Decoupled weight decay of the networks' optimisers (OptimizerConfig).
    weight_decay: float = 0.0
    #: Simulator steps per episode and per agent step (brax / playground).
    episode_length: Optional[int] = None
    action_repeat: int = 1
    extensions: Sequence[Extension] = ()

    #: The action spaces the algorithm acts in: "continuous", "discrete", "any".
    action_space: ClassVar[str] = "continuous"
    #: ``agent_config``'s class and the names it holds (:meth:`declare_config`).
    config_class: ClassVar[type] = BaseAgentConfig
    agent_config_fields: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        # The run config the loggers record: the call, before any check.
        self.config = {
            f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.init
        } | {"algo_name": self.name}
        self.check()
        shared = {name: getattr(self, name) for name in SHARED & self.field_names()}
        expert = getattr(self, "expert_policy", None)
        ActorCritic.__init__(self, **shared, **self.networks())
        # ActorCritic writes the expert in agent units over the field: the
        # wrapper moves to ``expert``, and the field keeps the caller's, which
        # dataclasses.replace passes to the next agent (on its own env).
        self._expert = getattr(self, "expert_policy", None)
        if expert is not None:
            self.expert_policy = expert
        if self.action_space != "any" and self.env_args.continuous != (
            self.action_space == "continuous"
        ):
            raise ValueError(
                f"{self.name} only supports {self.action_space} action spaces."
            )
        settings = self.optimizer_settings()
        self.actor_optimizer_args = self.actor_optimizer_args.replace(**settings)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(**settings)
        self.build()
        values = {name: getattr(self, name) for name in self.agent_config_fields}
        self.agent_config = self.config_class(**values)
        object.__setattr__(self, "_built", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # Fixed once built (owner, 2026-10-24), so what trains is what the
        # run config logs; other attributes (network_args, a monkeypatched
        # method) stay writable.
        fixed = name == "agent_config" or name in self.field_names()
        if fixed and getattr(self, "_built", False):
            raise AttributeError(
                f"{self.name}'s {name} is fixed once the agent is built (its run"
                f" config logs it): build another, dataclasses.replace(agent,"
                f" {name}=...)."
            )
        object.__setattr__(self, name, value)

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(f.name for f in dataclasses.fields(cls))  # type: ignore[arg-type]

    # -- what an agent may add (each a no-op here) ---------------------------
    def check(self) -> None:
        """Refuse arguments the algorithm cannot take, before anything is
        built."""

    def networks(self) -> dict[str, Any]:
        """The ActorCritic arguments the paper fixes (initialisations,
        heads) or derives from the fields (CrossQ's widths)."""
        return {}

    def optimizer_settings(self) -> dict[str, Any]:
        """OptimizerConfig settings both networks' optimisers take."""
        return {}

    def build(self) -> None:
        """What the agent builds from its fields once constructed (its
        family's replay)."""

    @property
    def expert(self) -> Any:
        """The agent's expert in agent units (ActorCritic wraps it), if any;
        the ``expert_policy`` field holds the one passed in."""
        return self._expert

    @classmethod
    def declare_config(cls, names: str) -> Any:
        """``agent_config``'s class, ``<name>Config``, which agents kept
        before their fields were declared: the ``names`` it holds, typed and
        defaulted as the fields (a property, as CrossQ's target entropy, is
        a required entry). A view until 0.5.0: the update reads the fields.
        Built at run time, it is Any to a type checker."""
        declared = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        spec: dict[str, Any] = {"__annotations__": {}, "__module__": cls.__module__}
        for name in names.split():
            f = declared.get(name)
            spec["__annotations__"][name] = Any if f is None else f.type
            if f is not None and f.default is not dataclasses.MISSING:
                spec[name] = f.default
        title = f"{cls.name}Config"
        view = type(title, (BaseAgentConfig,), spec | {"__qualname__": title})
        cls.config_class = struct.dataclass(view, kw_only=True)
        cls.agent_config_fields = tuple(names.split())
        return cls.config_class


@dataclass(kw_only=True, eq=False, repr=False)
class ReplayAgent(DeclaredAgent):
    """An off-policy agent on a replay of transitions
    (:meth:`TrainLoop.off_policy`): one step per env per iteration and, from
    its ``learning_starts`` (if it takes one), one :meth:`update`; offline
    (``train_offline``) one update per iteration on a replay of the dataset.
    The paper's replay sizes, which each agent states."""

    buffer_size: int
    batch_size: int

    replays_transitions = True
    supported_extension_phases = LOOP_PHASES | frozenset(
        "on_target critic_loss actor_loss transform_action actuate".split()
    )
    phase_batch_keys = REPLAY_BATCH_KEYS
    pretrain_context = frozenset("buffer env_args update".split())
    extension_context = frozenset(
        "actor_optimizer_args buffer env_args expert_policy".split()
    )

    def init(self, key: jax.Array, ctx: TrainContext) -> Any:
        """A fresh agent state, on the agent's own key layout."""
        raise NotImplementedError

    def update(self, state: Any, ctx: TrainContext) -> tuple[Any, Any]:
        """One update: ``(state, logged metrics)``."""
        raise NotImplementedError

    def optimizer_settings(self) -> dict[str, Any]:
        # The replay references' Adams keep their library's eps (optax,
        # torch): 1e-8, not Ajax's shared 1e-5.
        return {"eps": 1e-8}

    def build(self) -> None:
        self.buffer = self.replay_buffer()

    def replay_buffer(self) -> BufferType:
        return get_buffer(self.buffer_size, self.batch_size, self.n_envs)

    def value_estimate(self) -> Optional[ValueEstimate]:
        """The critic's estimate of the return each evaluation compares, on
        the agent's discount and reward scale; none without a discount (ASAC)
        or a discounted critic."""
        gamma, caps = getattr(self, "gamma", None), self.capabilities
        if gamma is None or "discounted" not in caps:
            return None
        if "q_critic" in caps:
            value = action_value
        elif "v_critic" in caps:
            value = state_value
        else:
            return None
        return ValueEstimate(value, gamma, getattr(self, "reward_scale", 1.0))

    def behaviour(self, ctx: TrainContext) -> Optional[Callable]:
        """The action pipeline collection runs; None: the policy's, uniform
        during the warm-up."""
        del ctx
        return None

    def keeps_carries(self) -> bool:
        """Whether the replay keeps the actor's carries (R2D2's stored state)."""
        return False

    def collector(
        self, key: jax.Array, loop: TrainLoop, network_args: NetworkConfig, **options
    ) -> Any:
        """The collector state, its replay storing what the run reads: the
        expert's data where it is held, the fields the extensions ask for,
        and ``options`` (init_collector_state's), an agent's own."""
        fields = loop.stack.buffer_fields()
        stored = self.stores_expert_actions and holds_expert(self.expert, loop.stack)
        return init_collector_state(
            key,
            env_args=loop.env_args,
            mode=loop.env_args.mode,
            buffer=self.buffer,
            actor_carry_dim=stored_actor_carry_dim(
                network_args.memory, self.keeps_carries()
            ),
            include_raw_obs=stored,
            include_raw_next_obs="raw_next_obs" in fields,
            include_return_to_go="return_to_go" in fields,
            include_next_action="next_action" in fields,
            include_is_expert=stored,
            expert_policy=self.expert,
            **options,
        )

    def make_train(
        self,
        env_args: EnvironmentConfig,
        actor_optimizer_args: OptimizerConfig,
        critic_optimizer_args: OptimizerConfig,
        network_args: NetworkConfig,
        agent_config: Any,
        total_timesteps: int,
        num_episode_test: int,
        run_ids: Optional[Sequence[str]] = None,
        logging_config: Any = None,
        start_timestep: int = 0,
        end_timestep: Optional[int] = None,
        offline_data: Any = None,
        extensions: Sequence[Extension] = (),
    ) -> Callable:
        """The train function :meth:`ActorCritic.train` builds: the loop
        over :meth:`init` and :meth:`update`."""
        if agent_config is not self.agent_config:
            raise ValueError(
                f"{self.name} trains on its fields, which agent_config only views:"
                " build another agent, dataclasses.replace(agent, ...), instead of"
                " passing another agent_config."
            )
        given = {
            "env_args": env_args,
            "actor_optimizer_args": actor_optimizer_args,
            "critic_optimizer_args": critic_optimizer_args,
            "network_args": network_args,
            "expert_policy": self.expert,
            "buffer": self.buffer,
        }
        context = {name: given[name] for name in sorted(self.extension_context)}
        bound = ExtensionStack(extensions).bind_to_agent(**context).extensions
        recurrent = network_args.memory is not None
        if recurrent:
            unsupported_recurrent_options(
                self.name, expert_policy=self.expert, extensions=tuple(bound) or None
            )
        loop = TrainLoop.create(
            env_args,
            total_timesteps,
            num_episode_test,
            run_ids,
            logging_config,
            bound,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            expert_policy=self.expert,
            offline_data=offline_data,
        )
        ctx = TrainContext(
            env_args=env_args,
            network_args=network_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            stack=loop.stack,
            total_timesteps=total_timesteps,
            collector=partial(self.collector, loop=loop, network_args=network_args),
            buffer=self.buffer,
            expert_policy=self.expert,
            recurrent=recurrent,
            executed=executed_fn(env_args),
        )
        return loop.off_policy(
            partial(self.init, ctx=ctx),
            lambda state, _transition: self.update(state, ctx),
            # The agents that warm up declare learning_starts (DAgger: none).
            getattr(self, "learning_starts", 0),
            recurrent=recurrent,
            collect_kwargs={
                "buffer": self.buffer,
                "action_pipeline": self.behaviour(ctx),
                "store_hidden": recurrent and self.keeps_carries(),
            },
            eval_kwargs={
                "expert_policy": self.expert,
                "value_estimate": self.value_estimate(),
            },
        )


@dataclass(kw_only=True, eq=False, repr=False)
class RecurrentReplayAgent(ReplayAgent):
    """A replay agent that may have memory: it then trains on replayed
    windows of ``burn_in + sequence_length + 1`` steps with R2D2's burn-in
    (:mod:`ajax.agents.recurrent`)."""

    memory: Optional[Union[MemoryConfig, dict]] = None
    burn_in: int = 8
    sequence_length: int = 16
    #: R2D2's stored-state replay: replayed windows start from the actor's
    #: carries recorded at collection (Kapturowski et al. 2019).
    stored_state: bool = False

    supports_memory = True

    def replay_buffer(self) -> BufferType:
        return make_replay_buffer(
            self,
            self.n_envs,
            self.network_args.memory,
            self.buffer_size,
            self.batch_size,
        )

    def keeps_carries(self) -> bool:
        return self.stored_state
