"""The agents' families. An agent declares its hyperparameters once, as the
fields of a dataclass, and writes what its paper defines: its networks, its
init (on its own key layout), its update and its equations. Its
family builds everything else from those fields, the same for every member:
the constructor, the logged run config, the agent_config view, the
optimisers, the replay, the extensions' binding, the collector's storage,
the training loop's shape and the evaluation's value estimate::

    @dataclass(kw_only=True, eq=False, repr=False)
    class IQL(ObservationNormalization, ExpertGuided, ReplayAgent):
        expectile: float = 0.7
        ...

        def init(self, key, context): ...
        def update(self, agent_state, context): ...

An agent learning from a replay of transitions is a ReplayAgent,
one learning from each rollout of its current policy an
OnPolicyAgent. The shared knobs an algorithm may take beyond the
ones every agent takes (gradient clipping, normalisation, an expert, image
observations) are field mixins. The family reads some fields by name
(DeclaredAgent, ReplayAgent, OnPolicyAgent).
Every agent, family and mixin carries the literal @dataclass decorator,
which the API docs read the constructor from; the contract an agent declares
for the extensions and the loop (name, capabilities, ...) stays
unannotated, so it is no field.
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
    CNNSpec,
    action_value,
    get_initialized_actor_critic,
    state_value,
)
from ajax.state import (
    BaseAgentConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.types import BufferType, EnvType

#: The arguments of ActorCritic's initialiser: a field so named reaches it.
ACTOR_CRITIC_ARGS = frozenset(
    inspect.signature(ActorCritic.__init__).parameters
).difference({"self"})
#: The fields the value estimate of an agent claiming the "discounted"
#: capability reads: its gamma, which it declares, and its
#: reward_scale, 1 (the rewards unscaled) where it declares none (PPO).
DISCOUNT_FIELDS = ("gamma", "reward_scale")


@dataclass(frozen=True, eq=False)
class TrainContext:
    """What one train() call gives the agent's init and update, beyond its
    hyperparameters.

    It holds:

    - the run's environment config (env_args);
    - the network and optimiser configs (network_args,
      actor_optimizer_args, critic_optimizer_args);
    - the extensions, bound to this run;
    - the replay buffer, for a replay agent (None on-policy);
    - how to build the collector (collector);
    - the run's horizon (total_timesteps);
    - whether the run is recurrent (its networks have memory);
    - the residual-action hook (executed), None without a residual policy.

    Why it exists: an agent's hyperparameters are its fields, fixed once it
    is built. Everything that depends on one train() call (the env and the
    configs it may override, the extensions bound to them, the horizon) is
    passed explicitly to init and update, where SB3 keeps it as attributes
    of the agent and PureJaxRL in closures inside its make_train. One agent
    then trains any number of runs, and a test calls init or update alone
    on agent.context(total_timesteps) (CLAUDE.md rule 7: passed in, never
    global).
    """

    env_args: EnvironmentConfig
    network_args: NetworkConfig
    actor_optimizer_args: OptimizerConfig
    critic_optimizer_args: OptimizerConfig
    extensions: ExtensionStack
    total_timesteps: int
    #: collector(key, **options): the collector state, its replay storing
    #: what the run needs (options: init_collector_state's, an agent's).
    collector: Callable[..., Any]
    buffer: Optional[BufferType] = None
    recurrent: bool = False
    #: (actions, a_expert) -> actions the critics score under a residual
    #: policy (controller.executed_fn); None otherwise.
    executed: Optional[Callable] = None

    def actor_critic(
        self,
        key: jax.Array,
        build: Callable = get_initialized_actor_critic,
        critic_optimizer_args: Optional[OptimizerConfig] = None,
        **options: Any,
    ) -> tuple[Any, Any]:
        """build's actor and critic train states on this run's configs
        (another critic optimiser's with critic_optimizer_args)."""
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
    """Gradient clipping of both networks' optimisers.

    Args:
        max_grad_norm: Clip each network's gradients to this global norm
            (OptimizerConfig); None leaves them unclipped.
    """

    max_grad_norm: Optional[float] = None


@dataclass(kw_only=True, eq=False, repr=False)
class ObservationNormalization:
    """Observations normalised inside the agent.

    Args:
        normalize_observations: Normalise the observations the networks read
            by running statistics.
    """

    normalize_observations: bool = False


@dataclass(kw_only=True, eq=False, repr=False)
class RewardNormalization:
    """Rewards normalised before the agent learns from them.

    Args:
        normalize_rewards: Divide the rewards learned from by a running
            standard deviation (the logged returns stay in the env's units):
            the single step's by default.
        reward_normalization_gamma: With normalize_rewards, the discount
            of the return whose running standard deviation divides instead
            (SB3's VecNormalize).
    """

    normalize_rewards: bool = False
    reward_normalization_gamma: Optional[float] = None


@dataclass(kw_only=True, eq=False, repr=False)
class ExpertGuided:
    """An expert the agent tracks.

    Args:
        expert_policy: An expert the collector tracks: its actions, from its
            state along each trajectory, feed the observation features, a
            residual policy and the extensions reading them; each evaluation
            logs its return. It speaks the env's units; the agent reads it in
            its own (agent.expert).
    """

    expert_policy: Optional[Callable] = None


@dataclass(kw_only=True, eq=False, repr=False)
class ImageObservations:
    """Observations holding an image, which the networks read through a CNN
    encoder (NetworkConfig).

    Args:
        cnn_image_shape: The image's (height, width, channels), flattened
            first in each observation; None reads the observations with the
            MLP encoder.
        cnn_extra_obs_dim: The features after the image, read beside the
            encoder's output.
        cnn_spec: The encoder's layers (CNNSpec); None: CNNEncoder's default.
    """

    cnn_image_shape: Optional[tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
    cnn_spec: Optional[CNNSpec] = None


@dataclass(eq=False, repr=False)
class DeclaredAgent(ActorCritic):
    """An agent whose constructor is its fields' (keyword-only but
    env_id): the knobs every agent takes, its family's, its mixins' and
    its own.

    The family reads some fields by name. Every field named like an
    ActorCritic argument (ACTOR_CRITIC_ARGS:
    actor_learning_rate, critic_architecture, ...) reaches it;
    expert_policy is the agent's expert; an agent claiming the
    discounted capability declares gamma, or construction raises
    TypeError, and reward_scale if it scales its rewards
    (DISCOUNT_FIELDS). Its family may read more
    (ReplayAgent, OnPolicyAgent).

    Args:
        env_id: The environment's id, or an environment.
        n_envs: The environments stepped in parallel.
        env_params: The environment's parameters; its defaults if None.
        weight_decay: Decoupled weight decay of the networks' optimisers
            (OptimizerConfig).
        episode_length: Simulator steps per episode (brax / playground).
        action_repeat: Simulator steps per agent step (brax / playground).
        extensions: The extensions applied in the training loop.
    """

    env_id: Union[str, EnvType]
    _: KW_ONLY
    n_envs: int = 1
    env_params: Optional[EnvParams] = None
    weight_decay: float = 0.0
    episode_length: Optional[int] = None
    action_repeat: int = 1
    extensions: Sequence[Extension] = ()

    #: The action spaces the algorithm acts in: "continuous", "discrete", "any".
    action_space: ClassVar[str] = "continuous"
    #: agent_config's class and the names it holds (declare_config).
    config_class: ClassVar[type] = BaseAgentConfig
    agent_config_fields: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        # The run config the loggers record: the call, before any check.
        self.config = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.init
        }
        self.config.update({"algo_name": self.name})
        names = self.field_names()
        if "discounted" in self.capabilities and "gamma" not in names:
            raise TypeError(
                f"{self.name} claims the 'discounted' capability, so it declares"
                " gamma, the discount its value estimate reads."
            )
        self.check()
        shared = {
            name: getattr(self, name) for name in ACTOR_CRITIC_ARGS.intersection(names)
        }
        derived = self.actor_critic_args()
        # A derived argument may resolve a field the caller left None (PPO's
        # reference initialisation), never replace a value the caller gave.
        given = sorted(
            name
            for name in set(derived).intersection(shared)
            if shared[name] is not None
        )
        if given:
            raise TypeError(
                f"{self.name}.actor_critic_args() may only fill a field left None;"
                f" it would replace {', '.join(given)}."
            )
        expert = getattr(self, "expert_policy", None)
        arguments = dict(shared)
        arguments.update(derived)
        ActorCritic.__init__(self, **arguments)
        # ActorCritic writes the expert in agent units over the field: the
        # wrapper moves to expert, and the field keeps the caller's, which
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
        self._build_family()
        self.setup()
        values = {name: getattr(self, name) for name in self.agent_config_fields}
        self.agent_config = self.config_class(**values)
        object.__setattr__(self, "_built", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # Fixed once built (owner, 2026-10-24), so what trains is what the
        # run config logs; other attributes (network_args, a monkeypatched
        # method) stay writable.
        built = getattr(self, "_built", False)
        if built and name == "agent_config":
            raise AttributeError(
                f"{self.name}'s agent_config views its fields, fixed once the"
                " agent is built: build another with a field replaced, as"
                " dataclasses.replace(agent, gamma=...)."
            )
        if built and name in self.field_names():
            raise AttributeError(
                f"{self.name}'s {name} is fixed once the agent is built (its run"
                f" config logs it): build another, dataclasses.replace(agent,"
                f" {name}=...)."
            )
        object.__setattr__(self, name, value)

    @classmethod
    def field_names(cls) -> frozenset[str]:
        return frozenset(field.name for field in dataclasses.fields(cls))  # type: ignore[arg-type]

    def _build_family(self) -> None:
        """What the family builds from the fields, before the agent's
        setup (ReplayAgent: the replay)."""

    # -- What an agent may add (each a no-op here; none needs super()) -------
    def check(self) -> None:
        """Refuse arguments the algorithm cannot take, before anything is
        built."""

    def actor_critic_args(self) -> dict[str, Any]:
        """ActorCritic arguments the paper fixes (initialisations, heads) or
        derives from the fields (CrossQ's learning rates and widths). One
        named like a field only resolves it where the caller left it None
        (TypeError otherwise)."""
        return {}

    def optimizer_settings(self) -> dict[str, Any]:
        """OptimizerConfig settings both networks' optimisers take (the
        reference's Adam: its eps, its betas)."""
        return {}

    def setup(self) -> None:
        """What the agent derives or checks once its env, networks and family
        are built: IQL's value optimiser, PPO's checks of its rollout split
        (they need n_envs)."""

    @property
    def expert(self) -> Any:
        """The agent's expert in agent units (ActorCritic wraps it), if any;
        the expert_policy field holds the one passed in."""
        return self._expert

    def value_estimate(self) -> Optional[ValueEstimate]:
        """The critic's estimate of the return each evaluation compares, on
        the agent's discount and reward scale; none without a discounted
        critic."""
        capabilities = self.capabilities
        if "discounted" not in capabilities:
            return None
        if "q_critic" in capabilities:
            value = action_value
        elif "v_critic" in capabilities:
            value = state_value
        else:
            return None
        # gamma is declared (__post_init__); reward_scale is 1 where it is not.
        gamma, reward_scale = (getattr(self, name, 1.0) for name in DISCOUNT_FIELDS)
        return ValueEstimate(value, gamma, reward_scale)

    def check_agent_config(self, agent_config: Any) -> None:
        """Refuse an agent_config other than the agent's own view."""
        if agent_config is not self.agent_config:
            raise ValueError(
                f"{self.name} trains on its fields, which agent_config only views:"
                " build another agent, dataclasses.replace(agent, ...), instead of"
                " passing another agent_config."
            )

    def context(
        self,
        total_timesteps: int,
        num_episode_test: int = 1,
        *,
        env_args: Optional[EnvironmentConfig] = None,
        actor_optimizer_args: Optional[OptimizerConfig] = None,
        critic_optimizer_args: Optional[OptimizerConfig] = None,
        network_args: Optional[NetworkConfig] = None,
        extensions: Optional[Sequence[Extension]] = None,
        run_ids: Optional[Sequence[str]] = None,
        logging_config: Any = None,
        start_timestep: int = 0,
        end_timestep: Optional[int] = None,
        offline_data: Any = None,
    ) -> tuple[TrainLoop, TrainContext]:
        """The loop and the context of one train function, on the agent's
        own configs and extensions unless others are given: the extensions
        bound to the keys of its extension_context. Its family's
        make_train builds the run on them; a test calls
        agent.context(n) to call init or update alone."""
        configs: dict[str, Any] = {
            "env_args": env_args,
            "actor_optimizer_args": actor_optimizer_args,
            "critic_optimizer_args": critic_optimizer_args,
            "network_args": network_args,
        }
        configs = {
            name: getattr(self, name) if config is None else config
            for name, config in configs.items()
        }
        env_args, network_args = configs["env_args"], configs["network_args"]
        if extensions is None:
            extensions = tuple(self.extension_stack.extensions)
        buffer = getattr(self, "buffer", None)  # a replay agent's
        given = dict(configs)
        given.update({"expert_policy": self.expert, "buffer": buffer})
        unknown = sorted(self.extension_context.difference(given.keys()))
        if unknown:
            raise ValueError(
                f"{self.name}'s extension_context names {', '.join(unknown)},"
                " which DeclaredAgent.context does not give its extensions (it"
                f" gives {', '.join(sorted(given))})."
            )
        agent_context = {name: given[name] for name in sorted(self.extension_context)}
        bound = ExtensionStack(extensions).bind_to_agent(**agent_context).extensions
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
        context = TrainContext(
            **configs,
            extensions=loop.extensions,
            total_timesteps=total_timesteps,
            collector=partial(self.collector, loop=loop, network_args=network_args),
            buffer=buffer,
            recurrent=recurrent,
            executed=executed_fn(env_args),
        )
        return loop, context

    def collector(
        self, key: jax.Array, loop: TrainLoop, network_args: NetworkConfig, **options
    ) -> Any:
        """The collector state (init_collector_state), holding a
        stateful expert's state, with options an agent's own."""
        raise NotImplementedError

    @classmethod
    def declare_config(cls, names: str) -> Any:
        """agent_config's class, the agent's name then Config: the names it holds,
        typed and defaulted as the fields (a property, as CrossQ's target
        entropy, is a required entry). Only the agents that had a Config
        before their fields were declared (IQL, CrossQ, TD3, PPO, APO, PQN) keep
        this view, until 0.5.0; the update reads the fields. Built at run
        time, it is Any to a type checker."""
        declared = {field.name: field for field in dataclasses.fields(cls)}  # type: ignore[arg-type]
        spec: dict[str, Any] = {"__annotations__": {}, "__module__": cls.__module__}
        for name in names.split():
            field = declared.get(name)
            spec["__annotations__"][name] = Any if field is None else field.type
            if field is not None and field.default is not dataclasses.MISSING:
                spec[name] = field.default
        title = f"{cls.name}Config"
        spec.update({"__qualname__": title})
        view = type(title, (BaseAgentConfig,), spec)
        cls.config_class = struct.dataclass(view, kw_only=True)
        cls.agent_config_fields = tuple(names.split())
        return cls.config_class


@dataclass(kw_only=True, eq=False, repr=False)
class ReplayAgent(DeclaredAgent):
    """An off-policy agent on a replay of transitions
    (TrainLoop.off_policy): one step per env per iteration and, from
    learning_starts, one update; offline (train_offline) one
    update per iteration on a replay of the dataset.

    Beyond DeclaredAgent's names, the family reads
    buffer_size, batch_size and learning_starts, which every
    member declares with its paper's values. An agent writes init
    and update, and may replace collection's action pipeline
    (behaviour).

    Args:
        buffer_size: The replay's capacity, in transitions.
        batch_size: The rows each update samples.
        learning_starts: The env steps collected before the first update
            (0: none).
    """

    buffer_size: int
    batch_size: int
    learning_starts: int

    replays_transitions = True
    supported_extension_phases = LOOP_PHASES.union(
        {"on_target", "critic_loss", "actor_loss", "transform_action", "actuate"}
    )
    phase_batch_keys = REPLAY_BATCH_KEYS
    pretrain_context = frozenset({"buffer", "env_args", "update"})
    extension_context = frozenset(
        {"actor_optimizer_args", "buffer", "env_args", "expert_policy"}
    )

    def init(self, key: jax.Array, context: TrainContext) -> Any:
        """A fresh agent state, on the agent's own key layout."""
        raise NotImplementedError

    def update(self, agent_state: Any, context: TrainContext) -> tuple[Any, Any]:
        """One update: (agent_state, logged metrics)."""
        raise NotImplementedError

    def behaviour(self, context: TrainContext) -> Optional[Callable]:
        """The action pipeline collection runs; None: the policy's, uniform
        during the warm-up."""
        del context
        return None

    def _build_family(self) -> None:
        self.buffer = self.replay_buffer()

    def replay_buffer(self) -> BufferType:
        return get_buffer(self.buffer_size, self.batch_size, self.n_envs)

    def stores_state(self) -> bool:
        """Whether the replay keeps the actor's recurrent state (R2D2's
        stored state)."""
        return False

    def collector(
        self, key: jax.Array, loop: TrainLoop, network_args: NetworkConfig, **options
    ) -> Any:
        """The collector state, its replay storing what the run reads: the
        expert's data where it is held, the fields the extensions ask for,
        and options (init_collector_state's), an agent's own."""
        fields = loop.extensions.buffer_fields()
        stored = self.stores_expert_actions and holds_expert(
            self.expert, loop.extensions
        )
        return init_collector_state(
            key,
            env_args=loop.env_args,
            mode=loop.env_args.mode,
            buffer=self.buffer,
            actor_carry_dim=stored_actor_carry_dim(
                network_args.memory, self.stores_state()
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
        """The train function ActorCritic.train builds: the loop
        over init and update."""
        self.check_agent_config(agent_config)
        loop, context = self.context(
            total_timesteps,
            num_episode_test,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            extensions=extensions,
            run_ids=run_ids,
            logging_config=logging_config,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            offline_data=offline_data,
        )
        recurrent = context.recurrent
        return loop.off_policy(
            partial(self.init, context=context),
            lambda agent_state, _transition: self.update(agent_state, context),
            self.learning_starts,
            recurrent=recurrent,
            collect_kwargs={
                "buffer": self.buffer,
                "action_pipeline": self.behaviour(context),
                "store_hidden": recurrent and self.stores_state(),
            },
            eval_kwargs={
                "expert_policy": self.expert,
                "value_estimate": self.value_estimate(),
            },
        )


@dataclass(kw_only=True, eq=False, repr=False)
class RecurrentReplayAgent(ReplayAgent):
    """A replay agent that may have memory: it then trains on replayed
    windows of burn_in + sequence_length + 1 steps with R2D2's burn-in
    (ajax.agents.recurrent), and its learning_starts must give
    each env more steps than a window.

    Args:
        memory: The networks' memory block (MemoryConfig or its dict form);
            None keeps them feedforward.
        burn_in: The steps that rebuild the recurrent state before a replayed
            window's trained steps.
        sequence_length: The trained steps of a replayed window.
        stored_state: R2D2's stored-state replay: replayed windows start from
            the actor's recurrent state recorded at collection (Kapturowski
            et al. 2019).
    """

    memory: Optional[Union[MemoryConfig, dict]] = None
    burn_in: int = 8
    sequence_length: int = 16
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

    def stores_state(self) -> bool:
        return self.stored_state


@dataclass(kw_only=True, eq=False, repr=False)
class OnPolicyAgent(DeclaredAgent):
    """An on-policy agent (TrainLoop.on_policy): each iteration
    collects n_steps per env with the current policy, then one
    update on that rollout, n_epochs passes over its
    minibatches (ajax.agents.minibatch). The rollout keeps each
    observation normalised with the statistics of its step
    (EnvironmentConfig.store_normalized_obs), as SB3's VecNormalize does.

    The family reads n_steps, n_epochs and num_minibatches,
    which every member declares with its paper's values, and brax's
    num_resets_per_eval and num_evals where an agent declares them.
    An agent writes init and update, and may replace
    collection's action pipeline (behaviour) and add to each
    evaluation (evaluation_options).

    Args:
        n_steps: The steps each env collects per iteration.
        n_epochs: The passes of each update over the rollout.
        num_minibatches: The minibatches of each pass (PPO derives them
            from batch_size at 0).
        expose_recent_rollout: Keep each rollout on
            agent_state.last_rollout for measurement extensions.
    """

    n_steps: int
    n_epochs: int
    num_minibatches: int
    expose_recent_rollout: bool = False

    def init(self, key: jax.Array, context: TrainContext) -> Any:
        """A fresh agent state, on the agent's own key layout."""
        raise NotImplementedError

    def update(
        self,
        agent_state: Any,
        rollout: Transition,
        agent_state_at_rollout_start: Any,
        context: TrainContext,
    ) -> tuple[Any, Any]:
        """One update on the (n_steps, n_envs) rollout collected from
        agent_state_at_rollout_start (a recurrent agent replays the rollout
        from that state's memories): (agent_state, logged metrics)."""
        raise NotImplementedError

    def behaviour(self, context: TrainContext) -> Optional[Callable]:
        """The action pipeline collection runs; None: the policy's sample."""
        del context
        return None

    def evaluation_options(self) -> dict[str, Any]:
        """Each evaluation's options beyond the expert and the critic's
        estimate (ajax.log.evaluate_and_log's: APO's
        avg_reward_mode)."""
        return {}

    def collector(
        self, key: jax.Array, loop: TrainLoop, network_args: NetworkConfig, **options
    ) -> Any:
        del network_args
        return init_collector_state(
            key,
            env_args=loop.env_args,
            mode=loop.env_args.mode,
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
        extensions: Sequence[Extension] = (),
    ) -> Callable:
        """The train function ActorCritic.train builds: the loop
        over init and update. It takes no dataset: an
        on-policy agent learns from its own rollouts (train_offline
        refuses it)."""
        self.check_agent_config(agent_config)
        loop, context = self.context(
            total_timesteps,
            num_episode_test,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            extensions=extensions,
            run_ids=run_ids,
            logging_config=logging_config,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
        )
        eval_kwargs = {
            "expert_policy": self.expert,
            "value_estimate": self.value_estimate(),
        }
        eval_kwargs.update(self.evaluation_options())
        return loop.on_policy(
            partial(self.init, context=context),
            partial(self.update, context=context),
            self.n_steps,
            recurrent=context.recurrent,
            expose_rollout=self.expose_recent_rollout,
            forced_resets=(
                getattr(self, "num_resets_per_eval", 0),
                getattr(self, "num_evals", 1),
            ),
            collect_kwargs={"action_pipeline": self.behaviour(context)},
            eval_kwargs=eval_kwargs,
        )
