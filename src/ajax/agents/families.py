"""The agents' families. An agent declares its hyperparameters once, as the
fields of a dataclass, and writes what its paper defines: its networks, its
init (on its own key layout), its update and its equations. Its
family builds everything else from those fields, the same for every member:
the constructor, the logged run config, the optimisers, the replay, the extensions' binding, the collector's storage,
the training loop's shape and the evaluation's value estimate::

    @dataclass(kw_only=True, eq=False, repr=False)
    class IQL(ObservationNormalization, ExpertGuided, ReplayAgent):
        expectile: float = 0.7
        ...

        def init(self, key, context): ...
        def update(self, agent_state, context): ...

An agent learning from a replay of transitions is a ReplayAgent,
one learning from each rollout of its current policy an
OnPolicyAgent, one whose iteration is its own (a differentiable rollout,
a replay of segments) an OwnIterationAgent. The shared knobs an algorithm may take beyond the
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
import jax.numpy as jnp
from gymnax import EnvParams

from ajax.agents import update_steps
from ajax.agents.base import ActorCritic
from ajax.agents.loop import LOOP_PHASES, Evaluation, TrainLoop

# Imported under another name: ReplayAgent's own hook is make_replay_buffer.
from ajax.agents.recurrent import make_replay_buffer as make_recurrent_replay_buffer
from ajax.agents.recurrent import stored_actor_carry_dim, unsupported_recurrent_options
from ajax.agents.update_steps import REPLAY_BATCH_KEYS, reached
from ajax.buffers.utils import get_buffer
from ajax.controller import (
    ScoredActions,
    executed_fn,
    make_scored_actions,
    policy_action_dim,
)
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
from ajax.schedule import Schedule, at
from ajax.state import (
    AlphaConfig,
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
#: What an agent gives BellmanPretrain: its discount, its memory flag and
#: its family's critic step without extensions (ReplayAgent.extension_values).
BELLMAN_CONTEXT = frozenset(
    {"gamma", "recurrent", "reward_scale", "update_target_fn", "update_value_fn"}
)
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
    - how to build the collector (make_collector);
    - the run's horizon (total_timesteps);
    - whether the run is recurrent (its networks have memory);
    - the residual-action hook (executed), None without a residual policy.

    Why it exists: an agent's hyperparameters are its fields, fixed once it
    is built. Everything that depends on one train() call (the env and the
    configs it may override, the extensions bound to them, the horizon) is
    passed explicitly to init and update, where SB3 keeps it as attributes
    of the agent and PureJaxRL in closures inside its make_train. One agent
    then trains any number of runs, and a test calls init or update alone
    on agent.make_context(total_timesteps) (CLAUDE.md rule 7: passed in, never
    global).
    """

    env_args: EnvironmentConfig
    network_args: NetworkConfig
    actor_optimizer_args: OptimizerConfig
    critic_optimizer_args: OptimizerConfig
    extensions: ExtensionStack
    total_timesteps: int
    #: make_collector(key, **options): the collector state, its replay storing
    #: what the run needs (options: init_collector_state's, an agent's).
    make_collector: Callable[..., Any]
    buffer: Optional[BufferType] = None
    recurrent: bool = False
    #: (actions, a_expert) -> actions the critics score under a residual
    #: policy (controller.executed_fn); None otherwise.
    executed: Optional[Callable] = None

    def make_scored_actions(self, batch: Transition) -> ScoredActions:
        """The maps to the actions the critics score at batch's rows, at s
        and at s' (controller.make_scored_actions)."""
        return make_scored_actions(self.executed, batch)

    def make_actor_critic(
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
        eval_expert_policy: The expert whose return each evaluation logs
            instead, which takes no other part in the run.
    """

    expert_policy: Optional[Callable] = None
    eval_expert_policy: Optional[Callable] = None


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


# A mixin's methods read the agent's attributes (env_args, its optimisers),
# which only the agent has: their self is typed Any.
@dataclass(kw_only=True, eq=False, repr=False)
class EntropyTemperature:
    """The entropy temperature alpha of a soft actor-critic, learned towards
    a target entropy (Haarnoja et al., 2018b) unless the agent fixes it
    (ajax.agents.temperature).

    Args:
        alpha_learning_rate: The temperature's Adam learning rate; None: the
            actor's (CrossQ's authors step every network at one rate).
        alpha_init: The temperature's initial value.
        target_entropy_per_dim: The target entropy per dimension of the
            policy's actions (gain mode's gains): a constant, or a function
            of the timestep (ajax.schedule).
        alpha_update_start: The env steps collected before the
            temperature's steps are kept (0: from the first update).
    """

    alpha_learning_rate: Optional[float] = 3e-4
    alpha_init: float = 1.0
    target_entropy_per_dim: Schedule = -1.0
    alpha_update_start: int = 0

    @property
    def alpha_args(self: Any) -> AlphaConfig:
        """The temperature's learning rate and initial value."""
        rate = self.alpha_learning_rate
        if rate is None:
            rate = self.actor_optimizer_args.learning_rate
        return AlphaConfig(learning_rate=rate, alpha_init=self.alpha_init)

    @property
    def target_entropy(self: Any) -> Schedule:
        """target_entropy_per_dim times the policy's action width (gain
        mode's gains), at each timestep for a schedule."""
        per_dim, width = self.target_entropy_per_dim, policy_action_dim(self.env_args)
        if callable(per_dim):
            return lambda timestep: per_dim(timestep) * width
        return per_dim * width

    def target_entropy_at(self, timestep: Any) -> jax.Array:
        """The target entropy at timestep."""
        return jnp.asarray(at(self.target_entropy, timestep))

    def temperature_update_due(self, timestep: Any) -> Union[bool, jax.Array]:
        """Whether the temperature's step counts at timestep: from
        alpha_update_start on (update_steps.reached)."""
        return reached(self.alpha_update_start, timestep)


@dataclass(kw_only=True, eq=False, repr=False)
class PolicyUpdateStart:
    """An actor-critic whose actor's steps count from a timestep on, every
    replay actor-critic's knob (SAC's, kept for its probes and its users).

    Args:
        policy_update_start: The env steps collected before the actor's
            steps are kept (0: from the first update).
    """

    policy_update_start: int = 0

    def policy_update_due(self, timestep: Any) -> Union[bool, jax.Array]:
        """Whether the actor's step counts at timestep: from
        policy_update_start on (update_steps.reached)."""
        return reached(self.policy_update_start, timestep)

    def keeps_actor_metrics(self) -> bool:
        """An actor waiting for policy_update_start skips the updates before
        it, which log its last step's metrics (update_steps.run_actor_if)."""
        return self.policy_update_start > 0


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
        network_settings = self.network_settings()
        if network_settings:
            self.network_args = self.network_args.replace(**network_settings)
        self._build_family()
        self.setup()
        object.__setattr__(self, "_built", True)

    def __setattr__(self, name: str, value: Any) -> None:
        # Fixed once built (owner, 2026-10-24), so what trains is what the
        # run config logs; other attributes (network_args, a monkeypatched
        # method) stay writable.
        built = getattr(self, "_built", False)
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

    def network_settings(self) -> dict[str, Any]:
        """NetworkConfig settings the reference fixes beyond ActorCritic's
        arguments (REDQ's squash_log_det_eps, TQC's critic_quantiles)."""
        return {}

    def extension_values(self, configs: dict[str, Any]) -> dict[str, Any]:
        """Values only the agent has, which its extension_context may name
        (configs: the run's env_args, network_args and optimisers)."""
        del configs
        return {}

    def evaluation_options(self) -> dict[str, Any]:
        """Each evaluation's options beyond the expert and the critic's
        estimate (ajax.log.evaluate_and_log's: APO's
        avg_reward_mode)."""
        return {}

    def evaluation_kwargs(self) -> dict[str, Any]:
        """Each evaluation's arguments: the expert, the critic's estimate,
        the expert whose return it logs (eval_expert_policy, in agent units)
        and the agent's evaluation_options."""
        kwargs = {"expert_policy": self.expert, "value_estimate": self.value_estimate()}
        eval_expert = getattr(self, "eval_expert_policy", None)
        if eval_expert is not None:
            kwargs.update({"eval_expert_policy": self._in_agent_units(eval_expert)})
        kwargs.update(self.evaluation_options())
        return kwargs

    def uniform_steps(self) -> Optional[int]:
        """The steps an off-policy family's collection acts uniformly for:
        the warm-up, until learning_starts (None), for an agent claiming the
        warmup capability; none otherwise (TQC's authors act with the policy
        from the first step)."""
        return None if "warmup" in self.capabilities else 0

    def keeps_actor_metrics(self) -> bool:
        """Whether the update keeps its actor's last metrics on the state, for
        the updates whose actor step it skips to log
        (update_steps.run_actor_if): the loop gives the state their slot."""
        return False

    def setup(self) -> None:
        """What the agent derives or checks once its env, networks and family
        are built: IQL's value optimiser, PPO's checks of its rollout split
        (they need n_envs)."""

    def check_recurrent_options(self, extensions: Sequence[Extension]) -> None:
        """Refuse what a run with memory cannot take: an expert or
        extensions, which the shared collection does not carry through a
        recurrent policy (APG's rollout does)."""
        unsupported_recurrent_options(
            self.name, expert_policy=self.expert, extensions=tuple(extensions) or None
        )

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

    def make_context(
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
        agent.make_context(n) to call init or update alone."""
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
        recurrent = network_args.memory is not None
        given = dict(configs)
        given.update(
            {
                "expert_policy": self.expert,
                "buffer": buffer,
                "extensions": tuple(extensions),
                "recurrent": recurrent,
            }
        )
        declared = self.field_names()
        given.update(
            {
                name: getattr(self, name)
                for name in ("gamma", "reward_scale", "num_critics")
                if name in declared
            }
        )
        given.update(self.extension_values(configs))
        unknown = sorted(self.extension_context.difference(given.keys()))
        if unknown:
            raise ValueError(
                f"{self.name}'s extension_context names {', '.join(unknown)},"
                " which DeclaredAgent.make_context does not give its extensions (it"
                f" gives {', '.join(sorted(given))})."
            )
        agent_context = {name: given[name] for name in sorted(self.extension_context)}
        bound = ExtensionStack(extensions).bind_to_agent(**agent_context).extensions
        if recurrent:
            self.check_recurrent_options(bound)
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
            make_collector=partial(
                self.make_collector, loop=loop, network_args=network_args
            ),
            buffer=buffer,
            recurrent=recurrent,
            executed=executed_fn(env_args),
        )
        return loop, context

    def make_collector(
        self, key: jax.Array, loop: TrainLoop, network_args: NetworkConfig, **options
    ) -> Any:
        """The collector state (init_collector_state), holding a
        stateful expert's state, with options an agent's own."""
        del network_args
        return init_collector_state(
            key,
            env_args=loop.env_args,
            mode=loop.env_args.mode,
            expert_policy=self.expert,
            **options,
        )


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
    (choose_behaviour).

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
    # The default pipelines run the warm-up's chooser (choose_warmup_action);
    # an agent replacing them passes it on (choose_behaviour). The shared
    # collector writes the replay (choose_replay_action).
    supported_extension_phases = LOOP_PHASES.union(
        {
            "on_target",
            "critic_loss",
            "actor_loss",
            "transform_action",
            "actuate",
            "choose_warmup_action",
            "choose_replay_action",
        }
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

    def choose_behaviour(self, context: TrainContext) -> Optional[Callable]:
        """The action pipeline collection runs; None: the policy's, uniform
        during the warm-up or its chooser's. An agent's own runs the
        chooser too (context.env_args.warmups)."""
        del context
        return None

    def _build_family(self) -> None:
        self.buffer = self.make_replay_buffer()

    def make_replay_buffer(self) -> BufferType:
        return get_buffer(self.buffer_size, self.batch_size, self.n_envs)

    def stores_state(self) -> bool:
        """Whether the replay keeps the actor's recurrent state (R2D2's
        stored state)."""
        return False

    def make_collector(
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
            count_episode_steps=any(
                extension.reads_step_in_episode for extension in loop.extensions
            ),
            **options,
        )

    def make_train(
        self,
        env_args: EnvironmentConfig,
        actor_optimizer_args: OptimizerConfig,
        critic_optimizer_args: OptimizerConfig,
        network_args: NetworkConfig,
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
        loop, context = self.make_context(
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
            uniform_steps=self.uniform_steps(),
            recurrent=recurrent,
            collect_kwargs={
                "buffer": self.buffer,
                "action_pipeline": self.choose_behaviour(context),
                "store_hidden": recurrent and self.stores_state(),
            },
            eval_kwargs=self.evaluation_kwargs(),
            keeps_actor_metrics=self.keeps_actor_metrics(),
        )

    def extension_values(self, configs: dict[str, Any]) -> dict[str, Any]:
        """A discounted Q critic with targets gives BellmanPretrain its
        critic step on a replay batch without any extension
        (update_value_fn, the agent's update_critic) and the targets'
        Polyak update (update_target_fn): an agent naming them in its
        extension_context writes update_critic(agent_state, batch,
        carries, context) -> (agent_state, aux with critic_loss, metrics)."""
        wanted = {"discounted", "q_critic", "target_critic"}
        if not wanted.issubset(self.capabilities) or not hasattr(self, "update_critic"):
            return {}
        update_critic = self.update_critic

        def update_critic_alone(
            agent_state: Any,
            batch: Transition,
            recurrent: bool,
            gamma: float,
            extensions: Any,
            reward_scale: float = 1.0,
        ) -> tuple[Any, Any, dict]:
            # BellmanPretrain's call (pretrain_critic_bellman). The step reads
            # the agent's own gamma and reward scale, with no extension, no
            # residual policy (executed None) and a horizon of 1, as SAC's
            # update_value_functions was called before B4.
            del gamma, reward_scale
            context = TrainContext(
                **configs,
                extensions=extensions,
                total_timesteps=1,
                make_collector=self.make_collector,
                buffer=self.buffer,
                recurrent=recurrent,
            )
            return update_critic(agent_state, batch, None, context)

        return {
            "update_value_fn": update_critic_alone,
            "update_target_fn": update_steps.update_target_networks,
        }


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

    def make_replay_buffer(self) -> BufferType:
        return make_recurrent_replay_buffer(
            self,
            self.n_envs,
            self.network_args.memory,
            self.buffer_size,
            self.batch_size,
        )

    def stores_state(self) -> bool:
        return self.stored_state


@dataclass(kw_only=True, eq=False, repr=False)
class IncrementalAgent(DeclaredAgent):
    """An agent learning from each step as it comes, without a replay
    (TrainLoop.off_policy): one step per env per iteration, then one update
    on that step alone, for each step collected from learning_starts on, so
    never on a warm-up step (AVG, Vasan et al., 2024). An agent writes init
    and update, and may record statistics of every collected step
    (update_step_statistics).

    Args:
        learning_starts: The env steps collected before the first step the
            agent learns from.
        expose_recent_rollout: Keep each step on agent_state.last_rollout,
            a one-step rollout, for measurement extensions.
    """

    learning_starts: int
    expose_recent_rollout: bool = False

    # The default pipeline runs the warm-up's chooser (choose_warmup_action).
    supported_extension_phases = LOOP_PHASES.union(
        {
            "on_target",
            "critic_loss",
            "actor_loss",
            "transform_action",
            "actuate",
            "choose_warmup_action",
        }
    )
    pretrain_context = frozenset({"env_args", "update"})
    extension_context = frozenset({"actor_optimizer_args", "env_args", "expert_policy"})

    def init(self, key: jax.Array, context: TrainContext) -> Any:
        """A fresh agent state, on the agent's own key layout."""
        raise NotImplementedError

    def update(
        self, agent_state: Any, transition: Transition, context: TrainContext
    ) -> tuple[Any, Any]:
        """One update on the step just collected: (agent_state, logged
        metrics)."""
        raise NotImplementedError

    def update_step_statistics(
        self, agent_state: Any, transition: Transition, context: TrainContext
    ) -> Any:
        """What the agent records from every collected step, before the
        update (AVG's TD-error scale)."""
        del transition, context
        return agent_state

    def make_train(
        self,
        env_args: EnvironmentConfig,
        actor_optimizer_args: OptimizerConfig,
        critic_optimizer_args: OptimizerConfig,
        network_args: NetworkConfig,
        total_timesteps: int,
        num_episode_test: int,
        run_ids: Optional[Sequence[str]] = None,
        logging_config: Any = None,
        start_timestep: int = 0,
        end_timestep: Optional[int] = None,
        extensions: Sequence[Extension] = (),
    ) -> Callable:
        """The train function ActorCritic.train builds: the loop over init,
        update_step_statistics and update. It takes no dataset: without a
        replay there is no offline training (train_offline refuses it)."""
        loop, context = self.make_context(
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
        stores_raw_next_obs = "raw_next_obs" in loop.extensions.buffer_fields()
        return loop.off_policy(
            partial(self.init, context=context),
            partial(self.update, context=context),
            self.learning_starts,
            uniform_steps=self.uniform_steps(),
            expose_rollout=self.expose_recent_rollout,
            after_collect=partial(self.update_step_statistics, context=context),
            collect_kwargs={"store_raw_next_obs": stores_raw_next_obs},
            eval_kwargs=self.evaluation_kwargs(),
            keeps_actor_metrics=self.keeps_actor_metrics(),
        )


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
    collection's action pipeline (choose_behaviour) and add to each
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

    def choose_behaviour(self, context: TrainContext) -> Optional[Callable]:
        """The action pipeline collection runs; None: the policy's sample."""
        del context
        return None

    def make_train(
        self,
        env_args: EnvironmentConfig,
        actor_optimizer_args: OptimizerConfig,
        critic_optimizer_args: OptimizerConfig,
        network_args: NetworkConfig,
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
        loop, context = self.make_context(
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
            collect_kwargs={"action_pipeline": self.choose_behaviour(context)},
            eval_kwargs=self.evaluation_kwargs(),
        )


@dataclass(kw_only=True, eq=False, repr=False)
class OwnIterationAgent(DeclaredAgent):
    """An agent whose iteration is its own (TrainLoop.train): it acts and
    learns its way, then the loop evaluates on its cadence, logs and
    resumes. UDRL collects a segment and fits its replay of segments
    (Schmidhuber, 2019); APG takes a gradient step through a differentiable
    rollout.

    An agent writes init, iterate (applying the extensions' post_update
    where its order puts it), report (the training metrics an evaluation
    logs), evaluate and count_iteration_steps. It may take options per train
    call (configure_call), and return every iteration's output instead of
    the evaluations (returns_iterations, make_iteration_output). It takes no
    dataset: train_offline refuses it.
    """

    # The loop's phases but observe, which no shared collector or evaluation
    # meets; an agent adds the phases its iteration applies.
    supported_extension_phases = LOOP_PHASES.difference({"observe"})
    #: Whether train returns every iteration's output (make_iteration_output),
    #: stacked, logging or not, instead of the evaluations.
    returns_iterations = False

    def init(self, key: jax.Array, context: TrainContext) -> Any:
        """A fresh agent state, on the agent's own key layout."""
        raise NotImplementedError

    def iterate(self, agent_state: Any, context: TrainContext) -> tuple[Any, Any]:
        """One iteration, the extensions' post_update applied: (agent_state,
        aux, which report and make_iteration_output read)."""
        raise NotImplementedError

    def report(self, agent_state: Any, aux: Any) -> dict:
        """The training metrics an evaluation logs, timestep included."""
        raise NotImplementedError

    def evaluate(
        self,
        agent_state: Any,
        key: jax.Array,
        context: TrainContext,
        num_episode_test: int,
    ) -> dict:
        """The policy's evaluation metrics, on num_episode_test episodes."""
        raise NotImplementedError

    def count_iteration_steps(self, context: TrainContext) -> tuple[int, str]:
        """The env steps one iteration takes, and how a warning names them."""
        raise NotImplementedError

    def configure_call(
        self, context: TrainContext, n_iterations: int
    ) -> tuple[TrainContext, Optional[Callable[[Any], Any]]]:
        """The context of one train call of n_iterations iterations, and how
        a state it resumes from is transformed (None: as it is). An agent
        taking options per call declares them as keywords here."""
        del n_iterations
        return context, None

    def make_iteration_output(self, aux: Any, evaluations: Optional[dict]) -> Any:
        """What an iteration returns: its evaluations, which the loop records
        (None without logging); with returns_iterations, the agent's own."""
        del aux
        return evaluations

    def make_train(
        self,
        env_args: EnvironmentConfig,
        actor_optimizer_args: OptimizerConfig,
        critic_optimizer_args: OptimizerConfig,
        network_args: NetworkConfig,
        total_timesteps: int,
        num_episode_test: int,
        run_ids: Optional[Sequence[str]] = None,
        logging_config: Any = None,
        start_timestep: int = 0,
        end_timestep: Optional[int] = None,
        extensions: Sequence[Extension] = (),
        **call_options: Any,
    ) -> Callable:
        """The train function ActorCritic.train builds: the loop over init
        and iterate, call_options going to configure_call. It takes no
        dataset (train_offline refuses it)."""
        # An option configure_call does not take is refused before
        # make_context starts the logging worker.
        options = set(inspect.signature(self.configure_call).parameters)
        options = options.difference({"context", "n_iterations"})
        unknown = sorted(set(call_options).difference(options))
        if unknown:
            raise TypeError(f"{self.name}.train takes no {', '.join(unknown)}.")
        loop, context = self.make_context(
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
        steps, unit = self.count_iteration_steps(context)
        loop.warn_if_partial(steps, unit)
        n_iterations = max(loop.budget // steps, 1)
        context, resume = self.configure_call(context, n_iterations, **call_options)
        evaluation = Evaluation(
            metrics=self.report,
            every=loop.log_frequency and max(loop.log_frequency // steps, 1),
            evaluate=partial(
                self.evaluate, context=context, num_episode_test=num_episode_test
            ),
        )

        def iteration(agent_state: Any, index: Any, tick: Any) -> tuple[Any, Any]:
            agent_state, aux = self.iterate(agent_state, context)
            agent_state, evaluations = loop.evaluate_every(
                agent_state, aux, index, tick, evaluation
            )
            return agent_state, self.make_iteration_output(aux, evaluations)

        return loop.train(
            partial(self.init, context=context),
            iteration,
            n_iterations,
            evaluations=evaluation.count(n_iterations),
            resume=resume,
            returns_iterations=self.returns_iterations,
        )
