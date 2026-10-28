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

An agent learning from a replay of transitions is a ReplayAgent, one
learning from each rollout of its current policy an OnPolicyAgent, one
whose iteration is its own (a differentiable rollout, a replay of segments)
an OwnIterationAgent, and one storing rows in a replay of its own, sized
per call (a world model's), an OwnReplayAgent. The shared knobs an
algorithm may take beyond the ones every agent takes (gradient clipping,
normalisation, an expert, image observations) are field mixins. The family
reads some fields by name (DeclaredAgent, ReplayAgent, OnPolicyAgent).
Every agent, family and mixin carries the literal @dataclass decorator,
which the API docs read the constructor from; the contract an agent declares
for the extensions and the loop (name, capabilities, ...) stays
unannotated, so it is no field.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import KW_ONLY, dataclass
from functools import partial
from typing import Any, ClassVar, Optional, Union

import jax
import jax.numpy as jnp
from gymnax import EnvParams

from ajax.agents import update_steps
from ajax.agents.base import ActorCritic, run_horizon, shared_counters
from ajax.agents.loop import LOOP_PHASES, Evaluation, TrainLoop

# Imported under another name: ReplayAgent's own hook is make_replay_buffer.
from ajax.agents.recurrent import make_replay_buffer as make_recurrent_replay_buffer
from ajax.agents.recurrent import stored_actor_carry_dim, unsupported_recurrent_options
from ajax.agents.update_steps import REPLAY_BATCH_KEYS, reached
from ajax.buffers.utils import get_buffer, mean_of_written_rows
from ajax.controller import (
    ScoredActions,
    executed_fn,
    make_run_controller,
    make_scored_actions,
    policy_action_dim,
)
from ajax.environments.interaction import init_collector_state
from ajax.environments.row_collector import resume_tick
from ajax.environments.utils import get_state_action_shapes
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import Extension, ExtensionStack, holds_expert
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import (
    CNNSpec,
    action_value,
    get_initialized_actor_critic,
    state_value,
)
from ajax.policy import Policy, PolicyParts, evaluation_policy
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
    - the replay: a replay agent's buffer, an own-replay agent's own (None
      on-policy);
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
    #: A replay agent's BufferType, or an own-replay agent's replay.
    buffer: Any = None
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

    def policy(self, agent_state: Any, seed_index: int = 0) -> Policy:
        """The policy of seed seed_index of agent_state (train's state, or
        its (state, out)), acting in the environment's units exactly as the
        agent's evaluation acts (ajax.policy.Policy)."""
        if isinstance(agent_state, tuple) and len(agent_state) == 2:
            agent_state = agent_state[0]
        one_seed = jax.tree.map(lambda leaf: leaf[seed_index], agent_state)
        return Policy(self.policy_parts(), one_seed)

    def policy_parts(self) -> Callable[[Any], PolicyParts]:
        """How the agent acts on one seed's state: as the shared evaluation
        acts (ajax.policy.evaluation_policy), through the run's controller,
        its extensions bound as in training."""
        loop, _ = self.make_context(1)
        return partial(
            evaluation_policy,
            self.env_args,
            expert_policy=self.expert,
            controller=make_run_controller(self.env_args, None, loop.extensions),
            recurrent=self.network_args.memory is not None,
        )

    def predict(
        self,
        agent_state: Any,
        observation: Any,
        seed_index: int = 0,
        deterministic: bool = True,
    ) -> jax.Array:
        """The action at observation (one, or a batch with a row per env) in
        the environment's units, as agent.policy(agent_state, seed_index)
        acts at an episode's start; the policy is built once per state and
        seed. A policy carrying memory from one step to the next refuses:
        one call cannot carry it, agent.policy does."""
        if not deterministic:
            raise ValueError(
                "predict acts as the evaluation does, on the actor's mean or"
                " most likely action: deterministic=False is not supported."
            )
        cached = getattr(self, "_predicting", None)
        if cached is None or cached[0] is not agent_state or cached[1] != seed_index:
            policy = self.policy(agent_state, seed_index)
            if policy.has_memory:
                raise ValueError(
                    f"{self.name}'s policy carries memory from one step to the"
                    " next (a recurrent actor, a stateful expert or feature, a"
                    " command, a world model's state), which one predict call"
                    " cannot carry: act with policy = agent.policy(state),"
                    " carry = policy.reset(n_envs), policy.act(obs, carry, ...)."
                )
            self._predicting = (agent_state, seed_index, policy)
        policy = self._predicting[2]
        observation = jnp.asarray(observation)
        obs_shape, _ = get_state_action_shapes(self.env_args.env)
        single = observation.ndim == len(obs_shape)
        batch = observation[None] if single else observation
        action, _ = policy.act(batch, policy.reset(batch.shape[0]))
        return action[0] if single else action

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
                for name in ("gamma", "reward_scale", "num_critics", "subset_size")
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
        stateful expert's state and each env's step in its episode for an
        extension that reads it (JSRLCurriculum), with options an agent's
        own."""
        del network_args
        return init_collector_state(
            key,
            env_args=loop.env_args,
            mode=loop.env_args.mode,
            expert_policy=self.expert,
            count_episode_steps=reads_step_in_episode(loop.extensions),
            **options,
        )


def measure_replay(agent_state: Any) -> dict:
    """What each evaluation logs of a replay agent's replay: the share of its
    rows an expert acted in (replay/expert_share), where the replay keeps the
    flags (a run with an expert). Read at each evaluation, not each update,
    over the whole replay, not one batch."""
    buffer_state = agent_state.collector_state.buffer_state
    if buffer_state is None or "is_expert" not in buffer_state.experience:
        return {}
    return {"replay/expert_share": mean_of_written_rows(buffer_state, "is_expert")}


def reads_step_in_episode(extensions: Iterable[Extension]) -> bool:
    """Whether an extension reads each env's step in its episode, which the
    collector then counts."""
    return any(extension.reads_step_in_episode for extension in extensions)


def reads_last_rollout(extensions: Iterable[Extension]) -> bool:
    """Whether an extension reads the agent's last rollout, which the agent
    then keeps."""
    return any(extension.reads_last_rollout for extension in extensions)


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
    # A dataset's next action, which its replay keeps for a reader (TargetBC);
    # a collected row's is unknown (NaN).
    supported_buffer_fields = frozenset({"next_action"})
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

    def evaluation_kwargs(self) -> dict[str, Any]:
        """DeclaredAgent's, and what each evaluation logs of the replay
        (measure_replay)."""
        kwargs = super().evaluation_kwargs()
        kwargs.update({"agent_metrics": measure_replay})
        return kwargs

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
            count_episode_steps=reads_step_in_episode(loop.extensions),
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
            a one-step rollout, even when no extension reads it.
    """

    learning_starts: int
    expose_recent_rollout: bool = False

    keeps_last_rollout = True

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
        # The update learns from the step itself: its critic scores the action
        # the step executed, as a replay agent's scores its rows' actions.
        collect_kwargs = {
            "store_raw_next_obs": stores_raw_next_obs,
            "store_executed_action": True,
        }
        return loop.off_policy(
            partial(self.init, context=context),
            partial(self.update, context=context),
            self.learning_starts,
            uniform_steps=self.uniform_steps(),
            expose_rollout=self.expose_recent_rollout
            or reads_last_rollout(loop.extensions),
            after_collect=partial(self.update_step_statistics, context=context),
            collect_kwargs=collect_kwargs,
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
            agent_state.last_rollout even when no extension reads it.
    """

    n_steps: int
    n_epochs: int
    num_minibatches: int
    expose_recent_rollout: bool = False

    keeps_last_rollout = True

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
        # An update that stays correct on actions its policy did not choose
        # (PQN's Q-learning) learns from the actions the steps executed, which
        # the action stage may substitute (JSRL): their rewards and next
        # observations are the rollout's.
        collect_kwargs = {
            "action_pipeline": self.choose_behaviour(context),
            "store_executed_action": "off_policy" in self.capabilities,
        }
        return loop.on_policy(
            partial(self.init, context=context),
            partial(self.update, context=context),
            self.n_steps,
            recurrent=context.recurrent,
            expose_rollout=self.expose_recent_rollout
            or reads_last_rollout(loop.extensions),
            forced_resets=(
                getattr(self, "num_resets_per_eval", 0),
                getattr(self, "num_evals", 1),
            ),
            collect_kwargs=collect_kwargs,
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

    # The loop's phases but observe, action and eval_action, which no shared
    # collector or evaluation meets; an agent adds the phases its iteration
    # applies.
    supported_extension_phases = LOOP_PHASES.difference(
        {"observe", "action", "eval_action"}
    )
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

    def policy_parts(self) -> Callable[[Any], PolicyParts]:
        """How the agent acts: its own, as its evaluation is."""
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


@dataclass(kw_only=True, eq=False, repr=False)
class OwnReplayAgent(DeclaredAgent):
    """An agent storing rows in a replay of its own (TrainLoop.off_policy's
    hooks): each tick its row collector acts and stores one row per env, it
    runs the updates its schedule gives the absolute tick, and the loop
    evaluates on its cadence, logs and resumes. DreamerV3 trains on windows
    of the replay at a train ratio (Hafner et al., 2023).

    It is no ReplayAgent: its replay holds rows, not transitions, sized for
    each call (prepare_replay), and it takes no dataset. It assumes the row
    collector (ajax.environments.row_collector): a resumed run starts at the
    tick of the rows it counted, and each env copy may carry its own system.

    An agent writes prepare_replay, init, collect, count_updates, update,
    report and evaluate; it may count its ticks (count_ticks), restart what
    it logs after each log (make_after_log) and check a call (check_run).
    """

    # The own-iteration family's phases: no shared collector or evaluation
    # meets an observation.
    supported_extension_phases = OwnIterationAgent.supported_extension_phases
    pretrain_context = frozenset({"env_args", "update"})
    supports_batched_env_params = True

    def prepare_replay(
        self, start_tick: int, end_tick: int, agent_state: Any
    ) -> tuple[Any, Any]:
        """The replay of a call training from tick start_tick to end_tick,
        and the state it resumes from (None for a fresh run), its replay
        moved into that replay's shapes. The agent records the sizes."""
        raise NotImplementedError

    def init(self, key: jax.Array, context: TrainContext) -> Any:
        """A fresh agent state, on the agent's own key layout."""
        raise NotImplementedError

    def collect(
        self, agent_state: Any, tick: jax.Array, context: TrainContext
    ) -> tuple[Any, Any]:
        """Act and store one row per env at the absolute tick: (agent_state,
        what update reads)."""
        raise NotImplementedError

    def count_updates(
        self, agent_state: Any, tick: jax.Array, context: TrainContext
    ) -> jax.Array:
        """The updates run after the tick's rows (an int32 count)."""
        raise NotImplementedError

    def update(
        self, agent_state: Any, experience: Any, context: TrainContext
    ) -> tuple[Any, Any]:
        """One update on the replay; the loop folds post_update after it."""
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

    def policy_parts(self) -> Callable[[Any], PolicyParts]:
        """How the agent acts: its own, as its evaluation is."""
        raise NotImplementedError

    def count_ticks(self, n_timesteps: int, start_tick: int) -> int:
        """The ticks of n_timesteps from start_tick: one row per env each."""
        del start_tick
        return n_timesteps // self.env_args.n_envs

    def make_after_log(self) -> Optional[Callable[[Any, jax.Array], Any]]:
        """What runs on every tick after the logging gate (Evaluation's
        after_log). None by default: an identity would still trace the gate."""
        return None

    def check_run(self, agent_state: Any, end_tick: int) -> None:
        """Check the state a call ended with, at tick end_tick."""
        del agent_state, end_tick

    def value_estimate(self) -> Optional[ValueEstimate]:
        """None: the agent evaluates itself (evaluate), so the shared
        evaluation's critic estimate is never compared; a world model's
        critic reads its latent state, not the observation that estimate
        passes."""
        return None

    def resume_iteration_offset(self, initial_state: Any) -> int:
        """The absolute tick of a resumed run, equal across seeds
        (row_collector.resume_tick), so that every schedule of the tick
        continues instead of restarting."""
        return resume_tick(initial_state.collector_state, self.env_args.n_envs)

    def train(
        self,
        seed: Union[int, Sequence[int]] = 42,
        n_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Any = None,
        on_ids_ready: Optional[Callable] = None,
        initial_state: Any = None,
        total_timesteps: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """ActorCritic.train on the call's replay (prepare_replay), which
        continues the replay of the uninterrupted run when it resumes."""
        state = initial_state
        if isinstance(state, tuple) and len(state) == 2:
            state = state[0]
        # Refused before prepare_replay records the replay's sizes on the
        # agent: a refused call leaves the agent as it was.
        start = 0
        if state is not None:
            counters = shared_counters(state, self.per_seed_update_counts)
            start = int(counters.get("timestep", 0))
        run_horizon(start, n_timesteps, total_timesteps)
        start_tick = 0 if state is None else self.resume_iteration_offset(state)
        end_tick = start_tick + self.count_ticks(n_timesteps, start_tick)
        replay, state = self.prepare_replay(start_tick, end_tick, state)
        agent_state, out = super().train(
            seed=seed,
            n_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
            logging_config=logging_config,
            on_ids_ready=on_ids_ready,
            initial_state=state,
            total_timesteps=total_timesteps,
            replay=replay,
            start_tick=start_tick,
            **kwargs,
        )
        self.check_run(agent_state, end_tick)
        return agent_state, out

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
        *,
        replay: Any,
        start_tick: int = 0,
    ) -> Callable:
        """The train function ActorCritic.train builds: off_policy's ticks of
        the agent's collect, count_updates and update on the call's replay,
        evaluating every log_frequency timesteps. It takes no dataset
        (train_offline refuses it)."""
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
        context = dataclasses.replace(context, buffer=replay)
        loop.warn_if_partial(env_args.n_envs, "n_envs")
        log_frequency = loop.log_frequency
        evaluation = Evaluation(
            metrics=self.report,
            every=log_frequency and max(self.count_ticks(log_frequency, 0), 1),
            evaluate=partial(
                self.evaluate, context=context, num_episode_test=num_episode_test
            ),
            after_log=self.make_after_log(),
        )

        def count_updates(tick: jax.Array, agent_state: Any) -> jax.Array:
            return self.count_updates(agent_state, tick, context)

        return loop.off_policy(
            partial(self.init, context=context),
            partial(self.update, context=context),
            collect=partial(self.collect, context=context),
            n_updates=count_updates,
            evaluation=evaluation,
            num_ticks=self.count_ticks(loop.budget, start_tick),
        )
