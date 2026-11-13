import copy
import inspect
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Callable, ClassVar, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from gymnax import EnvParams

from ajax.agents.sweep import Sweep, member_row, member_values
from ajax.agents.sweep import check as check_sweep
from ajax.agents.sweep import configurations as sweep_configurations
from ajax.controller import (
    observers_of,
    replay_action_choosers_of,
    transforms_of,
    warmups_of,
)
from ajax.environments.action_stage import check_action_slots
from ajax.environments.create import prepare_env
from ajax.environments.system_class import env_params_is_batched
from ajax.environments.utils import ExpertInAgentUnits, steps_in_python
from ajax.extensions.base import (
    CONTROLLER_PHASES,
    Extension,
    ExtensionStack,
    check_extensions,
)
from ajax.extensions.capabilities import NotWired, run_capabilities
from ajax.logging.wandb_logging import (
    LoggingConfig,
    init_logging,
    new_run_id,
    stop_async_logging,
)
from ajax.networks.memory import MemoryConfig, parse_memory_config
from ajax.networks.networks import CNNSpec
from ajax.state import (
    LOG_STD_BOUNDS,
    BaseAgentState,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import EnvType, InitializationFunction


def run_config(
    arguments: dict[str, Any], name: str, exclude: Sequence[str] = ()
) -> dict[str, Any]:
    """The run config the loggers record (:meth:`ActorCritic.train`): an
    agent constructor's ``arguments`` (its ``locals()``) but ``self``,
    ``__class__`` (present once the constructor calls ``super()``) and
    ``exclude``, and the agent's ``algo_name``."""
    dropped = {"self", "__class__", *exclude}
    config = {k: v for k, v in arguments.items() if k not in dropped}
    config.update({"algo_name": name})
    return config


def _seed_by_seed(train: Callable, keys: jax.Array, index: jax.Array) -> Any:
    """``jax.vmap(train)(keys, index)``, one seed after another, for an env
    that steps in Python: JAX cannot batch its callback inside the ``cond``
    that gates evaluations ("IO effect not supported in vmap-of-cond").
    The seeds share the compilations of train's jitted programs."""
    runs = [train(keys[i], index[i]) for i in range(len(index))]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *runs)


def shared_counters(state: Any, per_seed_updates: bool = False) -> dict[str, jax.Array]:
    """The counters of a resumed ``state`` (leading seed axis) its seeds
    share, as unbatched scalars: the collector's timestep, the number of
    evaluations logged and the number of updates (``{}`` for a state
    without them).

    They gate the evaluations (:func:`ajax.log.evaluate_and_log`) and the
    updates' schedules (TD3's policy delay, DQN's target refresh, PPO's
    forced resets). A fresh run computes them from constants, unbatched
    under the seed vmap; left batched, a resumed run's gate would turn
    ``lax.cond`` into a select that runs both branches on every iteration,
    so seeds standing at different points are refused. ``per_seed_updates``
    (:attr:`ActorCritic.per_seed_update_counts`) keeps differing update
    counts batched instead.
    """
    collector = getattr(state, "collector_state", None)
    found = {
        "timestep": getattr(collector, "timestep", None),
        "n_logs": getattr(state, "n_logs", None),
        "n_updates": getattr(state, "n_updates", None),
    }
    out = {}
    for name, value in found.items():
        if value is None:
            continue
        seeds = np.asarray(jax.device_get(value)).reshape(-1)
        if seeds.size == 0 or np.any(seeds != seeds[0]):
            if name == "n_updates" and per_seed_updates:
                continue
            hint = (
                "; an agent whose seeds update unequally by design declares"
                " per_seed_update_counts"
                if name == "n_updates"
                else ""
            )
            raise ValueError(
                f"cannot resume: the seeds' {name} differ ({seeds.tolist()});"
                f" every seed of a resumed run must stand at the same point{hint}"
            )
        out[name] = jnp.asarray(seeds[0])
    return out


def run_horizon(
    start: int, n_timesteps: int, total_timesteps: Optional[int], unit: str = "timestep"
) -> int:
    """The horizon of a ``train`` call that trains ``n_timesteps`` from
    ``start`` (its ``unit``): the run's ``total_timesteps``, by default the
    call's end, ``start + n_timesteps``. A horizon that ends before the call
    does is refused."""
    end = start + n_timesteps
    horizon = end if total_timesteps is None else int(total_timesteps)
    if horizon < end:
        raise ValueError(
            f"total_timesteps={horizon} ends before this call does: it "
            f"trains from {unit} {start} for n_timesteps={n_timesteps}"
        )
    return horizon


def with_shared_counters(state: Any, counters: dict[str, jax.Array]) -> Any:
    """``state`` with the counters of :func:`shared_counters`."""
    for name in ("n_logs", "n_updates"):
        if name in counters:
            state = state.replace(**{name: counters[name]})
    if "timestep" in counters:
        collector = state.collector_state.replace(timestep=counters["timestep"])
        state = state.replace(collector_state=collector)
    return state


#: The discount of the return whose spread normalises the rewards of an
#: agent without a discount of its own (the average-reward agents): SB3
#: VecNormalize's default gamma.
REWARD_NORMALIZATION_GAMMA = 0.99


class ActorCritic:
    #: The agent's name, its run config's ``algo_name``.
    name: str
    #: The run config the loggers record, from the constructor's arguments
    #: (:func:`run_config`), with the values the agent resolves from them.
    config: dict[str, Any]
    # Agents that implement recurrent (memory-augmented) training set this
    # to True; every other agent gets a loud error instead of a silent
    # misconfiguration when `memory` is provided.
    supports_memory: bool = False
    # Agents whose losses run their networks in training, so that BatchRenorm
    # layers ("batchrenorm" in an architecture) learn their running
    # statistics (CrossQ), set this; every other agent refuses the layer
    # instead of normalising by statistics that never move.
    trains_batch_norm: bool = False
    # Whether the agent takes per-copy (batched) env params, one system per
    # env (ajax.environments.system_class). The shared loop's collector
    # compiles env_args as a static argument, which jax cannot hash with
    # array leaves.
    supports_batched_env_params: bool = False
    # What the algorithm has (ajax.extensions.capabilities'
    # ALGORITHM_CAPABILITIES), whatever the state of the code: an extension
    # needing anything else is refused at construction, naming it.
    capabilities: frozenset = frozenset()
    # Extension phases this agent's training loop folds; an extension that
    # implements any other phase is rejected at construction instead of
    # being silently ignored. Each agent declares the phases it folds (on
    # the shared loop: ajax.agents.loop.LOOP_PHASES plus its update's); the
    # default accepts none.
    supported_extension_phases: frozenset = frozenset()
    # The keys of the agent context its make_train binds its extensions to
    # (Extension.bind_to_agent); an extension needing another is refused.
    extension_context: frozenset = frozenset()
    # Per phase, the keys its folds put in the extensions' batch beyond those
    # every agent passes (ajax.extensions.base.COMMON_BATCH_KEYS); an
    # extension reading another is refused. On the shared steps, their sets
    # in ajax.agents.loop (TARGET_KEYS, CRITIC_LOSS_KEYS, ...) and its own.
    phase_batch_keys: Mapping[str, frozenset] = {}
    # The ExtensionContext fields its loop fills for its extensions'
    # pretraining (the shared loop: env_args and seed_index; off-policy also
    # update, and buffer with a replay).
    pretrain_context: frozenset = frozenset({"env_args", "seed_index"})
    # Replay-buffer fields beyond its own this agent stores on an
    # extension's declaration (Extension.buffer_fields); any other is
    # rejected at construction.
    supported_buffer_fields: frozenset = frozenset()
    # Whether the agent stores its expert_policy's actions from the expert's
    # state (a_expert, next_a_expert), which extensions may read
    # (check_extension_experts).
    stores_expert_actions: bool = False
    # Whether this agent's seeds may stand at different update counts
    # (TD-MPC2's episodic mode sets it: its updates wait on each seed's
    # episodes). A resumed
    # run keeps such counts batched (shared_counters); any other agent's
    # differing counts are refused, since a batched count turns the update
    # schedules' lax.cond into a select running both branches.
    per_seed_update_counts: bool = False
    # Whether the agent replays stored transitions. Its replay then keeps the
    # env's own observations, normalised with the current statistics when
    # read, as SB3's replay buffer does under VecNormalize; an on-policy
    # agent's rollout keeps them normalised with the statistics of their
    # collection (EnvironmentConfig.store_normalized_obs).
    replays_transitions: bool = False
    # Whether the agent can keep its last rollout on agent_state.last_rollout
    # (the on-policy and incremental families), for an extension reading it.
    keeps_last_rollout: bool = False
    # The fields a sweep may vary inside one program (train's sweep,
    # ajax.agents.sweep): numbers the training reads as numbers, which no
    # construction check reads. Each is covered by a test that a sweep trains
    # as separate runs of its configurations would.
    sweepable: ClassVar[frozenset[str]] = frozenset()
    #: Each leading row of the last train call's state: its seed and its
    #: configuration's swept values, configuration-major.
    members: list[dict[str, float]]
    # Set, before this initialiser, by the agents that take an expert.
    expert_policy: Any
    #: The agent's train module's ``make_train`` (:meth:`get_make_train`).
    make_train: Callable[..., Callable]

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 4,
        actor_learning_rate: Union[float, Callable[[int], float]] = 3e-4,
        critic_learning_rate: Union[float, Callable[[int], float]] = 3e-4,
        actor_architecture=("128", "tanh", "128", "tanh"),
        critic_architecture=("128", "tanh", "128", "tanh"),
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        memory: Optional[Union[MemoryConfig, dict]] = None,
        normalize_observations: bool = False,
        observation_clip: Optional[float] = None,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
        critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        cnn_image_shape: Optional[Tuple[int, int, int]] = None,
        cnn_extra_obs_dim: int = 0,
        cnn_spec: Optional[CNNSpec] = None,
        # Brax-style actor head knobs (see NetworkConfig).
        log_std_state_independent: bool = False,
        log_std_init: Optional[float] = -1.0,
        log_std_bounds: Optional[Tuple[float, float]] = LOG_STD_BOUNDS,
        mean_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        logits_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        log_std_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        value_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        tanh_mean: bool = False,
        encoder_layer_norm: bool = False,
        squash: bool = False,
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
        extensions: Sequence[Extension] = (),
    ) -> None:
        """
        Initialize the agent.

        Args:
            actor_learning_rate (float | Callable[[int], float]): Learning
                rate for the actor's optimizer, or its schedule.
            critic_learning_rate (float | Callable[[int], float]): Learning
                rate for the critic's optimizer, or its schedule.
            weight_decay (float): decoupled weight decay of the actor's and
                the critic's optimizers (``OptimizerConfig.weight_decay``):
                on top of the Adam step, each step multiplies their
                parameters by ``1 - lr * weight_decay``. It reaches the
                networks only: a scalar dual variable (SAC's temperature)
                keeps its own plain Adam. 0 keeps plain Adam.
            memory: Pluggable memory block (MemoryConfig or its dict form);
                None keeps the networks feedforward.
            normalize_rewards (bool): divide the rewards the agent learns
                from by the running standard deviation of the discounted
                return and clip them to +-10, as SB3's ``VecNormalize``
                does (logged returns stay in the env's units).
            reward_normalization_gamma (Optional[float]): with
                ``normalize_rewards``, that return's discount: the agent's
                ``gamma`` if None (0.99 for an agent without one), 0 for
                the single-step reward's standard deviation.
            action_repeat (int): simulator steps per agent step on brax /
                playground envs (``episode_length`` then counts simulator
                steps); stored on ``env_args``. gymnax envs and prebuilt
                envs support only 1.
        """

        self.network_args = NetworkConfig(
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            memory=parse_memory_config(memory),
            actor_bias_init=actor_bias_init,
            critic_bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=cnn_extra_obs_dim,
            cnn_spec=cnn_spec,
            log_std_state_independent=log_std_state_independent,
            log_std_init=log_std_init,
            log_std_bounds=log_std_bounds,
            mean_kernel_init=mean_kernel_init,
            logits_kernel_init=logits_kernel_init,
            log_std_kernel_init=log_std_kernel_init,
            value_kernel_init=value_kernel_init,
            tanh_mean=tanh_mean,
            encoder_layer_norm=encoder_layer_norm,
            squash=squash,
        )
        self._refuse_untrainable_networks(self.network_args)
        self.env_args = self._make_environment(
            env_id,
            env_params,
            n_envs,
            normalize_observations,
            normalize_rewards,
            reward_normalization_gamma,
            episode_length,
            action_repeat,
            observation_clip,
        )
        self.actor_optimizer_args = OptimizerConfig(
            learning_rate=actor_learning_rate,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
        )
        self.critic_optimizer_args = OptimizerConfig(
            learning_rate=critic_learning_rate,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
        )

        #: ``make_train``'s keyword arguments proper to the agent, beyond the
        #: ones every agent's takes (:meth:`train`); each agent fills it.
        self.train_kwargs: dict[str, Any] = {}
        self._set_up_extensions(extensions)

    def _refuse_untrainable_networks(self, networks: NetworkConfig) -> None:
        """Refuse the networks this agent cannot train."""
        if networks.memory is not None and not self.supports_memory:
            raise NotImplementedError(
                f"{type(self).__name__} does not support recurrent networks"
                " (memory) yet."
            )
        batch_norm = any(
            isinstance(arch, (tuple, list)) and "batchrenorm" in arch
            for arch in (networks.actor_architecture, networks.critic_architecture)
        )
        if batch_norm and not self.trains_batch_norm:
            raise ValueError(
                f"{type(self).__name__} runs its networks with their running"
                " statistics fixed, so a 'batchrenorm' layer would never learn"
                " them. CrossQ trains them; remove 'batchrenorm' here."
            )

    def _make_environment(
        self,
        env_id: str | EnvType,
        env_params: Optional[EnvParams],
        n_envs: int,
        normalize_observations: bool,
        normalize_rewards: bool,
        reward_normalization_gamma: Optional[float],
        episode_length: Optional[int],
        action_repeat: int,
        observation_clip: Optional[float] = None,
    ) -> EnvironmentConfig:
        """The environment the agent trains on, its rewards normalised by
        the env wrapper and its observations by the agent."""
        if reward_normalization_gamma is not None and not normalize_rewards:
            raise ValueError(
                "reward_normalization_gamma sets how normalize_rewards normalises"
                " the rewards: pass normalize_rewards=True with it."
            )
        if observation_clip is not None and self.replays_transitions:
            raise ValueError(
                "observation_clip clips the observations an on-policy agent stores"
                " normalised; a replay agent normalises inside its networks,"
                " which do not clip."
            )
        if normalize_rewards and reward_normalization_gamma is None:
            # SB3's VecNormalize divides by the spread of the discounted
            # return; RL Zoo, CleanRL and PureJaxRL give it the agent's
            # discount. A constant reward's own spread is 0.
            reward_normalization_gamma = getattr(
                self, "gamma", REWARD_NORMALIZATION_GAMMA
            )
        env, env_params, env_id, continuous = prepare_env(
            env_id,
            env_params=env_params,
            normalize_obs=False,
            normalize_reward=normalize_rewards,
            gamma=reward_normalization_gamma,
            n_envs=n_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
        )
        if env_params_is_batched(env_params) and not self.supports_batched_env_params:
            raise ValueError(
                f"{type(self).__name__} takes one env_params for every copy of"
                " the environment; these carry one system per copy (a"
                " SystemClass sample). APG, DreamerV3 and TDMPC2 take"
                " per-copy systems; APG also samples them, with system_class."
            )
        return EnvironmentConfig(
            env=env,
            env_params=env_params,
            n_envs=n_envs,
            continuous=continuous,
            action_repeat=action_repeat,
            normalize_obs_in_agent=normalize_observations,
            store_normalized_obs=normalize_observations
            and not self.replays_transitions,
            observation_clip=observation_clip,
        )

    def _set_up_extensions(self, extensions: Sequence[Extension]) -> None:
        """Build the agent's extension stack (composable research features:
        expert guidance, instrumentation...), which agents thread into
        make_train as a static argument and fold at the phase points of
        their training step; an empty stack is a true no-op
        (ajax.extensions.base). The extensions acting only in the controller
        move onto env_args."""
        self.extension_stack = ExtensionStack(extensions)
        self._check_extension_phases()
        # The expert door: an expert speaks env units, the agent reads it in
        # agent units, [-1, 1], as its own actions (checked raw above).
        self._experts_in_agent_units: dict[int, tuple[Any, Any]] = {}
        expert = getattr(self, "expert_policy", None)
        if expert is not None:
            self.expert_policy = self._in_agent_units(expert)
        self.extension_stack = self.extension_stack.with_experts(self._in_agent_units)
        self.env_args = self._env_args_with_controller_parts()
        # The action stage runs an action extension in one of its two slots.
        check_action_slots(self.extension_stack)
        self._refuse_expert_actions_as_gains()

        # An extension acting only in the controller (a feature, gain mode, a
        # warm-up, the action a replay stores) takes no part in the learner's
        # folds: no state, no key of its own. A residual policy stays: the
        # learner scores its executed actions.
        def controller_only(extension: Any) -> bool:
            phases = extension.implemented_phases()
            return bool(phases) and phases.issubset(CONTROLLER_PHASES)

        if any(controller_only(e) for e in self.extension_stack):
            self.extension_stack = ExtensionStack(
                [e for e in self.extension_stack if not controller_only(e)]
            )

    def _env_args_with_controller_parts(self) -> EnvironmentConfig:
        """env_args with the stack's parts that the controller
        (ajax.controller), which collection and evaluation run, acts with:
        the observation features, the action transform (residual RL, gain
        mode), the warm-up (ExpertWarmup) and the action a replay stores
        (StorePolicyAction)."""
        observers = observers_of(self.extension_stack)
        transforms = transforms_of(self.extension_stack)
        if len(transforms) > 1:
            raise ValueError(
                "An agent takes one action transform, got"
                f" {[e.name for e in transforms]}."
            )
        dim = None
        if transforms:
            expert = getattr(self, "expert_policy", None)
            with_features = self.env_args.replace(observers=observers)
            dim = transforms[0].policy_action_dim(with_features, expert)
        warmups = warmups_of(self.extension_stack)
        if len(warmups) > 1:
            raise ValueError(
                f"An agent takes one warm-up, got {[warmup.name for warmup in warmups]}."
            )
        choosers = replay_action_choosers_of(self.extension_stack)
        if len(choosers) > 1:
            raise ValueError(
                "An agent stores one action per replay row, got"
                f" {[chooser.name for chooser in choosers]}."
            )
        return self.env_args.replace(
            observers=observers,
            transforms=transforms,
            warmups=warmups,
            replay_action_choosers=choosers,
            policy_action_dim=dim,
        )

    def _refuse_expert_actions_as_gains(self) -> None:
        """Under gain mode the policy chooses gains, and an expert's actions,
        stored (a_expert) or called again, are env actions: an extension
        using them as the policy's would mix the two spaces."""
        mixers = [
            extension
            for extension in self.extension_stack
            if extension.uses_expert_actions_as_policy_actions()
        ]
        if self.env_args.policy_action_dim is not None and mixers:
            reason = "uses the expert's env actions where the policy's gains go"
            unwired = {type(mixer).__name__: reason for mixer in mixers}
            raise NotWired(
                f"{type(self).__name__} cannot run {sorted(unwired)} under gain"
                f" mode yet: each {reason}.",
                unwired,
            )

    def _in_agent_units(self, expert: Any) -> Any:
        """``expert`` read in agent units (:class:`ExpertInAgentUnits`),
        one wrapper per expert: an extension holding the agent's expert
        keeps holding the same object (``reads_stored_expert_actions``)."""
        if expert is None:
            return None
        if id(expert) not in self._experts_in_agent_units:
            wrapped = ExpertInAgentUnits.wrap(
                expert, self.env_args.env, self.env_args.env_params
            )
            # The raw expert is kept so its id is not reused.
            self._experts_in_agent_units[id(expert)] = (expert, wrapped)
        return self._experts_in_agent_units[id(expert)][1]

    def run_capabilities(self) -> frozenset[str]:
        """The capabilities of this run
        (:func:`~ajax.extensions.capabilities.run_capabilities`)."""
        return run_capabilities(
            self.capabilities,
            continuous=self.env_args.continuous,
            # Set before this initialiser by the agents that take one.
            has_expert=getattr(self, "expert_policy", None) is not None,
            num_critics=self.config.get("num_critics"),
            learning_starts=self.config.get("learning_starts"),
            keeps_rollout=self.keeps_last_rollout,
        )

    def _check_extension_phases(self) -> None:
        """Reject extensions this agent cannot run
        (:func:`~ajax.extensions.base.check_extensions`); without extensions
        there is nothing to check."""
        if not self.extension_stack:
            return
        check_extensions(
            type(self).__name__,
            self.extension_stack,
            self.run_capabilities(),
            self.supported_extension_phases,
            context=self.extension_context,
            pretrain_context=self.pretrain_context,
            buffer_fields=self.supported_buffer_fields,
            batch_keys=self.phase_batch_keys,
            agent_expert=getattr(self, "expert_policy", None),
            takes_expert_policy="expert_policy"
            in inspect.signature(type(self).__init__).parameters,
            stores_expert_actions=self.stores_expert_actions,
        )

    def resume_iteration_offset(self, initial_state: BaseAgentState) -> int:
        """Absolute index of the first scan iteration when resuming.

        ``train`` feeds it to the inner train function so a resumed run
        sees absolute iteration indices (see ``build_resumable_train``).
        The default 0 restarts the indices at every ``train`` call (every
        existing agent; APG's per-stage cadence relies on it). Agents whose
        schedules are functions of the absolute tick override this to
        return their tick counter -- a host-side int, equal across seeds.
        """
        del initial_state
        return 0

    def resumed_state(
        self, state: BaseAgentState, n_timesteps: int, train_options: dict
    ) -> BaseAgentState:
        """The state a resumed call of n_timesteps, with train's other
        options, continues from: state itself, but for a replay agent's
        replay, which grows to the uninterrupted run's."""
        del n_timesteps, train_options
        return state

    def sweepable_fields(self) -> frozenset[str]:
        """The fields this agent's sweep may vary: :attr:`sweepable`, less
        those this configuration captures when built."""
        return self.sweepable

    def member(self, values: Mapping[str, Any]) -> "ActorCritic":
        """This agent as one member of a sweep: the fields named in values
        set to them, traced numbers, and the optimiser configs construction
        built from its learning rates rebuilt (ajax.agents.sweep).

        The copy is not constructed again: a sweepable field is one no
        construction check reads, and a traced value could not be checked.
        Its fields are set past the rule that fixes them once built, which
        keeps what trains what the run config logs: a sweep logs each
        member's values in its own run config."""
        member = copy.copy(self)
        for name, value in values.items():
            object.__setattr__(member, name, value)
        for name, config in (
            ("actor_learning_rate", "actor_optimizer_args"),
            ("critic_learning_rate", "critic_optimizer_args"),
        ):
            if name in values:
                rebuilt = getattr(self, config).replace(
                    learning_rate=getattr(member, name)
                )
                object.__setattr__(member, config, rebuilt)
        return member

    def get_make_train(self) -> Callable:
        """:attr:`make_train` with the agent's own arguments
        (:attr:`train_kwargs`) and its extensions; :meth:`train` passes the
        rest."""
        return partial(
            self.make_train,
            **self.train_kwargs,
            extensions=tuple(self.extension_stack.extensions),
        )

    def train(
        self,
        seed: int | Sequence[int] = 42,
        n_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Optional[LoggingConfig] = None,
        on_ids_ready: Optional[Callable] = None,
        initial_state: Optional[BaseAgentState] = None,
        total_timesteps: Optional[int] = None,
        sweep: Optional[Sweep] = None,
        **kwargs,
    ) -> tuple[BaseAgentState, Any]:
        """
        Train the agent, every seed at once.

        Args:
            n_timesteps (int): Timesteps this call trains for. An agent
                that trains in iterations (an on-policy rollout of n_envs x
                n_steps steps) trains the whole iterations it holds, at
                least one, and warns when it is not a whole number of them.
            num_episode_test (int): Number of episodes for evaluation during training.
            initial_state: a state (or ``(state, out)``) ``train`` returned,
                to resume from: the run continues from its timestep.
            total_timesteps: the whole run's horizon, which every schedule
                (training fraction, extensions, exploration, the logging's
                end) runs over; ``None``: this call's end, its start (0, or
                the resumed timestep) plus ``n_timesteps``. Pass the same
                horizon to every call of a run split into calls of whole
                iterations to follow the schedules of one call and end
                where it ends.
            sweep: hyperparameter configurations trained with every seed,
                all in this one program (``ajax.agents.sweep``): a dict of
                value lists, whose grid is trained, or a list of
                configurations, over the agent's sweepable fields
                (:meth:`sweepable_fields`). A sweep resumes with the same
                seed and sweep.

        Returns ``(state, out)``, each leaf with a leading axis over the
        members: the seeds, or the sweep's configurations, each over the
        seeds, as :attr:`members` lists them. On the
        shared loop (``ajax.agents.loop``) ``out`` is ``None`` without a
        logging config, else the evaluations.
        """
        if isinstance(seed, int):
            seed = [seed]

        # ``agent.train()`` returns ``(state, out)``: accept either form.
        if isinstance(initial_state, tuple) and len(initial_state) == 2:
            initial_state = initial_state[0]
        in_python = steps_in_python(self.env_args.env)
        if initial_state is not None and in_python:
            raise ValueError(
                "A Gymnasium environment cannot resume from a saved state: its"
                " simulator's state lives in Python, outside the state, so the"
                " episodes it was in cannot continue."
            )
        found = sweep_configurations(sweep)
        if found:
            check_sweep(type(self).__name__, self.sweepable_fields(), list(found[0]))
            if in_python:
                raise ValueError(
                    "A Gymnasium environment trains its seeds one after"
                    " another: a sweep trains its members in one program."
                )
            if (
                logging_config is not None
                and logging_config.evaluation_device is not None
            ):
                raise ValueError(
                    "A sweep's members evaluate between chunks of training: a"
                    " sweep takes no evaluation_device."
                )
        # Configuration-major: each configuration over the seeds.
        pairs = [
            (configuration, run_seed)
            for configuration in (found or [{}])
            for run_seed in seed
        ]
        if initial_state is not None:
            rows = int(jax.tree.leaves(initial_state)[0].shape[0])
            if rows != len(pairs):
                raise ValueError(
                    f"The state holds {rows} members and this call's seeds and"
                    f" sweep make {len(pairs)}: a run resumes with the same"
                    " seed and sweep."
                )
        self.members = [member_row(configuration, s) for configuration, s in pairs]
        counters = (
            {}
            if initial_state is None
            else shared_counters(initial_state, self.per_seed_update_counts)
        )
        # The call trains from the run's timestep to its end, its schedules
        # run to the horizon; the start and the end are passed only when
        # they differ from 0 and the horizon, so other calls are unchanged.
        start = int(counters.get("timestep", 0))
        end = start + n_timesteps
        horizon = run_horizon(start, n_timesteps, total_timesteps)
        if initial_state is not None:
            initial_state = self.resumed_state(initial_state, n_timesteps, kwargs)

        if logging_config is not None:
            logging_config = logging_config.with_agent_config(self.config)
            self.run_ids = [new_run_id() for _ in pairs]
            for run_id, (configuration, run_seed) in zip(
                self.run_ids, pairs, strict=False
            ):
                # Each member's run config holds its own values.
                init_logging(
                    run_id,
                    logging_config.with_agent_config(configuration),
                    run_seed=int(run_seed),
                )

        else:
            self.run_ids = []

        if on_ids_ready is not None:
            on_ids_ready(self.run_ids)

        def make_train(agent: ActorCritic) -> Callable:
            return agent.get_make_train()(
                env_args=agent.env_args,
                actor_optimizer_args=agent.actor_optimizer_args,
                critic_optimizer_args=agent.critic_optimizer_args,
                network_args=agent.network_args,
                total_timesteps=horizon,
                num_episode_test=num_episode_test,
                run_ids=self.run_ids,
                logging_config=logging_config,
                **({"start_timestep": start} if start else {}),
                **({"end_timestep": end} if end != horizon else {}),
                **kwargs,
            )

        # A sweep's members each build their train function while the vmap
        # traces it, from their own values (member); without a sweep it is
        # built once, here.
        if found:

            def run(key: Any, index: Any, values: dict, **options: Any) -> Any:
                return make_train(self.member(values))(key, index, **options)

        else:
            train_jit = make_train(self)

            def run(key: Any, index: Any, values: dict, **options: Any) -> Any:
                del values  # none without a sweep
                return train_jit(key, index, **options)

        values = member_values([configuration for configuration, _ in pairs])
        index = jnp.arange(len(pairs))
        keys = jax.vmap(jax.random.PRNGKey)(jnp.array([s for _, s in pairs]))
        if in_python:
            result = _seed_by_seed(partial(run, values={}), keys, index)
        elif initial_state is None:
            result = jax.vmap(run)(keys, index, values)
        else:
            # Only passed when non-zero: the default (0) keeps the call --
            # and the train functions that predate the offset -- unchanged.
            iteration_offset = int(self.resume_iteration_offset(initial_state))
            offset_kwargs = (
                {}
                if iteration_offset == 0
                else {"iteration_offset": jnp.asarray(iteration_offset, jnp.int32)}
            )

            def resume(key, index, values, state, counters, offset_kwargs):
                state = with_shared_counters(state, counters)
                return run(
                    key,
                    index,
                    values,
                    initial_state=state,
                    resume_from_state=True,
                    **offset_kwargs,
                )

            # The offset and the shared counters are unbatched (in_axes
            # None): every schedule and gate derived from them must stay
            # unbatched under the vmap.
            result = jax.vmap(resume, in_axes=(0, 0, 0, 0, None, None))(
                keys, index, values, initial_state, counters, offset_kwargs
            )
        # Block until all XLA computation and debug.callbacks complete, then
        # drain and stop the logging worker.  Calling stop_async_logging()
        # inside the vmapped function was wrong: it ran as a Python side effect
        # during vmap *tracing* (before any training executed), killing the
        # worker before any log events were queued.
        jax.block_until_ready(result)
        stop_async_logging()
        return result

    def train_offline(
        self,
        dataset: Any,
        n_updates: int,
        seed: int | Sequence[int] = 42,
        num_episode_test: int = 10,
        logging_config: Optional[LoggingConfig] = None,
        on_ids_ready: Optional[Callable] = None,
    ) -> tuple[BaseAgentState, Any]:
        """Train on a transition dataset alone, with no environment step.

        ``n_updates`` of the agent's own update on a replay holding
        ``dataset``'s rows (:meth:`ajax.agents.loop.TrainLoop.offline`),
        evaluating every ``logging_config.log_frequency`` updates. Returns
        ``(state, out)`` as :meth:`train` does, with the state's timestep
        and evaluation count back at 0, so that ``train(initial_state=state,
        ...)`` fine-tunes online from it, its replay keeping the dataset's
        rows (with ``learning_starts=0``, from the first step).

        Raises ``ValueError`` on an agent without a transition replay.
        """
        takes = inspect.signature(self.make_train).parameters
        if "offline_data" not in takes:
            raise ValueError(
                f"{type(self).__name__} cannot train offline: offline training"
                " runs an agent's update on a replay of transitions (SAC, TD3,"
                " REDQ, ASAC, DQN, IQL)."
            )
        state, out = self.train(
            seed=seed,
            n_timesteps=n_updates,
            num_episode_test=num_episode_test,
            logging_config=logging_config,
            on_ids_ready=on_ids_ready,
            offline_data=dataset,
        )
        collector = state.collector_state
        state = state.replace(
            collector_state=collector.replace(
                timestep=jnp.zeros_like(collector.timestep)
            ),
            n_logs=jnp.zeros_like(state.n_logs),
        )
        return state, out
