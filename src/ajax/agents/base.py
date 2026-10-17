from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from gymnax import EnvParams

from ajax.environments.create import prepare_env
from ajax.environments.system_class import env_params_is_batched
from ajax.environments.utils import ExpertInAgentUnits
from ajax.extensions.base import (
    Extension,
    ExtensionStack,
    check_extension_phases,
)
from ajax.logging.wandb_logging import (
    LoggingConfig,
    init_logging,
    new_run_id,
    stop_async_logging,
)
from ajax.networks.memory import MemoryConfig, parse_memory_config
from ajax.networks.networks import CNNSpec
from ajax.state import (
    BaseAgentConfig,
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
    return config | {"algo_name": name}


def shared_counters(state: Any) -> dict[str, jax.Array]:
    """The counters of a resumed ``state`` (leading seed axis) its seeds
    share, as unbatched scalars: the collector's timestep, the number of
    evaluations logged and, when every seed has done as many, the number of
    updates (``{}`` for a state without them).

    They gate the evaluations (:func:`ajax.log.evaluate_and_log`) and the
    updates' schedules (TD3's policy delay, DQN's target refresh, PPO's
    forced resets). A fresh run computes them from constants, unbatched
    under the seed vmap; left batched, a resumed run's gate would turn
    ``lax.cond`` into a select that runs both branches on every iteration.
    The update counts may differ across seeds (TD-MPC2's wait on each
    seed's episodes); they then stay batched.
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
            if name == "n_updates":
                continue
            raise ValueError(
                f"cannot resume: the seeds' {name} differ ({seeds.tolist()});"
                " every seed of a resumed run must stand at the same point"
            )
        out[name] = jnp.asarray(seeds[0])
    return out


def with_shared_counters(state: Any, counters: dict[str, jax.Array]) -> Any:
    """``state`` with the counters of :func:`shared_counters`."""
    for name in ("n_logs", "n_updates"):
        if name in counters:
            state = state.replace(**{name: counters[name]})
    if "timestep" in counters:
        collector = state.collector_state.replace(timestep=counters["timestep"])
        state = state.replace(collector_state=collector)
    return state


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
    # Whether the agent takes per-copy (batched) env params, one system per
    # env (ajax.environments.system_class). The shared loop's collector
    # compiles env_args as a static argument, which jax cannot hash with
    # array leaves.
    supports_batched_env_params: bool = False
    # Extension phases this agent's training loop folds; an extension that
    # implements any other phase is rejected at construction instead of
    # being silently ignored. Each agent declares the phases it folds (on
    # the shared loop: ajax.agents.loop.LOOP_PHASES plus its update's); the
    # default accepts none.
    supported_extension_phases: frozenset = frozenset()
    # Replay-buffer fields beyond its own this agent stores on an
    # extension's declaration (Extension.buffer_fields); any other is
    # rejected at construction.
    supported_buffer_fields: frozenset = frozenset()
    # Whether the agent stores its expert_policy's actions from the expert's
    # state (a_expert, next_a_expert), which extensions may read
    # (check_extension_experts).
    stores_expert_actions: bool = False
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
        log_std_init: float = -1.0,
        mean_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_layer_norm: bool = False,
        squash: bool = False,
        episode_length: Optional[int] = None,
        apply_obs_normalization: bool = True,
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
                from by a running standard deviation (logged returns stay
                in the env's units). By default the single-step reward's.
            reward_normalization_gamma (Optional[float]): with
                ``normalize_rewards``, divide by the running standard
                deviation of the discounted return with this discount
                instead, as SB3's ``VecNormalize`` (``gamma``) and
                gymnasium's ``NormalizeReward`` do.
            action_repeat (int): simulator steps per agent step on brax /
                playground envs (``episode_length`` then counts simulator
                steps); stored on ``env_args``. gymnax envs and prebuilt
                envs support only 1.
        """

        if reward_normalization_gamma is not None and not normalize_rewards:
            raise ValueError(
                "reward_normalization_gamma sets how normalize_rewards normalises"
                " the rewards: pass normalize_rewards=True with it."
            )
        memory = parse_memory_config(memory)
        if memory is not None and not self.supports_memory:
            raise NotImplementedError(
                f"{type(self).__name__} does not support recurrent networks"
                " (memory) yet; supported agents: PPO, SAC, ASAC, REDQ, TD3, APG."
            )

        env, env_params, env_id, continuous = prepare_env(
            env_id,
            env_params=env_params,
            normalize_obs=normalize_observations,
            normalize_reward=normalize_rewards,
            gamma=reward_normalization_gamma,
            n_envs=n_envs,
            episode_length=episode_length,
            apply_obs_normalization=apply_obs_normalization,
            action_repeat=action_repeat,
        )

        if env_params_is_batched(env_params) and not self.supports_batched_env_params:
            raise ValueError(
                f"{type(self).__name__} takes one env_params for every copy of"
                " the environment; these carry one system per copy (a"
                " SystemClass sample). APG, DreamerV3 and TDMPC2 take"
                " per-copy systems; APG also samples them, with system_class."
            )
        self.env_args = EnvironmentConfig(
            env=env,
            env_params=env_params,
            n_envs=n_envs,
            continuous=continuous,
            action_repeat=action_repeat,
        )

        self.network_args = NetworkConfig(
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            memory=memory,
            actor_bias_init=actor_bias_init,
            critic_bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=cnn_extra_obs_dim,
            cnn_spec=cnn_spec,
            log_std_state_independent=log_std_state_independent,
            log_std_init=log_std_init,
            mean_kernel_init=mean_kernel_init,
            encoder_layer_norm=encoder_layer_norm,
            squash=squash,
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

        self.agent_config = BaseAgentConfig()
        #: ``make_train``'s keyword arguments proper to the agent, beyond the
        #: ones every agent's takes (:meth:`train`); each agent fills it.
        self.train_kwargs: dict[str, Any] = {}

        # Composable research features (expert guidance, instrumentation,
        # …). The stack is static — agents thread it into make_train as a
        # static argument and fold it at the phase points of their
        # training step. An empty stack is a true no-op. See
        # `ajax.extensions.base`.
        self.extension_stack = ExtensionStack(extensions)
        self._check_extension_phases()
        # The expert door: an expert speaks env units, the agent reads it in
        # agent units, [-1, 1], as its own actions (checked raw above).
        self._experts_in_agent_units: dict[int, tuple[Any, Any]] = {}
        expert = getattr(self, "expert_policy", None)
        if expert is not None:
            self.expert_policy = self._in_agent_units(expert)
        self.extension_stack = self.extension_stack.with_experts(self._in_agent_units)

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

    def _check_extension_phases(self) -> None:
        """Reject extensions implementing phases this agent never folds, or
        reading buffer fields it does not store."""
        check_extension_phases(
            type(self).__name__,
            self.extension_stack,
            self.supported_extension_phases,
            self.supported_buffer_fields,
            # Set before this initialiser by the agents that take one.
            agent_expert=getattr(self, "expert_policy", None),
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
        **kwargs,
    ) -> tuple[BaseAgentState, Any]:
        """
        Train the agent, every seed at once.

        Args:
            n_timesteps (int): Timesteps this call trains for.
            num_episode_test (int): Number of episodes for evaluation during training.
            initial_state: a state (or ``(state, out)``) ``train`` returned,
                to resume from: the run continues from its timestep, and its
                schedules (training fraction, extensions, exploration, the
                logging's end) run over the whole run, that timestep plus
                ``n_timesteps``.

        Returns ``(state, out)``, each leaf with a leading seed axis. On the
        shared loop (``ajax.agents.loop``) ``out`` is ``None`` without a
        logging config, else the evaluations.
        """
        if isinstance(seed, int):
            seed = [seed]

        if logging_config is not None:
            logging_config.config.update(self.config)
            self.run_ids = [new_run_id() for _ in range(len(seed))]
            for run_id, run_seed in zip(self.run_ids, seed, strict=False):
                init_logging(run_id, logging_config, run_seed=int(run_seed))

        else:
            self.run_ids = []

        if on_ids_ready is not None:
            on_ids_ready(self.run_ids)

        # ``agent.train()`` returns ``(state, out)``: accept either form.
        if isinstance(initial_state, tuple) and len(initial_state) == 2:
            initial_state = initial_state[0]
        counters = {} if initial_state is None else shared_counters(initial_state)
        # The call trains from the run's timestep to its horizon; the start
        # is passed only when non-zero, so fresh calls are unchanged.
        start = int(counters.get("timestep", 0))
        train_jit = self.get_make_train()(
            env_args=self.env_args,
            actor_optimizer_args=self.actor_optimizer_args,
            critic_optimizer_args=self.critic_optimizer_args,
            network_args=self.network_args,
            agent_config=self.agent_config,
            total_timesteps=start + n_timesteps,
            num_episode_test=num_episode_test,
            run_ids=self.run_ids,
            logging_config=logging_config,
            **({"start_timestep": start} if start else {}),
            **kwargs,
        )

        index = jnp.arange(len(seed))
        keys = jax.vmap(jax.random.PRNGKey)(jnp.array(seed))
        if initial_state is None:
            result = jax.vmap(train_jit)(keys, index)
        else:
            # Only passed when non-zero: the default (0) keeps the call --
            # and the train functions that predate the offset -- unchanged.
            iteration_offset = int(self.resume_iteration_offset(initial_state))
            offset_kwargs = (
                {}
                if iteration_offset == 0
                else {"iteration_offset": jnp.asarray(iteration_offset, jnp.int32)}
            )

            def resume(key, index, state, counters, offset_kwargs):
                state = with_shared_counters(state, counters)
                return train_jit(
                    key,
                    index,
                    initial_state=state,
                    resume_from_state=True,
                    **offset_kwargs,
                )

            # The offset and the shared counters are unbatched (in_axes
            # None): every schedule and gate derived from them must stay
            # unbatched under the vmap.
            result = jax.vmap(resume, in_axes=(0, 0, 0, None, None))(
                keys, index, initial_state, counters, offset_kwargs
            )
        # Block until all XLA computation and debug.callbacks complete, then
        # drain and stop the logging worker.  Calling stop_async_logging()
        # inside the vmapped function was wrong: it ran as a Python side effect
        # during vmap *tracing* (before any training executed), killing the
        # worker before any log events were queued.
        jax.block_until_ready(result)
        stop_async_logging()
        return result
