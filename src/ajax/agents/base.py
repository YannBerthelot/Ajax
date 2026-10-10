import time
import uuid
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import jax
import jax.numpy as jnp
from gymnax import EnvParams

# Defensive: a broken wandb install must not crash Ajax imports —
# downstream callers using TensorBoard-only (use_wandb=False) should
# still work. ``wandb.util.generate_id()`` is only reached when a
# ``LoggingConfig`` was passed; in that branch we fall back to
# ``uuid.uuid4().hex`` if wandb didn't import.
try:
    import wandb  # type: ignore[import-untyped]
except ImportError:
    wandb = None  # type: ignore[assignment]

from ajax.environments.create import prepare_env
from ajax.extensions.base import (
    PHASES,
    Extension,
    ExtensionStack,
    check_extension_phases,
)
from ajax.logging.wandb_logging import (
    LoggingConfig,
    init_logging,
    stop_async_logging,
)
from ajax.networks.memory import MemoryConfig, parse_memory_config
from ajax.state import (
    BaseAgentConfig,
    BaseAgentState,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import EnvType, InitializationFunction


class ActorCritic:
    # Agents that implement recurrent (memory-augmented) training set this
    # to True; every other agent gets a loud error instead of a silent
    # misconfiguration when `memory` is provided.
    supports_memory: bool = False
    # Extension phases this agent's training loop folds; an extension that
    # implements any other phase is rejected at construction instead of
    # being silently ignored. The check is opt-in per agent: the default
    # (every phase) checks nothing, and the existing agents keep it, so an
    # extension implementing a phase one of them does not fold (e.g.
    # `critic_loss` on APG) is still ignored there. Agents that own their
    # training loop (e.g. the world-model agents) declare the phases they
    # fold.
    supported_extension_phases: frozenset = frozenset(PHASES)

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        n_envs: int = 4,
        actor_learning_rate: Union[float, Callable[[int], float]] = 3e-4,
        critic_learning_rate: Union[float, Callable[[int], float]] = 3e-4,
        actor_architecture=("128", "tanh", "128", "tanh"),
        critic_architecture=("128", "tanh", "128", "tanh"),
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        memory: Optional[Union[MemoryConfig, dict]] = None,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        actor_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
        critic_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_bias_init: Optional[Union[str, InitializationFunction]] = None,
        cnn_image_shape: Optional[tuple] = None,
        cnn_extra_obs_dim: int = 0,
        cnn_spec: Optional[tuple] = None,
        # Brax-style actor head knobs (see NetworkConfig). Default = legacy.
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
        Initialize the PPO agent.

        Args:
            env_id (str | EnvType): Environment ID or environment instance.
            n_envs (int): Number of parallel environments.
            learning_rate (float): Learning rate for optimizers.
            actor_architecture (tuple): Architecture of the actor network.
            critic_architecture (tuple): Architecture of the critic network.
            gamma (float): Discount factor for rewards.
            env_params (Optional[EnvParams]): Parameters for the environment.
            max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            learning_starts (int): Timesteps before training starts.
            tau (float): Soft update coefficient for target networks.
            reward_scale (float): Scaling factor for rewards.
            alpha_init (float): Initial value for the temperature parameter.
            target_entropy_per_dim (float): Target entropy per action dimension.
            memory: Pluggable memory block (MemoryConfig or its dict form);
                None keeps the networks feedforward.
            action_repeat (int): simulator steps per agent step on brax /
                playground envs (``episode_length`` then counts simulator
                steps); stored on ``env_args``. gymnax envs and prebuilt
                envs support only 1.
        """

        memory = parse_memory_config(memory)
        if memory is not None and not self.supports_memory:
            raise NotImplementedError(
                f"{type(self).__name__} does not support recurrent networks"
                " (memory) yet; supported agents: PPO, SAC, ASAC, REDQ, TD3."
            )

        env, env_params, env_id, continuous = prepare_env(
            env_id,
            env_params=env_params,
            normalize_obs=normalize_observations,
            normalize_reward=normalize_rewards,
            n_envs=n_envs,
            episode_length=episode_length,
            apply_obs_normalization=apply_obs_normalization,
            action_repeat=action_repeat,
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
            actor_kernel_init=actor_kernel_init,
            actor_bias_init=actor_bias_init,
            critic_kernel_init=critic_kernel_init,
            critic_bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            encoder_bias_init=encoder_bias_init,
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
            clipped=max_grad_norm is not None,
        )
        self.critic_optimizer_args = OptimizerConfig(
            learning_rate=critic_learning_rate,
            max_grad_norm=max_grad_norm,
            clipped=max_grad_norm is not None,
        )

        self.agent_config = BaseAgentConfig()

        # Composable research features (expert guidance, instrumentation,
        # …). The stack is static — agents thread it into make_train as a
        # static argument and fold it at the phase points of their
        # training step. An empty stack is a true no-op. See
        # `ajax.extensions.base`.
        self.extension_stack = ExtensionStack(extensions)
        self._check_extension_phases()

    def _check_extension_phases(self) -> None:
        """Reject extensions implementing phases this agent never folds."""
        check_extension_phases(
            type(self).__name__, self.extension_stack, self.supported_extension_phases
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
        raise NotImplementedError

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
            seed (int | Sequence[int]): Random seed(s) for training.
            n_timesteps (int): Total number of timesteps for training.
            num_episode_test (int): Number of episodes for evaluation during training.

        Returns ``(state, out)``, each leaf with a leading seed axis. On the
        shared loop (``ajax.agents.loop``) ``out`` is ``None`` without a
        logging config, else the evaluations.
        """
        if isinstance(seed, int):
            seed = [seed]

        if logging_config is not None:
            logging_config.config.update(self.config)
            _gen_id = (
                wandb.util.generate_id
                if wandb is not None
                else lambda: uuid.uuid4().hex
            )
            self.run_ids = [_gen_id() for _ in range(len(seed))]
            for run_id, run_seed in zip(self.run_ids, seed):
                init_logging(run_id, logging_config, run_seed=int(run_seed))

        else:
            self.run_ids = []

        if on_ids_ready is not None:
            on_ids_ready(self.run_ids)

        train_jit = self.get_make_train()(
            env_args=self.env_args,
            actor_optimizer_args=self.actor_optimizer_args,
            critic_optimizer_args=self.critic_optimizer_args,
            network_args=self.network_args,
            agent_config=self.agent_config,
            total_timesteps=n_timesteps,
            num_episode_test=num_episode_test,
            run_ids=self.run_ids,
            logging_config=logging_config,
            **kwargs,
        )

        if initial_state is None:

            def set_key_and_train(seed, index):
                key = jax.random.PRNGKey(seed)
                return train_jit(key, index)

            index = jnp.arange(len(seed))
            seed = jnp.array(seed)
            _t0 = time.time()
            result = jax.vmap(set_key_and_train, in_axes=0)(seed, index)
        else:
            # Resume path. ``agent.train()`` returns a 2-tuple ``(state, metrics)``;
            # accept either form of ``initial_state`` and strip the metrics if
            # provided so the inner train_jit receives only the agent state.
            if isinstance(initial_state, tuple) and len(initial_state) == 2:
                initial_state = initial_state[0]

            # Only passed when non-zero: the default (0) keeps the call --
            # and the train functions that predate the offset -- unchanged.
            iteration_offset = int(self.resume_iteration_offset(initial_state))
            offset_kwargs = (
                {} if iteration_offset == 0 else {"iteration_offset": iteration_offset}
            )

            def set_key_and_train_resume(seed, index, state, offset_kwargs):
                key = jax.random.PRNGKey(seed)
                return train_jit(
                    key,
                    index,
                    initial_state=state,
                    resume_from_state=True,
                    **offset_kwargs,
                )

            index = jnp.arange(len(seed))
            seed = jnp.array(seed)
            _t0 = time.time()
            # The offset is unbatched (in_axes None): every schedule derived
            # from the iteration index must stay unbatched under the vmap.
            result = jax.vmap(set_key_and_train_resume, in_axes=(0, 0, 0, None))(
                seed,
                index,
                initial_state,
                jax.tree.map(lambda o: jnp.asarray(o, jnp.int32), offset_kwargs),
            )
        # Block until all XLA computation and debug.callbacks complete, then
        # drain and stop the logging worker.  Calling stop_async_logging()
        # inside the vmapped function was wrong: it ran as a Python side effect
        # during vmap *tracing* (before any training executed), killing the
        # worker before any log events were queued.
        jax.block_until_ready(result)
        stop_async_logging()
        return result
