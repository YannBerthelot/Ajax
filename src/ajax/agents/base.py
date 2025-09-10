from collections.abc import Sequence
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import wandb
from gymnax import EnvParams

from ajax.environments.create import prepare_env
from ajax.logging.wandb_logging import (
    LoggingConfig,
    init_logging,
    stop_async_logging,
)
from ajax.state import (
    BaseAgentConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import EnvType, InitializationFunction


class ActorCritic:
    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        n_envs: int = 4,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_architecture=("128", "tanh", "128", "tanh"),
        critic_architecture=("128", "tanh", "128", "tanh"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 0.5,
        lstm_hidden_size: Optional[int] = None,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        actor_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
        critic_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_bias_init: Optional[Union[str, InitializationFunction]] = None,
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
            lstm_hidden_size (Optional[int]): Hidden size for LSTM (if used).
        """

        env, env_params, env_id, continuous = prepare_env(
            env_id,
            env_params=env_params,
            normalize_obs=normalize_observations,
            normalize_reward=normalize_rewards,
            n_envs=n_envs,
            gamma=gamma,
        )

        self.env_args = EnvironmentConfig(
            env=env,
            env_params=env_params,
            n_envs=n_envs,
            continuous=continuous,
        )

        self.network_args = NetworkConfig(
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            lstm_hidden_size=lstm_hidden_size,
            actor_kernel_init=actor_kernel_init,
            actor_bias_init=actor_bias_init,
            critic_kernel_init=critic_kernel_init,
            critic_bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            encoder_bias_init=encoder_bias_init,
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

        self.agent_config = BaseAgentConfig(
            gamma=gamma,
        )

    def get_make_train(self) -> Callable:
        raise NotImplementedError

    # @with_wandb_silent
    def train(
        self,
        seed: int | Sequence[int] = 42,
        n_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Optional[LoggingConfig] = None,
        **kwargs,
    ) -> None:
        """
        Train the PPO agent.

        Args:
            seed (int | Sequence[int]): Random seed(s) for training.
            n_timesteps (int): Total number of timesteps for training.
            num_episode_test (int): Number of episodes for evaluation during training.
        """
        if isinstance(seed, int):
            seed = [seed]

        if logging_config is not None:
            logging_config.config.update(self.config)
            self.run_ids = [wandb.util.generate_id() for _ in range(len(seed))]
            for index, run_id in enumerate(self.run_ids):
                init_logging(run_id, index, logging_config)
        else:
            self.run_ids = []

        def set_key_and_train(seed, index):
            key = jax.random.PRNGKey(seed)

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

            agent_state = train_jit(key, index)
            stop_async_logging()
            return agent_state

        index = jnp.arange(len(seed))
        seed = jnp.array(seed)
        jax.vmap(set_key_and_train, in_axes=0)(seed, index)
