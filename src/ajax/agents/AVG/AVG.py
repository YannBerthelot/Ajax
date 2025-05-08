from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
import wandb
from gymnax import EnvParams

from ajax.agents.AVG.state import AVGConfig
from ajax.agents.AVG.train_AVG import make_train
from ajax.environments.create import prepare_env
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.logging.wandb_logging import (
    LoggingConfig,
    init_logging,
    stop_async_logging,
    with_wandb_silent,
)
from ajax.state import AlphaConfig, EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import EnvType


class AVG:
    """Action Value Gradient (AVG)"""

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        num_envs: int = 1,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 8.7e-4,
        alpha_learning_rate: float = 3e-4,
        actor_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        critic_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        gamma: float = 0.95,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 0.5,
        learning_starts: int = int(1e4),
        reward_scale: float = 1.0,
        alpha_init: float = 0.05,
        target_entropy_per_dim: float = -1.0,
        lstm_hidden_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the AVG agent.

        Args:
            env_id (str | EnvType): Environment ID or environment instance.
            num_envs (int): Number of parallel environments.
            learning_rate (float): Learning rate for optimizers.
            actor_architecture (tuple): Architecture of the actor network.
            critic_architecture (tuple): Architecture of the critic network.
            gamma (float): Discount factor for rewards.
            env_params (Optional[EnvParams]): Parameters for the environment.
            max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
            learning_starts (int): Timesteps before training starts.
            reward_scale (float): Scaling factor for rewards.
            alpha_init (float): Initial value for the temperature parameter.
            target_entropy_per_dim (float): Target entropy per action dimension.
            lstm_hidden_size (Optional[int]): Hidden size for LSTM (if used).
        """
        self.config = {**locals()}
        self.config.update({"algo_name": "AVG"})
        env, env_params, env_id, continuous = prepare_env(
            env_id,
            env_params=env_params,
            normalize_obs=True,
            normalize_reward=False,
            num_envs=num_envs,
            gamma=gamma,
        )

        if not check_if_environment_has_continuous_actions(env):
            raise ValueError("AVG only supports continuous action spaces.")

        self.env_args = EnvironmentConfig(
            env=env,
            env_params=env_params,
            num_envs=num_envs,
            continuous=continuous,
        )

        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate,
            alpha_init=alpha_init,
        )

        self.network_args = NetworkConfig(
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            lstm_hidden_size=lstm_hidden_size,
            squash=True,
            penultimate_normalization=True,
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
        action_dim = get_action_dim(env, env_params)
        target_entropy = target_entropy_per_dim * action_dim
        self.agent_args = AVGConfig(
            gamma=gamma,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
            reward_scale=reward_scale,
        )

    @with_wandb_silent
    def train(
        self,
        seed: int | Sequence[int] = 42,
        num_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Optional[LoggingConfig] = None,
    ) -> None:
        """
        Train the SAC agent.

        Args:
            seed (int | Sequence[int]): Random seed(s) for training.
            num_timesteps (int): Total number of timesteps for training.
            num_episode_test (int): Number of episodes for evaluation during training.
        """
        if isinstance(seed, int):
            seed = [seed]

        if logging_config is not None:
            logging_config.config.update(self.config)
            run_ids = [wandb.util.generate_id() for _ in range(len(seed))]
            for index, run_id in enumerate(run_ids):
                init_logging(run_id, index, logging_config)
        else:
            run_ids = None

        def set_key_and_train(seed, index):
            key = jax.random.PRNGKey(seed)

            train_jit = make_train(
                env_args=self.env_args,
                actor_optimizer_args=self.actor_optimizer_args,
                critic_optimizer_args=self.critic_optimizer_args,
                network_args=self.network_args,
                agent_args=self.agent_args,
                total_timesteps=num_timesteps,
                alpha_args=self.alpha_args,
                num_episode_test=num_episode_test,
                run_ids=run_ids,
                logging_config=logging_config,
            )

            agent_state = train_jit(key, index)
            stop_async_logging()
            return agent_state

        index = jnp.arange(len(seed))
        seed = jnp.array(seed)
        jax.vmap(set_key_and_train, in_axes=0)(seed, index)


if __name__ == "__main__":
    n_seeds = 1
    log_frequency = 5_000
    chunk_size = 1000
    num_envs = 1
    logging_config = LoggingConfig(
        "AVG_tests",
        "test",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
            "chunk_size": chunk_size,
        },
        log_frequency=int(log_frequency / num_envs),
        chunk_size=int(chunk_size / num_envs),
        horizon=10_000,
        use_tensorboard=True,
    )
    env_id = "halfcheetah"
    sac_agent = AVG(env_id=env_id, learning_starts=int(1e4), num_envs=num_envs)
    sac_agent.train(
        seed=list(range(n_seeds)),
        num_timesteps=int(1e7) * num_envs,
        logging_config=logging_config,
    )
