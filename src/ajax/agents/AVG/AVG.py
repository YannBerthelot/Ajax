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
    """Action Value Gradient (AVG) from Vasan et al. 2024. See https://arxiv.org/abs/2411.15370"""

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        n_envs: int = 1,
        actor_learning_rate: float = 6.3e-3,
        critic_learning_rate: float = 8.7e-3,
        alpha_learning_rate: float = 3e-4,
        actor_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        critic_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        learning_starts: int = 0,
        reward_scale: float = 1.0,
        alpha_init: float = 0.07,
        target_entropy_per_dim: float = -1.0,
        lstm_hidden_size: Optional[int] = None,
        beta_1: float = 0,
        beta_2: float = 0.999,
        num_critics: int = 1,
        batch_size: int = 1,  # only for experimentation, AVG uses batch size of 1
    ) -> None:
        """
        Initialize the AVG agent.

        Args:
            env_id (str | EnvType): Environment ID or environment instance.
            n_envs (int): Number of parallel environments.
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
            n_envs=n_envs,
            gamma=gamma,
        )

        if not check_if_environment_has_continuous_actions(env):
            raise ValueError("AVG only supports continuous action spaces.")

        self.env_args = EnvironmentConfig(
            env=env,
            env_params=env_params,
            n_envs=n_envs,
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
            beta_1=beta_1,
            beta_2=beta_2,
        )
        self.critic_optimizer_args = OptimizerConfig(
            learning_rate=critic_learning_rate,
            max_grad_norm=max_grad_norm,
            clipped=max_grad_norm is not None,
            beta_1=beta_1,
            beta_2=beta_2,
        )
        action_dim = get_action_dim(env, env_params)
        target_entropy = target_entropy_per_dim * action_dim
        self.agent_config = AVGConfig(
            gamma=gamma,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
            reward_scale=reward_scale,
            num_critics=num_critics,
            batch_size=batch_size,
        )

    @with_wandb_silent
    def train(
        self,
        seed: int | Sequence[int] = 42,
        n_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Optional[LoggingConfig] = None,
    ) -> None:
        """
        Train the SAC agent.

        Args:
            seed (int | Sequence[int]): Random seed(s) for training.
            n_timesteps (int): Total number of timesteps for training.
            num_episode_test (int): Number of episodes for evaluation during training.
        """
        if isinstance(seed, int):
            seed = [seed]

        if logging_config is not None:
            logging_config.config.update(self.config)
            run_ids = [wandb.util.generate_id() for _ in range(len(seed))]
            for run_id in run_ids:
                init_logging(run_id, logging_config)
        else:
            run_ids = None

        def set_key_and_train(seed, index):
            key = jax.random.PRNGKey(seed)

            train_jit = make_train(
                env_args=self.env_args,
                actor_optimizer_args=self.actor_optimizer_args,
                critic_optimizer_args=self.critic_optimizer_args,
                network_args=self.network_args,
                agent_config=self.agent_config,
                total_timesteps=n_timesteps,
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
    log_frequency = 5000
    n_envs = 100
    batch_size = 1
    logging_config = LoggingConfig(
        project_name="avg_multi_step_multi_env_benchmark_debug",
        run_name="test",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
            "num_envs": n_envs,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=False,
        use_wandb=True,
    )
    env_id = "hopper"
    avg_agent = AVG(
        env_id=env_id,
        n_envs=n_envs,
        batch_size=batch_size,
        actor_learning_rate=6.3e-4,
        critic_learning_rate=8.7e-4,
        alpha_init=0.07,
    )
    avg_agent.train(
        seed=list(range(n_seeds)),
        n_timesteps=int(2e6),
        logging_config=logging_config,
    )
