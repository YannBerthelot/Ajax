from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
import wandb
from gymnax import EnvParams

from ajax.agents.DynaSAC.state import AVGConfig, DynaSACConfig, SACConfig
from ajax.agents.DynaSAC.train_dyna_sac import make_train
from ajax.buffers.utils import get_buffer
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


class DynaSAC:
    """DynaSAC agent: combines Dyna and SAC algorithms for training and testing in continuous action spaces. Uses AVG for imagination part"""

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        n_envs: int = 1,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        actor_architecture=("256", "relu", "256", "relu"),
        critic_architecture=("256", "relu", "256", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 0.5,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = int(1e4),
        tau: float = 0.005,
        reward_scale: float = 1.0,
        alpha_init: float = 1.0,  # FIXME: check value
        target_entropy_per_dim: float = -1.0,
        lstm_hidden_size: Optional[int] = None,
        avg_length: int = 4,
        sac_length: int = 4,
    ) -> None:
        """
        Initialize the SAC agent.

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
        self.config = {**locals()}
        self.config.update({"algo_name": "DynaSAC"})
        env, env_params, env_id, continuous = prepare_env(
            env_id,
            env_params=env_params,
            normalize_obs=True,
            normalize_reward=False,
            n_envs=n_envs,
            gamma=gamma,
        )

        if not check_if_environment_has_continuous_actions(env):
            raise ValueError("SAC only supports continuous action spaces.")

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

        self.primary_actor_optimizer_args = OptimizerConfig(
            learning_rate=actor_learning_rate,
            max_grad_norm=max_grad_norm,
            clipped=max_grad_norm is not None,
        )
        self.primary_critic_optimizer_args = OptimizerConfig(
            learning_rate=critic_learning_rate,
            max_grad_norm=max_grad_norm,
            clipped=max_grad_norm is not None,
        )

        self.secondary_actor_optimizer_args = OptimizerConfig(
            learning_rate=6.3e-3,
            max_grad_norm=max_grad_norm,
            clipped=max_grad_norm is not None,
            beta_1=0,
        )
        self.secondary_critic_optimizer_args = OptimizerConfig(
            learning_rate=8.7e-3,
            max_grad_norm=max_grad_norm,
            clipped=max_grad_norm is not None,
            beta_1=0,
        )
        action_dim = get_action_dim(env, env_params)
        target_entropy = target_entropy_per_dim * action_dim
        sac_config = SACConfig(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
            reward_scale=reward_scale,
        )
        avg_config = AVGConfig(
            gamma=gamma,
            target_entropy=target_entropy,
            learning_starts=learning_starts,
            reward_scale=reward_scale,
            num_critics=2,
        )
        self.agent_config = DynaSACConfig(
            primary=sac_config,
            secondary=avg_config,
            avg_length=avg_length,
            sac_length=sac_length,
        )

        self.buffer = get_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            n_envs=n_envs,
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
            for index, run_id in enumerate(run_ids):
                init_logging(run_id, index, logging_config)
        else:
            run_ids = None

        def set_key_and_train(seed, index):
            key = jax.random.PRNGKey(seed)

            train_jit = make_train(
                env_args=self.env_args,
                primary_actor_optimizer_args=self.primary_actor_optimizer_args,
                primary_critic_optimizer_args=self.primary_critic_optimizer_args,
                secondary_actor_optimizer_args=self.secondary_actor_optimizer_args,
                secondary_critic_optimizer_args=self.secondary_critic_optimizer_args,
                network_args=self.network_args,
                buffer=self.buffer,
                agent_config=self.agent_config,
                total_timesteps=n_timesteps,
                alpha_args=self.alpha_args,
                num_episode_test=num_episode_test,
                run_ids=run_ids,
                logging_config=logging_config,
                sac_length=self.agent_config.sac_length,
                avg_length=self.agent_config.avg_length,
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
    logging_config = LoggingConfig(
        "dyna_sac_tests_hector",
        "primary_training_iteration_scan_fn_unroll_4",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=False,
        use_wandb=True,
    )
    env_id = "halfcheetah"
    sac_agent = DynaSAC(
        env_id=env_id,
        learning_starts=int(1e4),
        batch_size=256,
        avg_length=100,
        sac_length=100,
    )
    sac_agent.train(
        seed=list(range(n_seeds)),
        n_timesteps=int(1e6),
        logging_config=logging_config,
    )
