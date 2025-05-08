from collections.abc import Sequence
from typing import Optional

import jax
import jax.numpy as jnp
import wandb
from gymnax import EnvParams

from ajax.agents.sac.state import SACConfig
from ajax.agents.sac.train_sac import make_train
from ajax.buffers.utils import get_buffer
from ajax.environments.create import prepare_env
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.logging.wandb_logging import (
    LoggingConfig,
    stop_async_logging,
    with_wandb_silent,
)
from ajax.state import AlphaConfig, EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import EnvType


class SAC:
    """Soft Actor-Critic (SAC) agent for training and testing in continuous action spaces."""

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        num_envs: int = 1,
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
    ) -> None:
        """
        Initialize the SAC agent.

        Args:
            env_id (str | EnvType): Environment ID or environment instance.
            num_envs (int): Number of parallel environments.
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
        self.config.update({"algo_name": "SAC"})
        env, env_params, env_id, continuous = prepare_env(
            env_id,
            env_params=env_params,
            normalize_obs=False,
            normalize_reward=False,
            num_envs=num_envs,
            gamma=gamma,
        )

        if not check_if_environment_has_continuous_actions(env):
            raise ValueError("SAC only supports continuous action spaces.")

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
        self.agent_args = SACConfig(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
            reward_scale=reward_scale,
        )

        self.buffer = get_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            num_envs=num_envs,
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
                wandb.init(
                    project=logging_config.project_name,
                    name=f"{logging_config.run_name}  {index}",
                    id=run_id,
                    resume="never",
                    reinit=True,
                    config=logging_config.config,
                )
        else:
            run_ids = None

        def set_key_and_train(seed, index):
            key = jax.random.PRNGKey(seed)

            train_jit = make_train(
                env_args=self.env_args,
                actor_optimizer_args=self.actor_optimizer_args,
                critic_optimizer_args=self.critic_optimizer_args,
                network_args=self.network_args,
                buffer=self.buffer,
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
    logging_config = LoggingConfig(
        "match_SAC_reproducibility_truncated",
        "test",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
            "chunk_size": chunk_size,
        },
        log_frequency=log_frequency,
        chunk_size=chunk_size,
        horizon=10_000,
    )
    env_id = "halfcheetah"
    sac_agent = SAC(env_id=env_id, learning_starts=int(1e4), batch_size=256)
    sac_agent.train(
        seed=list(range(n_seeds)),
        num_timesteps=int(1e6),
        logging_config=logging_config,
    )
