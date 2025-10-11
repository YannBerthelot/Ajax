from functools import partial
from typing import Callable, Optional

# from gymnax import PlaneParams
from target_gym import PlaneParams

from ajax.agents.ASAC.state import ASACConfig
from ajax.agents.ASAC.train_ASAC import make_train
from ajax.agents.base import ActorCritic
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.logging.wandb_logging import (
    LoggingConfig,
    upload_tensorboard_to_wandb,
)
from ajax.state import AlphaConfig
from ajax.types import EnvType


class ASAC(ActorCritic):
    """Average-Reward Soft Actor-Critic (ASAC) from Adamczyk et al. 2025. See https://arxiv.org/abs/2501.09080v2"""

    name: str = "ASAC"

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        n_envs: int = 1,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        actor_architecture=("256", "relu", "256", "relu"),
        critic_architecture=("256", "relu", "256", "relu"),
        env_params: Optional[PlaneParams] = None,
        max_grad_norm: Optional[float] = 0.5,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = int(1e4),
        tau: float = 0.005,
        reward_scale: float = 1.0,
        alpha_init: float = 1.0,
        p_0=20,
        target_entropy_per_dim: float = -1.0,
        lstm_hidden_size: Optional[int] = None,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
    ) -> None:
        """
        Initialize the ASAC agent.

        Args:
            env_id (str | EnvType): Environment ID or environment instance.
            n_envs (int): Number of parallel environments.
            learning_rate (float): Learning rate for optimizers.
            actor_architecture (tuple): Architecture of the actor network.
            critic_architecture (tuple): Architecture of the critic network.
            gamma (float): Discount factor for rewards.
            env_params (Optional[PlaneParams]): Parameters for the environment.
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
        self.config.update({"algo_name": "ASAC"})

        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            lstm_hidden_size=lstm_hidden_size,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
        )

        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate,
            alpha_init=alpha_init,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("ASAC only supports continuous action spaces.")
        action_dim = get_action_dim(self.env_args.env, env_params)
        target_entropy = target_entropy_per_dim * action_dim
        self.agent_config = ASACConfig(
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
            reward_scale=reward_scale,
            p_0=p_0,
        )

        self.buffer = get_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            n_envs=n_envs,
        )

    def get_make_train(self) -> Callable:
        """
        Create a training function for the APO agent.

        Returns:
            Callable: A function that trains the APO agent.
        """
        return partial(make_train, buffer=self.buffer, alpha_args=self.alpha_args)


if __name__ == "__main__":
    n_seeds = 25
    log_frequency = 5_000
    logging_config = LoggingConfig(
        project_name="ASAC_benchmark",
        run_name="baseline",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=True,
        use_wandb=True,
    )
    env_id = "ant"
    ASAC_agent = ASAC(
        env_id=env_id, learning_starts=int(1e4), n_envs=1, alpha_init=1 / 5
    )
    ASAC_agent.train(
        seed=list(range(n_seeds)),
        n_timesteps=int(1e6),
        logging_config=logging_config,
    )
    upload_tensorboard_to_wandb(ASAC_agent.run_ids, logging_config, use_wandb=True)
