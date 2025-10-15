from functools import partial
from typing import Callable, Optional, Union

# from gymnax import PlaneParams
from target_gym import PlaneParams

from ajax.agents.base import ActorCritic
from ajax.agents.PPO.train_PPO_pre_train import CloningConfig
from ajax.agents.REDQ.state import REDQConfig
from ajax.agents.REDQ.train_REDQ import make_train
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.logging.wandb_logging import (
    LoggingConfig,
)
from ajax.state import AlphaConfig, NetworkConfig
from ajax.types import EnvType


class REDQ(ActorCritic):
    """Soft Actor-Critic (REDQ) agent for training and testing in continuous action spaces."""

    name: str = "REDQ"

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
        env_params: Optional[PlaneParams] = None,
        max_grad_norm: Optional[float] = 0.5,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = int(1e4),
        tau: float = 0.005,
        reward_scale: float = 1.0,
        alpha_init: float = 1.0,  # FIXME: check value
        target_entropy_per_dim: float = -1.0,
        num_critic_updates: int = 20,
        num_critics: int = 10,
        subset_size: int = 2,
        lstm_hidden_size: Optional[int] = None,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        actor_cloning_epochs: int = 10,
        critic_cloning_epochs: int = 10,
        actor_cloning_lr: float = 1e-3,
        critic_cloning_lr: float = 1e-3,
        actor_cloning_batch_size: int = 64,
        critic_cloning_batch_size: int = 64,
        pre_train_n_steps: int = 0,
        expert_policy: Optional[Callable] = None,
        imitation_coef: Union[float, Callable[[int], float]] = 0.0,
        distance_to_stable: Optional[Callable] = None,
        imitation_coef_offset: float = 0.0,
    ) -> None:
        """
        Initialize the REDQ agent.

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
        self.config.update({"algo_name": "REDQ"})

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

        self.network_args = NetworkConfig(
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            lstm_hidden_size=lstm_hidden_size,
            squash=True,
            penultimate_normalization=False,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("REDQ only supports continuous action spaces.")
        action_dim = get_action_dim(self.env_args.env, env_params)
        target_entropy = target_entropy_per_dim * action_dim
        self.agent_config = REDQConfig(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
            reward_scale=reward_scale,
            num_critic_updates=num_critic_updates,
            num_critics=num_critics,
            subset_size=subset_size,
        )

        self.buffer = get_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            n_envs=n_envs,
        )

        self.cloning_confing = CloningConfig(
            actor_epochs=actor_cloning_epochs,
            critic_epochs=critic_cloning_epochs,
            actor_lr=actor_cloning_lr,
            critic_lr=critic_cloning_lr,
            actor_batch_size=actor_cloning_batch_size,
            critic_batch_size=critic_cloning_batch_size,
            pre_train_n_steps=pre_train_n_steps,
            imitation_coef=imitation_coef,
            distance_to_stable=distance_to_stable,
            imitation_coef_offset=imitation_coef_offset,
        )
        self.expert_policy = expert_policy

    def get_make_train(self) -> Callable:
        """
        Create a training function for the APO agent.

        Returns:
            Callable: A function that trains the APO agent.
        """
        return partial(
            make_train,
            buffer=self.buffer,
            alpha_args=self.alpha_args,
            cloning_args=self.cloning_confing,
            expert_policy=self.expert_policy,
        )


if __name__ == "__main__":
    # def main():
    #     n_seeds = 1
    #     log_frequency = 1000
    #     logging_config = LoggingConfig(
    #         project_name="dyna_REDQ_tests_sweep",
    #         run_name="REDQ",
    #         config={
    #             "debug": False,
    #             "log_frequency": log_frequency,
    #             "n_seeds": n_seeds,
    #         },
    #         log_frequency=log_frequency,
    #         horizon=10_000,
    #         use_tensorboard=False,
    #         use_wandb=False,
    #     )
    #     env_id = "halfcheetah"

    #     def init_and_train(config):
    #         REDQ_agent = REDQ(env_id=env_id, **config)
    #         _, score = REDQ_agent.train(
    #             seed=list(range(n_seeds)),
    #             n_timesteps=int(1e4),
    #             logging_config=logging_config,
    #         )
    #         return score

    #     wandb.init(project="my-first-sweep")
    #     score = init_and_train(wandb.config)

    #     wandb.log({"score": score})

    # sweep_configuration = {
    #     "method": "random",
    #     "metric": {"goal": "maximize", "name": "score"},
    #     "parameters": {
    #         "actor_learning_rate": {"max": 0.1, "min": 0.01},
    #         "n_envs": {"values": [1, 3, 7]},
    #     },
    # }
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

    # wandb.agent(sweep_id, function=main, count=10)

    n_seeds = 1
    log_frequency = 5_000
    logging_config = LoggingConfig(
        project_name="REDQ_benchmark",
        run_name="baseline",
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

    env_id = "hopper"
    REDQ_agent = REDQ(env_id=env_id, learning_starts=int(1e4), n_envs=1)
    REDQ_agent.train(
        seed=list(range(n_seeds)),
        n_timesteps=int(1e6),
        logging_config=logging_config,
    )
    # upload_tensorboard_to_wandb(REDQ_agent.run_ids, logging_config, use_wandb=True)
