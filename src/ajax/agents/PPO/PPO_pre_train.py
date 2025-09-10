from functools import partial
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
from flax import struct

# from gymnax import EnvParams
from plane_env.env_jax import Airplane2D, EnvParams, EnvState

from ajax.agents.base import ActorCritic
from ajax.agents.PPO.state import PPOConfig
from ajax.agents.PPO.train_PPO_pre_train import CloningConfig, make_train
from ajax.logging.wandb_logging import (
    LoggingConfig,
    upload_tensorboard_to_wandb,
)
from ajax.types import EnvType, InitializationFunction


class PPO(ActorCritic):
    """Soft Actor-Critic (PPO) agent for training and testing in continuous action spaces."""

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
        ent_coef: float = 0,
        clip_range: float = 0.2,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        lstm_hidden_size: Optional[int] = None,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        actor_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
        critic_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_bias_init: Optional[Union[str, InitializationFunction]] = None,
        actor_epochs: int = 10,
        critic_epochs: int = 10,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        actor_batch_size: int = 64,
        critic_batch_size: int = 64,
        pre_train_n_steps: int = int(1e5),
        expert_policy: Optional[Callable] = None,
        imitation_coef: Union[float, Callable[[int], float]] = 1e-3,
        distance_to_stable: Optional[Callable] = None,
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
        self.config = {**locals()}
        self.config.update({"algo_name": "PPO"})

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
            actor_kernel_init=actor_kernel_init,
            actor_bias_init=actor_bias_init,
            critic_kernel_init=critic_kernel_init,
            critic_bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            encoder_bias_init=encoder_bias_init,
        )

        self.agent_config = PPOConfig(
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
        )

        self.cloning_confing = CloningConfig(
            actor_epochs=actor_epochs,
            critic_epochs=critic_epochs,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_batch_size=actor_batch_size,
            critic_batch_size=critic_batch_size,
            pre_train_n_steps=pre_train_n_steps,
            imitation_coef=imitation_coef,
            distance_to_stable=distance_to_stable,
        )
        self.expert_policy = expert_policy

    def get_make_train(self) -> Callable:
        """
        Create a training function for the PPO agent.

        Returns:
            Callable: A function that trains the PPO agent.
        """
        return partial(
            make_train,
            cloning_args=self.cloning_confing,
            expert_policy=self.expert_policy,
        )


@struct.dataclass
class StableState:
    z: float
    z_dot: float


if __name__ == "__main__":
    n_timesteps = int(1e7)
    n_seeds = 5
    log_frequency = 5000
    use_wandb = False
    target_altitude = 5000  # meters

    def expert_policy(x):
        return jnp.array([5, 0])

    def distance_to_stable(
        state: EnvState, stable_state: StableState, modification: Optional[str] = None
    ):
        z = state[..., 2]
        z_dot = state[..., 3]
        if modification is None:
            return jnp.abs(z - stable_state.z) + jnp.abs(z_dot - stable_state.z_dot)
        elif modification == "squared":
            return (
                jnp.abs(z - stable_state.z) + jnp.abs(z_dot - stable_state.z_dot)
            ) ** 2
        elif modification == "sqrt":
            return (
                jnp.abs(z - stable_state.z) + jnp.abs(z_dot - stable_state.z_dot)
            ) ** (1 / 2)

    imitation_coef_list = []
    # imitation_coef_list += ["auto"]
    imitation_coef_list += ["auto_squared", "auto_sqrt"]
    # imitation_coef_list += [
    #     "lin_1.0",
    #     "lin_0.1",
    #     "lin_0.01",
    #     "lin_0.001",
    # ]
    # imitation_coef_list += [
    #     1.0,
    #     1e-1,
    #     1e-2,
    #     1e-3,
    # ]
    # imitation_coef_list += [0.0]
    for pre_train_n_steps in (0, int(1e5)):
        for imitation_coef in imitation_coef_list:
            distance_to_stable = partial(
                distance_to_stable,
                stable_state=StableState(z=target_altitude, z_dot=0),
                modification=(
                    imitation_coef.split("_")[1]
                    if len(imitation_coef.split("_")) > 1
                    else None
                ),
            )
            logging_config = LoggingConfig(
                project_name="plane_pre_train_tests_3",
                run_name="PPO",
                config={
                    "debug": False,
                    "log_frequency": log_frequency,
                    "n_seeds": n_seeds,
                    "pre_train": pre_train_n_steps > 0,
                },
                log_frequency=log_frequency,
                horizon=10_000,
                use_tensorboard=True,
                use_wandb=use_wandb,
            )

            key = jax.random.PRNGKey(42)
            env = Airplane2D()
            env_params = EnvParams(
                target_altitude_range=(target_altitude, target_altitude)
            )

            # TODO : have the constant action be infered from the environment
            action = env.action_space(env_params).sample(key)

            PPO_agent = PPO(
                env_id=env,
                env_params=env_params,  # **init_hyperparams
                normalize_observations=True,
                normalize_rewards=True,
                expert_policy=expert_policy,
                pre_train_n_steps=pre_train_n_steps,
                imitation_coef=imitation_coef,
                distance_to_stable=distance_to_stable,
            )  # Remove version from env_id for brax compatibility
            PPO_agent.train(
                seed=list(range(n_seeds)),
                logging_config=logging_config,
                n_timesteps=n_timesteps,
            )
            upload_tensorboard_to_wandb(PPO_agent.run_ids, logging_config)

    # logging_config = LoggingConfig(
    #     project_name="plane_pre_train_tests",
    #     run_name="PPO",
    #     config={
    #         "debug": False,
    #         "log_frequency": log_frequency,
    #         "n_seeds": n_seeds,
    #         "pre_train": False,
    #     },
    #     log_frequency=log_frequency,
    #     horizon=10_000,
    #     use_tensorboard=True,
    #     use_wandb=use_wandb,
    # )

    # PPO_vanilla_agent = PPO_vanilla(
    #     env_id=env,
    #     env_params=env_params,  # **init_hyperparams
    #     normalize_observations=True,
    #     normalize_rewards=True,
    # )  # Remove version from env_id for brax compatibility
    # PPO_vanilla_agent.train(
    #     seed=list(range(n_seeds)), logging_config=logging_config, n_timesteps=int(1e7)
    # )
    # upload_tensorboard_to_wandb(PPO_vanilla_agent.run_ids, logging_config)
