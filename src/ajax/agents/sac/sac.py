from functools import partial
from typing import Callable, Optional

# from gymnax import PlaneParams
from target_gym import PlaneParams

from ajax.agents.base import ActorCritic
from ajax.agents.SAC.state import SACConfig
from ajax.agents.SAC.train_SAC import make_train
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


class SAC(ActorCritic):
    """Soft Actor-Critic (SAC) agent for training and testing in continuous action spaces."""

    name: str = "SAC"

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
        lstm_hidden_size: Optional[int] = None,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
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
        self.config.update({"algo_name": "SAC"})

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
            raise ValueError("SAC only supports continuous action spaces.")
        action_dim = get_action_dim(self.env_args.env, env_params)
        target_entropy = target_entropy_per_dim * action_dim
        self.agent_config = SACConfig(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
            reward_scale=reward_scale,
        )

        self.buffer = get_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            n_envs=n_envs,
        )

    # @with_wandb_silent
    # def train(
    #     self,
    #     seed: int | Sequence[int] = 42,
    #     n_timesteps: int = int(1e6),
    #     num_episode_test: int = 10,
    #     logging_config: Optional[LoggingConfig] = None,
    # ) -> None:
    #     """
    #     Train the SAC agent.

    #     Args:
    #         seed (int | Sequence[int]): Random seed(s) for training.
    #         n_timesteps (int): Total number of timesteps for training.
    #         num_episode_test (int): Number of episodes for evaluation during training.
    #     """
    #     if isinstance(seed, int):
    #         seed = [seed]

    #     if logging_config is not None:
    #         logging_config.config.update(make_json_serializable(self.config))
    #         run_ids = [wandb.util.generate_id() for _ in range(len(seed))]
    #         for index, run_id in enumerate(run_ids):
    #             init_logging(run_id, index, logging_config)
    #     else:
    #         run_ids = None
    #     self.run_ids = run_ids

    #     def set_key_and_train(seed, index):
    #         key = jax.random.PRNGKey(seed)

    #         train_jit = make_train(
    #             env_args=self.env_args,
    #             actor_optimizer_args=self.actor_optimizer_args,
    #             critic_optimizer_args=self.critic_optimizer_args,
    #             network_args=self.network_args,
    #             buffer=self.buffer,
    #             agent_config=self.agent_config,
    #             total_timesteps=n_timesteps,
    #             alpha_args=self.alpha_args,
    #             num_episode_test=num_episode_test,
    #             run_ids=run_ids,
    #             logging_config=logging_config,
    #         )

    #         agent_state, out = train_jit(key, index)
    #         # stop_async_logging()
    #         return agent_state, out

    #     index = jnp.arange(len(seed))
    #     seed = jnp.array(seed)
    #     return jax.vmap(set_key_and_train, in_axes=0)(seed, index)

    def get_make_train(self) -> Callable:
        """
        Create a training function for the APO agent.

        Returns:
            Callable: A function that trains the APO agent.
        """
        return partial(make_train, buffer=self.buffer, alpha_args=self.alpha_args)


if __name__ == "__main__":
    # def main():
    #     n_seeds = 1
    #     log_frequency = 1000
    #     logging_config = LoggingConfig(
    #         project_name="dyna_SAC_tests_sweep",
    #         run_name="SAC",
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
    #         SAC_agent = SAC(env_id=env_id, **config)
    #         _, score = SAC_agent.train(
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
        project_name="test_SAC",
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

    # target_altitude = 5000
    # env = Airplane2D()
    # # env_params = PlaneParams(target_velocity_range=(120, 120))
    # env_params = PlaneParams(
    #     target_altitude_range=(target_altitude, target_altitude),
    #     # initial_altitude_range=(target_altitude, target_altitude),
    # )
    env_id = "ant"
    SAC_agent = SAC(env_id=env_id, learning_starts=int(1e4), n_envs=1)
    SAC_agent.train(
        seed=list(range(n_seeds)),
        n_timesteps=int(1e6),
        logging_config=logging_config,
    )
    # upload_tensorboard_to_wandb(SAC_agent.run_ids, logging_config, use_wandb=True)
