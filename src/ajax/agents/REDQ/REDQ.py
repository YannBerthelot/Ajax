from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.cloning import CloningConfig
from ajax.agents.recurrent import make_replay_buffer
from ajax.agents.REDQ.state import REDQConfig
from ajax.agents.REDQ.train_REDQ import make_train
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.extensions.base import Extension
from ajax.modules.pid_actor import PIDActorConfig
from ajax.networks.memory import MemoryConfig
from ajax.state import AlphaConfig
from ajax.types import EnvType


class REDQ(ActorCritic):
    """Soft Actor-Critic (REDQ) agent for training and testing in continuous action spaces."""

    name: str = "REDQ"
    supports_memory: bool = True

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
        max_grad_norm: Optional[float] = None,
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
        repulsion_coef: float = 0.0,
        # Pluggable memory block (see ajax.networks.memory); trains on
        # replayed sequences with R2D2-style burn-in (see ajax.agents.recurrent).
        memory: Optional[Union[MemoryConfig, dict]] = None,
        burn_in: int = 8,
        sequence_length: int = 16,
        # R2D2 stored-state replay: initialize replayed sequences from the
        # actor carries recorded at collection time instead of zero+burn-in
        # (Kapturowski et al. 2019 show this mitigates state staleness best).
        # Stores flatten_carry(actor carry) per step in the buffer.
        stored_state: bool = False,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        actor_cloning_epochs: int = 10,
        critic_cloning_epochs: int = 10,
        actor_cloning_lr: float = 1e-3,
        critic_cloning_lr: float = 1e-3,
        actor_cloning_batch_size: int = 64,
        critic_cloning_batch_size: int = 64,
        pre_train_n_steps: int = 0,
        # Expert for the cloning pre-training and the eval expert-bias
        # metric; online BC is the ImitationLoss extension.
        expert_policy: Optional[Callable] = None,
        pid_actor_config: Optional[PIDActorConfig] = None,
        # --- New surface: composable research features as Extensions ---
        extensions: Sequence[Extension] = (),
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
            env_params (Optional[EnvParams]): Parameters for the environment.
            max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            learning_starts (int): Timesteps before training starts.
            tau (float): Soft update coefficient for target networks.
            reward_scale (float): Scaling factor for rewards.
            alpha_init (float): Initial value for the temperature parameter.
            target_entropy_per_dim (float): Target entropy per action dimension.
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
            memory=memory,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            squash=True,
            extensions=extensions,
        )

        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate,
            alpha_init=alpha_init,
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
            repulsion_coef=repulsion_coef,
            burn_in=burn_in,
            sequence_length=sequence_length,
            stored_state=stored_state,
        )
        self.buffer = make_replay_buffer(
            self.agent_config, n_envs, self.network_args.memory, buffer_size, batch_size
        )

        self.cloning_config = CloningConfig(
            actor_epochs=actor_cloning_epochs,
            critic_epochs=critic_cloning_epochs,
            actor_lr=actor_cloning_lr,
            critic_lr=critic_cloning_lr,
            actor_batch_size=actor_cloning_batch_size,
            critic_batch_size=critic_cloning_batch_size,
            pre_train_n_steps=pre_train_n_steps,
        )
        self.expert_policy = expert_policy
        self.pid_actor_config = pid_actor_config

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
            cloning_args=self.cloning_config,
            expert_policy=self.expert_policy,
            pid_actor_config=self.pid_actor_config,
            extensions=tuple(self.extension_stack.extensions),
        )
