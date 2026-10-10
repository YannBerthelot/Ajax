from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.cloning import CloningConfig
from ajax.agents.recurrent import make_replay_buffer
from ajax.agents.TD3.state import TD3Config
from ajax.agents.TD3.train_TD3 import make_train
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import Extension
from ajax.modules.pid_actor import PIDActorConfig
from ajax.networks.memory import MemoryConfig
from ajax.types import EnvType


class TD3(ActorCritic):
    """Twin Delayed DDPG (Fujimoto et al., 2018) for continuous action spaces.

    Defaults match the original paper (Table 1 / Algorithm 1):
      - Actor / critic: 400-300 ReLU, plain MLPs (no normalisation layer)
      - Adam, lr=1e-3
      - tau=0.005, gamma=0.99
      - Exploration noise: N(0, 0.1)
      - Target policy noise: N(0, 0.2), clipped to +/-0.5
      - Policy delay: 2
      - Replay buffer: 1e6, batch=100, learning_starts=1e4
    """

    name: str = "TD3"
    supports_memory: bool = True

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_architecture=("400", "relu", "300", "relu"),
        critic_architecture=("400", "relu", "300", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        buffer_size: int = int(1e6),
        batch_size: int = 100,
        learning_starts: int = int(1e4),
        tau: float = 0.005,
        reward_scale: float = 1.0,
        num_critics: int = 2,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
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
        # --- Cloning pretraining (mirrors REDQ); online BC is the
        # ImitationLoss extension ---
        actor_cloning_epochs: int = 10,
        actor_cloning_lr: float = 1e-3,
        skip_actor_pretrain: bool = False,
        actor_cloning_batch_size: int = 64,
        pre_train_n_steps: int = 0,
        expert_policy: Optional[Callable] = None,
        # PID actor: actor predicts PID gains instead of raw actions.
        pid_actor_config: Optional[PIDActorConfig] = None,
        # Collection-time action selection override (used downstream).
        action_pipeline: Optional[Callable] = None,
        # --- New surface: composable research features as Extensions ---
        extensions: Sequence[Extension] = (),
    ) -> None:
        self.config = {**locals()}
        self.config.update({"algo_name": "TD3"})

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

        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("TD3 only supports continuous action spaces.")

        self.agent_config = TD3Config(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            reward_scale=reward_scale,
            num_critics=num_critics,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            exploration_noise=exploration_noise,
            burn_in=burn_in,
            sequence_length=sequence_length,
            stored_state=stored_state,
        )
        self.buffer = make_replay_buffer(
            self.agent_config, n_envs, self.network_args.memory, buffer_size, batch_size
        )
        self.cloning_config = CloningConfig(
            actor_epochs=actor_cloning_epochs,
            actor_lr=actor_cloning_lr,
            actor_batch_size=actor_cloning_batch_size,
            pre_train_n_steps=pre_train_n_steps,
            skip_actor_pretrain=skip_actor_pretrain,
        )
        self.expert_policy = expert_policy
        self.pid_actor_config = pid_actor_config
        self.action_pipeline = action_pipeline

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            buffer=self.buffer,
            cloning_args=self.cloning_config,
            expert_policy=self.expert_policy,
            pid_actor_config=self.pid_actor_config,
            action_pipeline=self.action_pipeline,
            extensions=tuple(self.extension_stack.extensions),
        )
