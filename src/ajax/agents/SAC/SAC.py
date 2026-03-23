from functools import partial
from typing import Callable, Optional, Union

from target_gym import PlaneParams

from ajax.agents.base import ActorCritic
from ajax.agents.PPO.train_PPO_pre_train import CloningConfig
from ajax.agents.SAC.state import SACConfig
from ajax.agents.SAC.train_SAC import make_train
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import AlphaConfig, NetworkConfig
from ajax.types import EnvType


class SAC(ActorCritic):
    """Soft Actor-Critic agent for continuous action spaces."""

    name: str = "SAC"

    def __init__(
        self,
        env_id: str | EnvType,
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
        alpha_init: float = 1.0,
        target_entropy_per_dim: float = -1.0,
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
        eval_expert_policy: Optional[Callable] = None,  # for eval logging only
        imitation_coef: Union[float, Callable[[int], float]] = 0.0,
        distance_to_stable: Optional[Callable] = None,
        imitation_coef_offset: float = 0.0,
        action_scale: float = 1.0,
        early_termination_condition: Optional[Callable] = None,
        residual: bool = False,
        fixed_alpha: bool = False,
        num_critics: int = 4,
        # --- Expert guidance (all disabled by default) ---
        use_expert_guidance: bool = False,
        num_critic_updates: int = 1,
        expert_buffer_n_steps: int = 20_000,
        expert_mix_fraction: float = 0.1,
        # AWBC proximity decay (proximity_scale=None = no decay)
        box_threshold: float = 500.0,
        proximity_scale: Optional[float] = None,
        altitude_obs_idx: int = 1,
        target_obs_idx: int = 6,
        # MC critic pretraining (unbiased, replaces Bellman pretraining)
        use_mc_critic_pretrain: bool = False,
        mc_pretrain_n_mc_steps: int = 10_000,
        mc_pretrain_n_mc_episodes: int = 100,
        mc_pretrain_n_steps: int = 5_000,
        # Value constraint: Q_π >= Q_expert (action-agnostic floor)
        value_constraint_coef: float = 0.0,
        # Obs augmentation: append a_expert(target) to obs
        augment_obs_with_expert_action: bool = False,
    ) -> None:
        self.config = {**locals()}
        self.config.update({"algo_name": "SAC"})

        super().__init__(
            env_id=env_id, n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            env_params=env_params, max_grad_norm=max_grad_norm,
            lstm_hidden_size=lstm_hidden_size,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
        )
        self.alpha_args = AlphaConfig(learning_rate=alpha_learning_rate, alpha_init=alpha_init)
        self.network_args = NetworkConfig(
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            lstm_hidden_size=lstm_hidden_size,
            squash=True, penultimate_normalization=False,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("SAC only supports continuous action spaces.")

        action_dim = get_action_dim(self.env_args.env, env_params)
        self.agent_config = SACConfig(
            gamma=gamma, tau=tau, learning_starts=learning_starts,
            target_entropy=target_entropy_per_dim * action_dim,
            reward_scale=reward_scale,
        )
        self.buffer = get_buffer(buffer_size=buffer_size, batch_size=batch_size, n_envs=n_envs)
        self.num_critics = num_critics
        self.cloning_confing = CloningConfig(
            actor_epochs=actor_cloning_epochs, critic_epochs=critic_cloning_epochs,
            actor_lr=actor_cloning_lr, critic_lr=critic_cloning_lr,
            actor_batch_size=actor_cloning_batch_size,
            critic_batch_size=critic_cloning_batch_size,
            pre_train_n_steps=pre_train_n_steps,
            imitation_coef=imitation_coef, distance_to_stable=distance_to_stable,
            imitation_coef_offset=imitation_coef_offset, action_scale=action_scale,
        )
        self.expert_policy = expert_policy
        self.eval_expert_policy = eval_expert_policy
        self.early_termination_condition = early_termination_condition
        self.residual = residual
        self.fixed_alpha = fixed_alpha
        self.use_expert_guidance = use_expert_guidance
        self.num_critic_updates = num_critic_updates
        self.expert_buffer_n_steps = expert_buffer_n_steps
        self.expert_mix_fraction = expert_mix_fraction
        self.box_threshold = box_threshold
        self.proximity_scale = proximity_scale
        self.altitude_obs_idx = altitude_obs_idx
        self.target_obs_idx = target_obs_idx
        self.use_mc_critic_pretrain = use_mc_critic_pretrain
        self.mc_pretrain_n_mc_steps = mc_pretrain_n_mc_steps
        self.mc_pretrain_n_mc_episodes = mc_pretrain_n_mc_episodes
        self.mc_pretrain_n_steps = mc_pretrain_n_steps
        self.value_constraint_coef = value_constraint_coef
        self.augment_obs_with_expert_action = augment_obs_with_expert_action

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            buffer=self.buffer, alpha_args=self.alpha_args, cloning_args=self.cloning_confing,
            expert_policy=self.expert_policy, eval_expert_policy=self.eval_expert_policy,
            early_termination_condition=self.early_termination_condition,
            residual=self.residual, fixed_alpha=self.fixed_alpha,
            num_critics=self.num_critics,
            use_expert_guidance=self.use_expert_guidance,
            num_critic_updates=self.num_critic_updates,
            expert_buffer_n_steps=self.expert_buffer_n_steps,
            expert_mix_fraction=self.expert_mix_fraction,
            box_threshold=self.box_threshold, proximity_scale=self.proximity_scale,
            altitude_obs_idx=self.altitude_obs_idx, target_obs_idx=self.target_obs_idx,
            use_mc_critic_pretrain=self.use_mc_critic_pretrain,
            mc_pretrain_n_mc_steps=self.mc_pretrain_n_mc_steps,
            mc_pretrain_n_mc_episodes=self.mc_pretrain_n_mc_episodes,
            mc_pretrain_n_steps=self.mc_pretrain_n_steps,
            value_constraint_coef=self.value_constraint_coef,
            augment_obs_with_expert_action=self.augment_obs_with_expert_action,
        )