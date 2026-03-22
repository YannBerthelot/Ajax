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
        # --- Expert guidance ---
        use_expert_guidance: bool = False,
        # --- AWBC training parameters ---
        num_critic_updates: int = 1,
        expert_buffer_n_steps: int = 20_000,
        critic_pretrain_steps: int = 5_000,
        actor_pretrain_steps: int = 0,  # 0 = no BC pre-training
        expert_mix_fraction: float = 0.1,
        # --- Asymmetric AWBC (disabled by default) ---
        # Set use_asymmetric_awbc=True to activate.
        # above_expert_coef: small AWBC constant when policy beats expert,
        #   prevents drift without blocking improvement.
        #   encouraging precision over exploration near the performance ceiling.
        use_asymmetric_awbc: bool = False,
        above_expert_coef: float = 0.01,
        # --- Proximity-weighted AWBC ---
        # box_threshold: distance at which expert takes over — must match trunc_condition.
        #   Single source of truth: changing this automatically updates both the box
        #   boundary and the proximity weight normalization.
        # proximity_scale: decay rate; weight = exp(-dist/box_threshold/scale)
        #   None  → no decay (uniform expert trust everywhere)
        #   0.5   → very local (≈0 beyond 2× box boundary)
        #   1.0   → moderate (0.37 at box boundary, ≈0 at 5× boundary)
        #   2.0   → gentle (still 8% trust at 5× box boundary)
        box_threshold: float = 500.0,
        proximity_scale: Optional[float] = None,
        altitude_obs_idx: int = 1,  # raw_obs index for current altitude
        target_obs_idx: int = 6,  # raw_obs index for target altitude
    ) -> None:
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
        self.num_critics = num_critics
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
            action_scale=action_scale,
        )
        self.expert_policy = expert_policy
        self.eval_expert_policy = eval_expert_policy
        self.early_termination_condition = early_termination_condition
        self.residual = residual
        self.fixed_alpha = fixed_alpha
        self.use_expert_guidance = use_expert_guidance
        self.num_critic_updates = num_critic_updates
        self.expert_buffer_n_steps = expert_buffer_n_steps
        self.critic_pretrain_steps = critic_pretrain_steps
        self.actor_pretrain_steps = actor_pretrain_steps
        self.expert_mix_fraction = expert_mix_fraction

        # When use_asymmetric_awbc=False, disable asymmetry:
        #   above_expert_coef=0.0  → pure SAC when above expert (symmetric)
        self.above_expert_coef = above_expert_coef if use_asymmetric_awbc else 0.0
        self.box_threshold = box_threshold
        self.proximity_scale = proximity_scale
        self.altitude_obs_idx = altitude_obs_idx
        self.target_obs_idx = target_obs_idx

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            buffer=self.buffer,
            alpha_args=self.alpha_args,
            cloning_args=self.cloning_confing,
            # expert_policy: used for training (warmup seeding, AWBC gradient).
            #   None → no expert involvement in training whatsoever.
            expert_policy=self.expert_policy,
            eval_expert_policy=self.eval_expert_policy,
            early_termination_condition=self.early_termination_condition,
            residual=self.residual,
            fixed_alpha=self.fixed_alpha,
            num_critics=self.num_critics,
            use_expert_guidance=self.use_expert_guidance,
            num_critic_updates=self.num_critic_updates,
            expert_buffer_n_steps=self.expert_buffer_n_steps,
            critic_pretrain_steps=self.critic_pretrain_steps,
            actor_pretrain_steps=self.actor_pretrain_steps,
            expert_mix_fraction=self.expert_mix_fraction,
            above_expert_coef=self.above_expert_coef,
            box_threshold=self.box_threshold,
            proximity_scale=self.proximity_scale,
            altitude_obs_idx=self.altitude_obs_idx,
            target_obs_idx=self.target_obs_idx,
        )
