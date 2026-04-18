from functools import partial
from typing import Callable, Optional, Union

from target_gym import PlaneParams

from ajax.agents.base import ActorCritic
from ajax.agents.cloning import CloningConfig
from ajax.agents.SAC.state import SACConfig
from ajax.agents.SAC.train_SAC import make_train
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.modules.pid_actor import PIDActorConfig
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
        # PID policy: execute expert action directly (no actor used for env interaction)
        use_pid_policy: bool = False,
        fixed_alpha: bool = False,
        num_critics: int = 2,
        # --- Expert guidance (all disabled by default) ---
        use_expert_guidance: bool = False,
        num_critic_updates: int = 1,
        expert_buffer_n_steps: int = 20_000,
        expert_mix_fraction: float = 0.1,
        box_threshold: float = 500.0,
        altitude_obs_idx: int = 1,
        target_obs_idx: int = 6,
        # MC critic pretraining (unbiased, replaces Bellman pretraining)
        use_mc_critic_pretrain: bool = False,
        mc_pretrain_n_mc_steps: int = 10_000,
        mc_pretrain_n_mc_episodes: int = 100,
        mc_pretrain_n_steps: int = 5_000,
        # Online critic light pre-regression (requires MC critic pretrain)
        use_online_critic_light_pretrain: bool = True,
        online_critic_pretrain_steps: int = 500,
        online_critic_pretrain_lr_scale: float = 0.1,
        # Obs augmentation: append a_expert(target) to obs
        augment_obs_with_expert_action: bool = False,
        # Bellman critic pretraining (legacy fallback, mutually exclusive with MC)
        use_bellman_critic_pretrain: bool = False,
        detach_obs_aug_action: bool = False,
        # Train-fraction conditioning: append timestep/total_timesteps to obs
        use_train_frac: bool = False,
        # Update start thresholds
        policy_update_start: int = 2_000,
        alpha_update_start: int = 2_000,
        # Potential-based reward shaping
        use_critic_blend: bool = False,
        critic_warmup_frac: float = 0.15,
        # Value-threshold box (v_min/v_max inferred from MC pretraining)
        use_box: bool = False,
        # Online decaying BC term (active for all MC pretrain runs unless disabled)
        use_online_bc: bool = True,
        bc_coef: float = 1.0,
        # EDGE (Expert Decayed Guided Exploration)
        use_expert_guided_exploration: bool = False,
        exploration_decay_frac: float = 0.30,
        exploration_tau: float = 1.0,
        exploration_boltzmann: bool = False,
        fixed_exploration_prob: float = 0.5,
        exploration_argmax: bool = False,
        # IBRL bootstrap: max(Q_policy, Q_expert) TD target consistent with argmax policy
        ibrl_bootstrap: bool = False,
        # Distance-modulated entropy target (None = disabled)
        target_entropy_far: Optional[float] = None,
        # Online MC correction for high-variance states (None = disabled)
        mc_variance_threshold: Optional[float] = None,
        # Periodic self-consistent φ* refresh via expert-flagged buffer transitions
        use_phi_refresh: bool = False,
        phi_refresh_interval: int = 500,
        phi_refresh_steps: int = 20,
        # PID actor: actor network predicts PID gains instead of raw actions.
        pid_actor_config: Optional[PIDActorConfig] = None,
        # --- Composable hook overrides (None = build from flags above) ---
        action_pipeline: Optional[Callable] = None,
        target_modifier: Optional[Callable] = None,
        obs_preprocessor: Optional[Callable] = None,
        policy_action_transform: Optional[Callable] = None,
        eval_action_transform: Optional[Callable] = None,
        runtime_maintenance: Optional[Callable] = None,
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
            learning_rate=alpha_learning_rate, alpha_init=alpha_init
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
        self.agent_config = SACConfig(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy_per_dim * action_dim,
            reward_scale=reward_scale,
        )
        self.buffer = get_buffer(
            buffer_size=buffer_size, batch_size=batch_size, n_envs=n_envs
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
        self.use_pid_policy = use_pid_policy
        self.fixed_alpha = fixed_alpha
        self.use_expert_guidance = use_expert_guidance
        self.num_critic_updates = num_critic_updates
        self.expert_buffer_n_steps = expert_buffer_n_steps
        self.expert_mix_fraction = expert_mix_fraction
        self.box_threshold = box_threshold
        self.altitude_obs_idx = altitude_obs_idx
        self.target_obs_idx = target_obs_idx
        self.use_mc_critic_pretrain = use_mc_critic_pretrain
        self.mc_pretrain_n_mc_steps = mc_pretrain_n_mc_steps
        self.mc_pretrain_n_mc_episodes = mc_pretrain_n_mc_episodes
        self.mc_pretrain_n_steps = mc_pretrain_n_steps
        self.use_online_critic_light_pretrain = use_online_critic_light_pretrain
        self.online_critic_pretrain_steps = online_critic_pretrain_steps
        self.online_critic_pretrain_lr_scale = online_critic_pretrain_lr_scale
        self.augment_obs_with_expert_action = augment_obs_with_expert_action
        self.use_bellman_critic_pretrain = use_bellman_critic_pretrain
        self.detach_obs_aug_action = detach_obs_aug_action
        self.use_train_frac = use_train_frac
        self.policy_update_start = policy_update_start
        self.alpha_update_start = alpha_update_start
        self.use_critic_blend = use_critic_blend
        self.critic_warmup_frac = critic_warmup_frac
        self.use_box = use_box
        self.use_online_bc = use_online_bc
        self.bc_coef = bc_coef
        self.use_expert_guided_exploration = use_expert_guided_exploration
        self.exploration_decay_frac = exploration_decay_frac
        self.exploration_tau = exploration_tau
        self.exploration_boltzmann = exploration_boltzmann
        self.fixed_exploration_prob = fixed_exploration_prob
        self.exploration_argmax = exploration_argmax
        self.ibrl_bootstrap = ibrl_bootstrap
        self.target_entropy_far = target_entropy_far
        self.mc_variance_threshold = mc_variance_threshold
        self.use_phi_refresh = use_phi_refresh
        self.phi_refresh_interval = phi_refresh_interval
        self.phi_refresh_steps = phi_refresh_steps
        self.pid_actor_config = pid_actor_config
        self.action_pipeline = action_pipeline
        self.target_modifier = target_modifier
        self.obs_preprocessor = obs_preprocessor
        self.policy_action_transform = policy_action_transform
        self.eval_action_transform = eval_action_transform
        self.runtime_maintenance = runtime_maintenance

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            buffer=self.buffer,
            alpha_args=self.alpha_args,
            cloning_args=self.cloning_confing,
            expert_policy=self.expert_policy,
            eval_expert_policy=self.eval_expert_policy,
            early_termination_condition=self.early_termination_condition,
            residual=self.residual,
            use_pid_policy=self.use_pid_policy,
            fixed_alpha=self.fixed_alpha,
            num_critics=self.num_critics,
            use_expert_guidance=self.use_expert_guidance,
            num_critic_updates=self.num_critic_updates,
            expert_buffer_n_steps=self.expert_buffer_n_steps,
            expert_mix_fraction=self.expert_mix_fraction,
            box_threshold=self.box_threshold,
            altitude_obs_idx=self.altitude_obs_idx,
            target_obs_idx=self.target_obs_idx,
            use_mc_critic_pretrain=self.use_mc_critic_pretrain,
            mc_pretrain_n_mc_steps=self.mc_pretrain_n_mc_steps,
            mc_pretrain_n_mc_episodes=self.mc_pretrain_n_mc_episodes,
            mc_pretrain_n_steps=self.mc_pretrain_n_steps,
            use_online_critic_light_pretrain=self.use_online_critic_light_pretrain,
            online_critic_pretrain_steps=self.online_critic_pretrain_steps,
            online_critic_pretrain_lr_scale=self.online_critic_pretrain_lr_scale,
            augment_obs_with_expert_action=self.augment_obs_with_expert_action,
            use_bellman_critic_pretrain=self.use_bellman_critic_pretrain,
            detach_obs_aug_action=self.detach_obs_aug_action,
            use_train_frac=self.use_train_frac,
            policy_update_start=self.policy_update_start,
            alpha_update_start=self.alpha_update_start,
            use_critic_blend=self.use_critic_blend,
            critic_warmup_frac=self.critic_warmup_frac,
            use_box=self.use_box,
            use_online_bc=self.use_online_bc,
            bc_coef=self.bc_coef,
            use_expert_guided_exploration=self.use_expert_guided_exploration,
            exploration_decay_frac=self.exploration_decay_frac,
            exploration_tau=self.exploration_tau,
            exploration_boltzmann=self.exploration_boltzmann,
            fixed_exploration_prob=self.fixed_exploration_prob,
            exploration_argmax=self.exploration_argmax,
            ibrl_bootstrap=self.ibrl_bootstrap,
            use_residual_rl=self.residual,
            target_entropy_far=self.target_entropy_far,
            mc_variance_threshold=self.mc_variance_threshold,
            use_phi_refresh=self.use_phi_refresh,
            phi_refresh_interval=self.phi_refresh_interval,
            phi_refresh_steps=self.phi_refresh_steps,
            pid_actor_config=self.pid_actor_config,
            action_pipeline=self.action_pipeline,
            target_modifier=self.target_modifier,
            obs_preprocessor=self.obs_preprocessor,
            policy_action_transform=self.policy_action_transform,
            eval_action_transform=self.eval_action_transform,
            runtime_maintenance=self.runtime_maintenance,
        )
