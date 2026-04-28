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
        residual_scale: float = 1.0,
        # True JSRL curriculum (Uchendu et al. 2023)
        jsrl_curriculum: bool = False,
        jsrl_episode_length: int = 1000,
        jsrl_decay_frac: float = 0.5,
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
        # Quality-aware (LCB) gate
        exploration_lcb: bool = False,
        exploration_thompson: bool = False,
        lcb_beta_init: float = 1.0,
        lcb_beta_decay_k: float = 2.0,
        lcb_temperature: float = 1.0,
        # ε-floor on Thompson: minimum probability of picking the policy
        # regardless of the Thompson sample. Required when expert
        # dominates by many σ_critic units (Thompson otherwise never
        # picks the policy → critic Q for policy actions stays stale).
        # Set carefully on brittle envs.
        epsilon_floor: float = 0.0,
        # Augment the obs presented to BOTH actor and critic with the
        # expert's flattened internal state (e.g. PID integrator). This
        # is the missing Markov component when the expert has hidden
        # state: with this on, Q regression on (s, a, expert_state) is
        # near-perfect (RMSE/std ≈ 0.06) vs ~0.3 without. Asymmetric
        # vs `augment_obs_with_expert_action`: this exposes a STATE,
        # not an ACTION, so it doesn't bias the policy toward copying
        # the expert.
        augment_obs_with_expert_state: bool = False,
        # Agent-side running observation normalisation. When True the
        # full augmented obs (env_obs + flatten(expert_state)) is z-scored
        # using running mean/var maintained inside the agent's
        # CollectorState. Stats live on the actor/critic states (synced
        # from the collector each online step) so get_pi / predict_value
        # apply the normalisation transparently to all callers. Default
        # False keeps vanilla SAC unaffected. Required when
        # augment_obs_with_expert_state=True because the appended
        # integrator dims can have std up to ~10^4 and break the encoder
        # without normalisation.
        normalize_obs_running: bool = False,
        # When True, the replay buffer stores the policy's sampled action
        # rather than the action that was actually executed. Off-policy-
        # incorrect when a gate selects the expert (the (s', r) belongs to
        # a_expert but is attributed to a_policy). Used for the
        # ``store_policy_action`` ablation; default False is the correct
        # behaviour.
        store_policy_action: bool = False,
        # When True, the critic's TD target uses the action chosen by the
        # LCB gate at the next state (Q(s', a_lcb_gated')) instead of the
        # standard SAC bootstrap on a sampled policy action. Pairs with
        # ``exploration_lcb=True`` — applies LCB consistently at action
        # selection AND at value bootstrap.
        lcb_gated_bootstrap: bool = False,
        # Warmup action mix: during the learning_starts pre-update phase,
        # actions sent to the env are sampled from
        # (expert with prob expert_fraction, uniform otherwise).
        # Setting expert_fraction=1.0 disables uniform exploration in
        # favour of always rolling out the expert during warmup; this
        # keeps the agent alive on brittle envs and seeds the buffer
        # with on-distribution expert data, complementing the explicit
        # expert_buffer prefill.
        expert_fraction: float = 0.7,
        # target_entropy time ramp. If both are set, the effective target
        # entropy linearly interpolates from
        # `target_entropy_initial_per_dim * action_dim` at t=0 to
        # `target_entropy_per_dim * action_dim` at
        # t = ramp_frac * total_timesteps, and stays at the latter after.
        # Use a low initial target (e.g. -5 per dim) to keep alpha small
        # at start (near-deterministic policy ≈ expert via BC mean), then
        # ramp to standard SAC target as the policy learns where to
        # deviate from the expert.
        target_entropy_initial_per_dim: Optional[float] = None,
        target_entropy_ramp_frac: float = 0.5,
        # Critic MC pretrain controls (route through CloningConfig)
        skip_actor_pretrain: bool = False,
        skip_critic_pretrain: bool = True,
        # If True, BC the actor (mean + log_std) then reset log_std back to
        # its init values — keeps μ ≈ expert without compressing entropy.
        reset_log_std_after_bc: bool = False,
        # If True, BC the actor then reset BOTH mean and log_std heads,
        # keeping only the encoder BC-warmed. Trunk features encode
        # expert-relevant state info; head is fresh → policy starts random
        # without sharp policy break or entropy compression.
        reset_actor_head_after_bc: bool = False,
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
        extra_actor_loss_fn: Optional[Callable] = None,
        extra_critic_loss_fn: Optional[Callable] = None,
        init_transform: Optional[Callable] = None,
        auxiliary_update: Optional[Callable] = None,
        extra_eval_metrics: Optional[Callable] = None,
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

        # Gain-policy mode: actor outputs PID gains (see make_action_pipeline).
        # Critic / buffer / target_entropy then operate in gain-space.
        self.action_dim_override: Optional[int] = None
        if use_pid_policy:
            if expert_policy is None or not hasattr(expert_policy, "learnable_fields"):
                raise ValueError(
                    "use_pid_policy=True requires a FunctionalExpertPolicy with"
                    " learnable_fields (registered via register_learnable_gains)."
                )
            incompat = {
                "residual": residual,
                "use_mc_critic_pretrain": use_mc_critic_pretrain,
                "use_online_bc": use_online_bc,
                "use_bellman_critic_pretrain": use_bellman_critic_pretrain,
                "augment_obs_with_expert_action": augment_obs_with_expert_action,
                "use_critic_blend": use_critic_blend,
                "use_box": use_box,
                "use_expert_guidance": use_expert_guidance,
                "ibrl_bootstrap": ibrl_bootstrap,
                "use_expert_guided_exploration": use_expert_guided_exploration,
            }
            bad = [k for k, v in incompat.items() if v]
            if bad:
                raise ValueError(
                    f"use_pid_policy (gain-mode) is incompatible with: {bad}."
                    " Gain-mode uses a 7-D action space; these flags assume env"
                    " action space."
                )
            self.action_dim_override = len(expert_policy.learnable_fields)

        action_dim = (
            self.action_dim_override
            if self.action_dim_override is not None
            else get_action_dim(self.env_args.env, env_params)
        )
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
            skip_actor_pretrain=skip_actor_pretrain,
            skip_critic_pretrain=skip_critic_pretrain,
            reset_log_std_after_bc=reset_log_std_after_bc,
            reset_actor_head_after_bc=reset_actor_head_after_bc,
        )
        self.expert_policy = expert_policy
        self.eval_expert_policy = eval_expert_policy
        self.early_termination_condition = early_termination_condition
        self.residual = residual
        self.residual_scale = residual_scale
        self.jsrl_curriculum = jsrl_curriculum
        self.jsrl_episode_length = jsrl_episode_length
        self.jsrl_decay_frac = jsrl_decay_frac
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
        self.exploration_lcb = exploration_lcb
        self.exploration_thompson = exploration_thompson
        self.lcb_beta_init = lcb_beta_init
        self.lcb_beta_decay_k = lcb_beta_decay_k
        self.lcb_temperature = lcb_temperature
        self.expert_fraction = expert_fraction
        self.epsilon_floor = epsilon_floor
        self.augment_obs_with_expert_state = augment_obs_with_expert_state
        self.store_policy_action = store_policy_action
        self.lcb_gated_bootstrap = lcb_gated_bootstrap
        self.normalize_obs_running = normalize_obs_running
        # Resolve the static expert_state_aug_dim once. 0 for stateless or
        # when the augmentation is off.
        if augment_obs_with_expert_state and expert_policy is not None:
            from ajax.environments.interaction import expert_state_dim
            self.expert_state_aug_dim = expert_state_dim(expert_policy)
        else:
            self.expert_state_aug_dim = 0
        self.target_entropy_initial = (
            target_entropy_initial_per_dim * action_dim
            if target_entropy_initial_per_dim is not None
            else None
        )
        self.target_entropy_ramp_frac = target_entropy_ramp_frac
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
        self.extra_actor_loss_fn = extra_actor_loss_fn
        self.extra_critic_loss_fn = extra_critic_loss_fn
        self.init_transform = init_transform
        self.auxiliary_update = auxiliary_update
        self.extra_eval_metrics = extra_eval_metrics

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
            exploration_lcb=self.exploration_lcb,
            exploration_thompson=self.exploration_thompson,
            expert_fraction=self.expert_fraction,
            epsilon_floor=self.epsilon_floor,
            augment_obs_with_expert_state=self.augment_obs_with_expert_state,
            store_policy_action=self.store_policy_action,
            lcb_gated_bootstrap=self.lcb_gated_bootstrap,
            expert_state_aug_dim=self.expert_state_aug_dim,
            normalize_obs_running=self.normalize_obs_running,
            target_entropy_initial=self.target_entropy_initial,
            target_entropy_ramp_frac=self.target_entropy_ramp_frac,
            lcb_beta_init=self.lcb_beta_init,
            lcb_beta_decay_k=self.lcb_beta_decay_k,
            lcb_temperature=self.lcb_temperature,
            ibrl_bootstrap=self.ibrl_bootstrap,
            use_residual_rl=self.residual,
            residual_scale=self.residual_scale,
            jsrl_curriculum=self.jsrl_curriculum,
            jsrl_episode_length=self.jsrl_episode_length,
            jsrl_decay_frac=self.jsrl_decay_frac,
            target_entropy_far=self.target_entropy_far,
            mc_variance_threshold=self.mc_variance_threshold,
            use_phi_refresh=self.use_phi_refresh,
            phi_refresh_interval=self.phi_refresh_interval,
            phi_refresh_steps=self.phi_refresh_steps,
            pid_actor_config=self.pid_actor_config,
            action_dim_override=self.action_dim_override,
            action_pipeline=self.action_pipeline,
            target_modifier=self.target_modifier,
            obs_preprocessor=self.obs_preprocessor,
            policy_action_transform=self.policy_action_transform,
            eval_action_transform=self.eval_action_transform,
            runtime_maintenance=self.runtime_maintenance,
            extra_actor_loss_fn=self.extra_actor_loss_fn,
            extra_critic_loss_fn=self.extra_critic_loss_fn,
            init_transform=self.init_transform,
            auxiliary_update=self.auxiliary_update,
            extra_eval_metrics=self.extra_eval_metrics,
        )
