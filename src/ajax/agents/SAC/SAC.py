from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional, Union

from target_gym import PlaneParams

from ajax.agents.base import ActorCritic
from ajax.agents.cloning import CloningConfig
from ajax.agents.SAC.sac import make_train
from ajax.agents.SAC.state import SACConfig
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.extensions.base import Extension
from ajax.modules.pid_actor import PIDActorConfig
from ajax.state import AlphaConfig, NetworkConfig
from ajax.types import EnvType


class SAC(ActorCritic):
    """Soft Actor-Critic agent for continuous action spaces.

    Phase-5 surface: the legacy ~75-kwarg back-compat shim
    (``ibrl_bootstrap`` / ``use_critic_blend`` / ``use_online_bc`` /
    ``use_phi_refresh`` / ``use_mc_critic_pretrain`` /
    ``use_expert_guided_exploration`` / the ``exploration_*`` +
    ``lcb_*`` family / ``jsrl_episode_length`` + ``jsrl_decay_frac`` /
    ``mc_pretrain_n_mc_*`` / ``online_critic_pretrain_*`` /
    ``mc_variance_threshold`` / ``lcb_gated_bootstrap`` /
    ``phi_refresh_interval`` / ``phi_refresh_steps`` /
    ``critic_warmup_frac`` / ``bc_coef``) was stripped. ``extensions=``
    is now the sole surface for composing research features.

    Several kwargs survived because they thread into network init,
    collection, or the action pipeline at a level the extension
    framework doesn't reach (``residual``, ``jsrl_curriculum``,
    ``use_box``, ``use_bellman_critic_pretrain``, ``use_pid_policy``,
    ``augment_obs_with_expert_action``, ``augment_obs_with_expert_state``,
    ``use_train_frac``, ``normalize_obs_running``, ``store_policy_action``,
    ``extra_critic_head_*``, the ``Optional[Callable]`` user-hook
    overrides). They mirror the matching extension's "static" flag.
    """

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
        # Cloning / pre-train kwargs (kept: route through CloningConfig)
        actor_cloning_epochs: int = 10,
        critic_cloning_epochs: int = 10,
        actor_cloning_lr: float = 1e-3,
        critic_cloning_lr: float = 1e-3,
        actor_cloning_batch_size: int = 64,
        critic_cloning_batch_size: int = 64,
        pre_train_n_steps: int = 0,
        # Expert objects: still used by collection / cloning / a fair
        # number of action-pipeline call sites. ``expert_policy=`` is
        # also accepted by individual Extensions and SHOULD be supplied
        # to them too — passing it here is shorthand for the legacy
        # "experts everywhere" path used by AjaxExperiments.
        expert_policy: Optional[Callable] = None,
        eval_expert_policy: Optional[Callable] = None,  # for eval logging only
        imitation_coef: Union[float, Callable[[int], float]] = 0.0,
        distance_to_stable: Optional[Callable] = None,
        imitation_coef_offset: float = 0.0,
        action_scale: float = 1.0,
        early_termination_condition: Optional[Callable] = None,
        # Residual RL: kept because the residual transform threads
        # through ``make_action_pipeline`` AND because ``residual``
        # toggles the network init's residual-aware head. The
        # ResidualPolicy extension still owns the actor-loss / TD-target
        # transform math via :meth:`transform_action`.
        residual: bool = False,
        residual_scale: float = 1.0,
        # JSRL: kept because ``jsrl_curriculum=True`` gates the
        # per-env ``step_in_episode`` counter init in :func:`init_SAC`.
        # The curriculum math lives on :class:`JSRLCurriculum`.action.
        jsrl_curriculum: bool = False,
        # PID policy: gain-mode short-circuit in the action pipeline,
        # changes the action_dim of actor/critic/buffer.
        use_pid_policy: bool = False,
        fixed_alpha: bool = False,
        num_critics: int = 2,
        # Multi-objective critic: extra value heads sharing the SAC
        # critic's encoder. SafeSAC sets this to ("v_safety",).
        extra_critic_head_names: tuple = (),
        extra_critic_head_dims: tuple = (),
        # ExpertGuidance secondary plumbing. ``use_expert_guidance``
        # gates the expert-action telemetry / loss term inside
        # ``update_agent`` (deeper than extension surface). The
        # ExpertGuidance extension owns the user-facing config; these
        # mirror its fields so ``init_SAC`` / ``training_iteration``
        # can read them as plain Python values.
        use_expert_guidance: bool = False,
        num_critic_updates: int = 1,
        expert_buffer_n_steps: int = 20_000,
        expert_mix_fraction: float = 0.1,
        box_threshold: float = 500.0,
        altitude_obs_idx: int = 1,
        target_obs_idx: int = 6,
        # MC sizing kwarg threaded into the inline Bellman-pretrain
        # block (``use_bellman_critic_pretrain``). MCPretrain extensions
        # own the equivalent for the MC path via ``n_steps``.
        mc_pretrain_n_steps: int = 5_000,
        # Obs augmentation: changes init_SAC / collect_experience
        # network input dim. The runtime stop-gradient on the augmented
        # dims lives on :meth:`ExpertObsAugmentation.on_obs`.
        augment_obs_with_expert_action: bool = False,
        use_bellman_critic_pretrain: bool = False,
        # Train-fraction conditioning: append timestep/total_timesteps
        # to obs (changes the network input dim).
        use_train_frac: bool = False,
        # Update start thresholds
        policy_update_start: int = 2_000,
        alpha_update_start: int = 2_000,
        # Value-threshold box: gates ``_box_v_min/_box_v_max``
        # resolution in ``make_scan_fn``; the ValueBox extension owns
        # the override math via :meth:`action`.
        use_box: bool = False,
        # EDGE telemetry plumbing (threaded into ``training_iteration``
        # for logging). The EDGEExploration extension owns the gate
        # math; this is just the τ scalar.
        exploration_tau: float = 1.0,
        # Warmup expert-vs-uniform mix fraction (gates uniform sampling
        # in the action pipeline). Distinct from the ExpertGuidance
        # extension's own fields.
        expert_fraction: float = 0.7,
        # Expert-state augmentation (PID integrator state etc.).
        # Changes the network input dim.
        augment_obs_with_expert_state: bool = False,
        # Agent-side running observation normalisation. When True the
        # full augmented obs (env_obs + flatten(expert_state)) is
        # z-scored using running mean/var maintained inside the agent's
        # CollectorState.
        normalize_obs_running: bool = False,
        # Off-policy-correctness ablation: store the policy's sampled
        # action in the buffer instead of the executed action.
        store_policy_action: bool = False,
        # target_entropy time ramp (threaded into
        # ``temperature_loss_function``).
        target_entropy_initial_per_dim: Optional[float] = None,
        target_entropy_ramp_frac: float = 0.5,
        # Critic / actor BC controls (route through CloningConfig)
        skip_actor_pretrain: bool = False,
        skip_critic_pretrain: bool = True,
        reset_log_std_after_bc: bool = False,
        reset_actor_head_after_bc: bool = False,
        # Distance-modulated entropy target (None = disabled).
        target_entropy_far: Optional[float] = None,
        # PID actor: actor network predicts PID gains instead of raw
        # actions (affects network architecture).
        pid_actor_config: Optional[PIDActorConfig] = None,
        # --- Composable hook overrides (None = build from extensions / defaults) ---
        action_pipeline: Optional[Callable] = None,
        obs_preprocessor: Optional[Callable] = None,
        policy_action_transform: Optional[Callable] = None,
        eval_action_transform: Optional[Callable] = None,
        extra_actor_loss_fn: Optional[Callable] = None,
        extra_critic_loss_fn: Optional[Callable] = None,
        her_relabel_fn: Optional[Callable] = None,
        init_transform: Optional[Callable] = None,
        auxiliary_update: Optional[Callable] = None,
        extra_eval_metrics: Optional[Callable] = None,
        # --- Composable research features ---
        extensions: Sequence[Extension] = (),
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
            extensions=extensions,
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
            if residual:
                raise ValueError(
                    "use_pid_policy=True is incompatible with residual=True;"
                    " gain-mode uses a 7-D action space."
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
        self.extra_critic_head_names = tuple(extra_critic_head_names)
        self.extra_critic_head_dims = tuple(extra_critic_head_dims)
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
        self.use_pid_policy = use_pid_policy
        self.fixed_alpha = fixed_alpha
        self.use_expert_guidance = use_expert_guidance
        self.num_critic_updates = num_critic_updates
        self.expert_buffer_n_steps = expert_buffer_n_steps
        self.expert_mix_fraction = expert_mix_fraction
        self.box_threshold = box_threshold
        self.altitude_obs_idx = altitude_obs_idx
        self.target_obs_idx = target_obs_idx
        self.mc_pretrain_n_steps = mc_pretrain_n_steps
        self.augment_obs_with_expert_action = augment_obs_with_expert_action
        self.use_bellman_critic_pretrain = use_bellman_critic_pretrain
        self.use_train_frac = use_train_frac
        self.policy_update_start = policy_update_start
        self.alpha_update_start = alpha_update_start
        self.use_box = use_box
        self.exploration_tau = exploration_tau
        self.expert_fraction = expert_fraction
        self.augment_obs_with_expert_state = augment_obs_with_expert_state
        self.store_policy_action = store_policy_action
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
        self.target_entropy_far = target_entropy_far
        self.pid_actor_config = pid_actor_config
        self.action_pipeline = action_pipeline
        self.obs_preprocessor = obs_preprocessor
        self.policy_action_transform = policy_action_transform
        self.eval_action_transform = eval_action_transform
        self.extra_actor_loss_fn = extra_actor_loss_fn
        self.extra_critic_loss_fn = extra_critic_loss_fn
        self.her_relabel_fn = her_relabel_fn
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
            extra_critic_head_names=self.extra_critic_head_names,
            extra_critic_head_dims=self.extra_critic_head_dims,
            use_expert_guidance=self.use_expert_guidance,
            num_critic_updates=self.num_critic_updates,
            expert_buffer_n_steps=self.expert_buffer_n_steps,
            expert_mix_fraction=self.expert_mix_fraction,
            box_threshold=self.box_threshold,
            altitude_obs_idx=self.altitude_obs_idx,
            target_obs_idx=self.target_obs_idx,
            mc_pretrain_n_steps=self.mc_pretrain_n_steps,
            augment_obs_with_expert_action=self.augment_obs_with_expert_action,
            use_bellman_critic_pretrain=self.use_bellman_critic_pretrain,
            use_train_frac=self.use_train_frac,
            policy_update_start=self.policy_update_start,
            alpha_update_start=self.alpha_update_start,
            use_box=self.use_box,
            exploration_tau=self.exploration_tau,
            expert_fraction=self.expert_fraction,
            augment_obs_with_expert_state=self.augment_obs_with_expert_state,
            store_policy_action=self.store_policy_action,
            expert_state_aug_dim=self.expert_state_aug_dim,
            normalize_obs_running=self.normalize_obs_running,
            target_entropy_initial=self.target_entropy_initial,
            target_entropy_ramp_frac=self.target_entropy_ramp_frac,
            use_residual_rl=self.residual,
            residual_scale=self.residual_scale,
            jsrl_curriculum=self.jsrl_curriculum,
            target_entropy_far=self.target_entropy_far,
            pid_actor_config=self.pid_actor_config,
            action_dim_override=self.action_dim_override,
            action_pipeline=self.action_pipeline,
            obs_preprocessor=self.obs_preprocessor,
            policy_action_transform=self.policy_action_transform,
            eval_action_transform=self.eval_action_transform,
            extra_actor_loss_fn=self.extra_actor_loss_fn,
            extra_critic_loss_fn=self.extra_critic_loss_fn,
            her_relabel_fn=self.her_relabel_fn,
            init_transform=self.init_transform,
            auxiliary_update=self.auxiliary_update,
            extra_eval_metrics=self.extra_eval_metrics,
            extensions=tuple(self.extension_stack.extensions),
        )
