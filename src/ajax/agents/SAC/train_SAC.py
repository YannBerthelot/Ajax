from collections.abc import Sequence
from dataclasses import fields
from math import floor
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial

from ajax.agents.cloning import (
    CloningConfig,
    get_cloning_args,
    get_pre_trained_agent,
)
from ajax.agents.SAC import core
from ajax.agents.SAC.state import SACConfig, SACState
from ajax.agents.SAC.utils import SquashedNormal
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    collect_experience,
    get_action_and_log_probs,
    get_pi,
    init_collector_state,
    should_use_uniform_sampling,
)
from ajax.environments.utils import (
    check_env_is_gymnax,
    get_action_dim,
    get_state_action_shapes,
)
from ajax.log import evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.modules.expert import (
    augment_obs_if_needed,
    blend_modify_target,
    compute_behavior_kpis,
    compute_expert_diagnostics,
    compute_online_bc_loss,
    detach_obs_expert_dims,
    mc_correction_modify_target,
    residual_action_transform,
)
from ajax.modules.exploration import (
    EDGEAuxiliaries,
    box_action_override,
    box_compute_state,
    box_compute_threshold,
    compute_edge_diagnostics,
    edge_argmax_gate,
    edge_boltzmann_gate,
    edge_compute_decay,
    edge_compute_value_gap,
    edge_fixed_gate,
)
from ajax.modules.pretrain import (
    PhiRefreshAuxiliaries,
    collect_and_store_expert_transitions,
    pretrain_critic_bellman,
    pretrain_critic_mc,
    pretrain_critic_online_light,
    refresh_phi_star,
)
from ajax.networks.networks import (
    get_initialized_actor_critic,
    get_initialized_critic,
    predict_value,
)
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.types import BufferType

# ---------------------------------------------------------------------------
# Action pipeline — composable exploration for collect_experience
# ---------------------------------------------------------------------------


class ActionPipelineResult(NamedTuple):
    """Result from the SAC action pipeline used by collect_experience."""

    env_action: jax.Array  # action sent to env
    policy_action: jax.Array  # actor's original action (stored in transition)
    log_probs: jax.Array  # actor's log probs
    is_expert_flag: jax.Array  # expert tracking for buffer
    in_value_box: jax.Array  # box membership (zeros when no box)
    entry_bonus: jax.Array  # box entry bonus (zeros when no box)
    rng: jax.Array  # updated rng (EDGE may consume it)


def make_action_pipeline(
    expert_policy,
    recurrent,
    env_args,
    # Box
    use_box=False,
    box_v_min=0.0,
    box_v_max=0.0,
    # EDGE
    use_expert_guided_exploration=False,
    exploration_decay_frac=0.30,
    exploration_tau=1.0,
    exploration_boltzmann=False,
    exploration_argmax=False,
    fixed_exploration_prob=0.5,
    # Action transforms
    use_residual_rl=False,
    use_pid_policy=False,
    # Obs augmentation
    augment_obs_with_expert_action=False,
    # Context
    total_timesteps=1,
    expert_fraction=0.7,
):
    """Compose the SAC action pipeline for collect_experience.

    Returns None for vanilla SAC (no expert). When provided, the pipeline
    handles obs augmentation, action selection (EDGE, box, residual, PID),
    and warmup expert/uniform mixing.

    All boolean flags are resolved at Python level (trace time), so the
    compiled graph only contains the active branches.
    """
    if expert_policy is None:
        return None

    def pipeline(agent_state, raw_obs, rng, uniform, mix_key, action_key):
        # --- Box state computation ---
        if use_box:
            train_frac = agent_state.collector_state.timestep / total_timesteps
            threshold = box_compute_threshold(box_v_min, box_v_max, train_frac)
            obs_for_box = agent_state.collector_state.last_obs
            raw_for_box = raw_obs if raw_obs is not None else obs_for_box[..., :-1]
            in_value_box, entry_bonus, _ = box_compute_state(
                obs_for_box,
                raw_for_box,
                expert_policy,
                agent_state.critic_state,
                agent_state.expert_critic_params,
                threshold,
                agent_state.collector_state.last_in_box,
            )
        else:
            in_value_box = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)
            entry_bonus = jnp.zeros((env_args.n_envs, 1), dtype=jnp.float32)

        # --- Obs augmentation: [env_obs | a_expert | train_frac] ---
        if augment_obs_with_expert_action:
            _raw_for_aug = (
                raw_obs if raw_obs is not None else agent_state.collector_state.last_obs
            )
            _a_expert = jax.lax.stop_gradient(expert_policy(_raw_for_aug))
            _last_obs = agent_state.collector_state.last_obs
            _augmented_obs = jnp.concatenate(
                [_last_obs[..., :-1], _a_expert, _last_obs[..., -1:]], axis=-1
            )
            agent_state_for_actor = agent_state.replace(
                collector_state=agent_state.collector_state.replace(
                    last_obs=_augmented_obs
                )
            )
        else:
            agent_state_for_actor = agent_state

        # --- Policy action ---
        action, log_probs = get_action_and_log_probs(
            action_key=action_key,
            agent_state=agent_state_for_actor,
            recurrent=recurrent,
            uniform=False,
        )

        # --- Expert action ---
        expert_action = jax.lax.stop_gradient(expert_policy(raw_obs))

        # --- Uniform action (for warmup) ---
        uniform_action = jax.random.uniform(
            mix_key, minval=-1.0, maxval=1.0, shape=action.shape
        )

        # --- Post-warmup action ---
        in_box = (
            env_args.env.trunc_condition(
                agent_state.collector_state.env_state, env_args.env_params
            )
            if "trunc_condition" in dir(env_args.env)
            else jnp.zeros_like(action[..., :1])
        )
        if use_residual_rl:
            post_warmup_action = jnp.clip(expert_action + action, -1.0, 1.0)
        else:
            post_warmup_action = (1 - in_box) * action + in_box * expert_action

        # --- EDGE (Expert Decayed Guided Exploration) ---
        if use_expert_guided_exploration:
            obs_for_edge = agent_state.collector_state.last_obs
            edge_critic_params = (
                agent_state.expert_critic_params
                if agent_state.expert_critic_params is not None
                else agent_state.critic_state.params
            )
            gap, q_policy = edge_compute_value_gap(
                obs_for_edge,
                action,
                expert_action,
                agent_state.critic_state,
                edge_critic_params,
            )
            decay = edge_compute_decay(
                agent_state.collector_state.timestep,
                total_timesteps,
                exploration_decay_frac,
            )
            if exploration_argmax:
                use_expert_edge, rng = edge_argmax_gate(gap, decay, rng)
            elif exploration_boltzmann:
                use_expert_edge, rng = edge_boltzmann_gate(
                    gap,
                    decay,
                    rng,
                    q_policy,
                    exploration_tau,
                )
            else:
                use_expert_edge, rng = edge_fixed_gate(
                    gap,
                    decay,
                    rng,
                    fixed_exploration_prob,
                )
            post_warmup_action = jnp.where(
                use_expert_edge, expert_action, post_warmup_action
            )

        # --- Warmup action ---
        if use_residual_rl:
            warmup_action = uniform_action
            use_expert_this_step = jnp.zeros((), dtype=jnp.bool_)
        else:
            use_expert_this_step = jax.random.uniform(mix_key) < expert_fraction
            warmup_action = jnp.where(
                use_expert_this_step, expert_action, uniform_action
            )

        # --- Expert flag tracking ---
        _post_expert = jnp.zeros_like(action[..., :1], dtype=jnp.float32)
        if use_expert_guided_exploration:
            _post_expert = jnp.maximum(
                _post_expert, use_expert_edge.astype(jnp.float32)
            )
        if use_box:
            _post_expert = jnp.maximum(_post_expert, in_value_box.astype(jnp.float32))
        _warmup_expert = jnp.ones_like(
            action[..., :1], dtype=jnp.float32
        ) * use_expert_this_step.astype(jnp.float32)
        is_expert_flag = jax.lax.cond(
            uniform, lambda: _warmup_expert, lambda: _post_expert
        )

        # --- PID override ---
        if use_pid_policy:
            post_warmup_action = expert_action
            warmup_action = expert_action

        # --- Final action selection ---
        env_action = jax.lax.cond(
            uniform, lambda: warmup_action, lambda: post_warmup_action
        )

        # --- Box action override ---
        if use_box:
            env_action = box_action_override(env_action, expert_action, in_value_box)

        return ActionPipelineResult(
            env_action=env_action,
            policy_action=action,
            log_probs=log_probs,
            is_expert_flag=is_expert_flag,
            in_value_box=in_value_box,
            entry_bonus=entry_bonus,
            rng=rng,
        )

    return pipeline


# ---------------------------------------------------------------------------
# Critic target modifier — composable IBRL / blend / MC correction
# ---------------------------------------------------------------------------


def make_target_modifier(
    ibrl_bootstrap=False,
    use_critic_blend=False,
    critic_warmup_frac=0.15,
    mc_variance_threshold=None,
    expert_policy=None,
    total_timesteps=1,
    augment_obs_with_expert_action=False,
    recurrent=False,
):
    """Compose critic target modifiers: IBRL → blend → MC correction.

    Returns None when no modifiers are active. When provided, the modifier
    replaces 6 params in update_value_functions with a single callable.
    Boolean flags are resolved at Python level (trace time).
    """
    has_ibrl = ibrl_bootstrap and expert_policy is not None
    has_blend = use_critic_blend and expert_policy is not None
    has_mc = mc_variance_threshold is not None

    if not (has_ibrl or has_blend or has_mc):
        return None

    def modifier(
        target_q,
        agent_state,
        observations,
        actions,
        next_observations,
        dones,
        gamma,
        rng,
        q_preds,
    ):
        alpha_blend_logged = jnp.zeros(1)
        mc_correction_frac = jnp.zeros(1)

        if has_ibrl:
            _next_raw = (
                next_observations[..., :-1]
                if augment_obs_with_expert_action
                else next_observations
            )
            next_expert_actions = jax.lax.stop_gradient(expert_policy(_next_raw))
            q_targets_expert = predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.critic_state.target_params,
                x=jnp.concatenate((next_observations, next_expert_actions), axis=-1),
            )
            min_q_expert = jnp.min(q_targets_expert, axis=0, keepdims=False)

            next_pi, _ = get_pi(
                actor_state=agent_state.actor_state,
                actor_params=agent_state.actor_state.params,
                obs=next_observations,
                done=dones,
                recurrent=recurrent,
            )
            ibrl_key, _ = jax.random.split(rng)
            next_actions_ibrl, _ = next_pi.sample_and_log_prob(seed=ibrl_key)
            q_targets_policy = predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.critic_state.target_params,
                x=jnp.concatenate((next_observations, next_actions_ibrl), axis=-1),
            )
            min_q_policy = jnp.min(q_targets_policy, axis=0, keepdims=False)

            gap = jax.lax.stop_gradient(
                gamma * (1.0 - dones) * jnp.maximum(min_q_expert - min_q_policy, 0.0)
            )
            target_q = target_q + gap

        if has_blend and agent_state.expert_critic_params is not None:
            next_raw = next_observations[..., :-1]
            a_expert_next = jax.lax.stop_gradient(expert_policy(next_raw))
            v_expert_next = jax.lax.stop_gradient(
                jnp.min(
                    predict_value(
                        critic_state=agent_state.critic_state,
                        critic_params=agent_state.expert_critic_params,
                        x=jnp.concatenate([next_observations, a_expert_next], axis=-1),
                    ),
                    axis=0,
                )
            )
            train_frac = agent_state.collector_state.timestep / total_timesteps
            alpha_blend_val = jnp.maximum(1.0 - train_frac / critic_warmup_frac, 0.0)
            target_q, alpha_blend_logged = blend_modify_target(
                target_q,
                v_expert_next,
                alpha_blend_val,
            )
            target_q = jax.lax.stop_gradient(target_q)

        if has_mc and agent_state.expert_critic_params is not None:
            q_var = q_preds.var(axis=0)[..., 0]
            target_q, mc_correction_frac = mc_correction_modify_target(
                target_q,
                agent_state.critic_state,
                agent_state.expert_critic_params,
                observations,
                actions,
                q_var,
                mc_variance_threshold,
            )

        return target_q, alpha_blend_logged, mc_correction_frac

    return modifier


# ---------------------------------------------------------------------------
# Policy modifiers — composable obs preprocessing, action transform, BC loss
# ---------------------------------------------------------------------------


def make_policy_obs_preprocessor(
    augment_obs_with_expert_action, detach_obs_aug_action, action_dim
):
    """Compose obs preprocessing for policy: stop-gradient expert-action dims.

    Returns None when not needed (no augmentation or no detach).
    """
    if not (augment_obs_with_expert_action and detach_obs_aug_action):
        return None

    def preprocess(observations):
        return detach_obs_expert_dims(observations, action_dim)

    return preprocess


def make_policy_action_transform(use_residual_rl, expert_policy):
    """Compose policy action transform: residual RL.

    Returns None for vanilla SAC. When provided, transforms actions
    before Q evaluation in the actor loss.
    """
    if not use_residual_rl or expert_policy is None:
        return None

    def transform(actions, raw_obs, a_expert_precomputed):
        a_exp = (
            a_expert_precomputed
            if a_expert_precomputed is not None
            else jax.lax.stop_gradient(expert_policy(raw_obs))
        )
        return residual_action_transform(actions, a_exp)

    return transform


def make_bc_loss_fn(use_online_bc, bc_coef, critic_warmup_frac, expert_policy):
    """Compose online decaying BC loss.

    Returns None when BC is disabled. Captures bc_coef and critic_warmup_frac
    in the closure so they don't need to be threaded through the call chain.
    """
    if not use_online_bc or expert_policy is None or critic_warmup_frac <= 0:
        return None

    def compute(
        pi_loc,
        a_expert,
        critic_state,
        expert_critic_params,
        observations,
        train_frac,
        expert_v_min,
        expert_v_max,
    ):
        return compute_online_bc_loss(
            pi_loc,
            a_expert,
            critic_state,
            expert_critic_params,
            observations,
            train_frac,
            critic_warmup_frac,
            expert_v_min,
            expert_v_max,
            bc_coef,
        )

    return compute


# ---------------------------------------------------------------------------
# Runtime maintenance — composable periodic φ* refresh
# ---------------------------------------------------------------------------


def make_runtime_maintenance(
    use_phi_refresh=False,
    phi_refresh_interval=500,
    phi_refresh_steps=20,
    gamma=0.99,
    reward_scale=1.0,
    expert_policy=None,
    buffer=None,
):
    """Compose periodic runtime maintenance (φ* refresh).

    Returns None when phi_refresh is disabled. When provided, runs
    self-consistent Bellman steps on expert buffer transitions periodically.
    Replaces use_phi_refresh, phi_refresh_interval, phi_refresh_steps
    in training_iteration.
    """
    if not use_phi_refresh or expert_policy is None:
        return None

    _zero = PhiRefreshAuxiliaries(
        loss_before=jnp.zeros(1),
        loss_after=jnp.zeros(1),
        expert_buffer_size=jnp.zeros(1),
    )

    def maintenance(agent_state):
        return jax.lax.cond(
            agent_state.collector_state.timestep % phi_refresh_interval == 0,
            lambda s: refresh_phi_star(
                s,
                buffer,
                phi_refresh_steps,
                gamma,
                reward_scale,
                expert_policy,
            ),
            lambda s: (s, _zero),
            operand=agent_state,
        )

    return maintenance


# ---------------------------------------------------------------------------
# Eval action transform — composable residual RL / PID for evaluate.py
# ---------------------------------------------------------------------------


def make_eval_action_transform(use_residual_rl=False, use_pid_policy=False):
    """Compose eval-time action transform for residual RL / PID.

    Returns None for vanilla SAC (default box-based handover in evaluate.py).
    Replaces use_residual_rl and use_pid_policy flags in evaluate.py/log.py.
    """
    if use_pid_policy:

        def transform(raw_actions, expert_actions):
            return expert_actions

        return transform
    if use_residual_rl:

        def transform(raw_actions, expert_actions):
            return jnp.clip(expert_actions + raw_actions, -1.0, 1.0)

        return transform
    return None


# ---------------------------------------------------------------------------
# Auxiliary dataclasses for logging
# ---------------------------------------------------------------------------


@struct.dataclass
class TemperatureAuxiliaries:
    alpha: jax.Array
    log_alpha: jax.Array
    effective_target_entropy: (
        jax.Array
    )  # actual target used in alpha update (distance-modulated when active)


@struct.dataclass
class PolicyAuxiliaries:
    # Core loss
    raw_loss: jax.Array  # α·log π - Q: pure SAC gradient
    policy_loss: jax.Array  # raw_loss + value constraint terms

    # Entropy diagnostics
    log_pi: jax.Array  # entropy proxy; tracks target_entropy
    policy_std: jax.Array  # mean unsquashed std; lower = more deterministic

    # Q-value diagnostics
    q_min: jax.Array  # Q(s, π(s)): what policy optimises
    q_expert: jax.Array  # Q(s, a_expert): expert value estimate

    # Expert diagnostics
    l2_expert: jax.Array  # ||π(s) - a_expert||^2: L2 distance to expert action
    above_expert_frac: jax.Array  # fraction of batch where policy beats expert

    # Online decaying BC term
    bc_term: jax.Array  # decaying online BC loss magnitude (0 after warmup_frac)

    # Policy behavior KPIs (from raw_obs — tell us what the policy actually does)
    altitude_error: jax.Array  # mean |z - target| over batch
    z_dot_mean: jax.Array  # mean |z_dot| over batch: 0 = stable, high = aggressive


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q_pred_min: jax.Array  # min over ensemble
    q_expert_mean: jax.Array  # critic's estimate of expert value
    q_gap: jax.Array  # q_expert - q_min: >0 = room to improve
    var_preds: jax.Array  # inter-critic variance
    expert_frac_in_buffer: jax.Array  # fraction of sampled batch flagged as expert
    alpha_blend: jax.Array  # current blend coefficient (1=pure expert, 0=pure Bellman)
    effective_threshold: jax.Array  # box threshold at current train_frac
    box_entry_rate: jax.Array  # fraction of batch inside value box
    mc_correction_frac: jax.Array  # fraction of batch where MC target replaced Bellman
    phi_star_q_gap_ood: (
        jax.Array
    )  # |Q_φ*(s,π*) - Q_φ(s,π*)| mean: φ* OOD coverage error


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    edge: EDGEAuxiliaries
    phi_refresh: PhiRefreshAuxiliaries


# ---------------------------------------------------------------------------
# Scalar alpha (temperature)
# ---------------------------------------------------------------------------


def create_alpha_train_state(
    learning_rate: float = 3e-4,
    alpha_init: float = 1.0,
) -> TrainState:
    return core.create_alpha_train_state(learning_rate, alpha_init)


# ---------------------------------------------------------------------------
# SAC initialization
# ---------------------------------------------------------------------------


def init_SAC(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    buffer: BufferType,
    window_size: int = 10,
    expert_policy: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    residual: bool = False,
    fixed_alpha: bool = False,
    max_timesteps: Optional[int] = None,
    num_critics: int = 2,
    expert_buffer_n_steps: int = 20_000,
    augment_obs_with_expert_action: bool = False,
    pid_actor_config=None,
) -> SACState:
    rng, init_key, collector_key, expert_key = jax.random.split(key, num=4)

    # When augment_obs_with_expert_action=True, the actor and critic receive
    # obs augmented with a_expert at runtime (action_dim extra dimensions).
    # We must initialise the networks with the matching inflated input size.
    if augment_obs_with_expert_action:
        _, action_shape = get_state_action_shapes(env_args.env)
        extra_obs_dim = action_shape[0]
    else:
        extra_obs_dim = 0

    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=num_critics,
        expert_policy=expert_policy,
        residual=False,
        fixed_alpha=False,
        max_timesteps=max_timesteps,
        extra_obs_dim=extra_obs_dim,
        pid_actor_config=pid_actor_config,
    )

    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
        window_size=window_size,
        max_timesteps=max_timesteps,
    )

    if (
        expert_policy is not None
        and buffer is not None
        and collector_state.buffer_state is not None
        and expert_buffer_n_steps > 0
    ):
        collector_state = collector_state.replace(
            buffer_state=collect_and_store_expert_transitions(
                expert_policy=expert_policy,
                env_args=env_args,
                buffer=buffer,
                buffer_state=collector_state.buffer_state,
                rng=expert_key,
                n_steps=expert_buffer_n_steps,
                max_timesteps=max_timesteps,
            )
        )

    alpha = create_alpha_train_state(**to_state_dict(alpha_args))

    return SACState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=alpha,
        collector_state=collector_state,
        lambda_param=1.0,
    )


# ---------------------------------------------------------------------------
# Critic update
# ---------------------------------------------------------------------------
# Uses core.compute_td_target for the base Bellman target, then applies
# expert target modifiers (IBRL, blend, MC correction) before the loss.


def update_value_functions(
    agent_state: SACState,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float = 1.0,
    expert_q: Optional[jax.Array] = None,
    target_modifier: Optional[Callable] = None,
) -> Tuple[SACState, ValueAuxiliaries]:
    value_loss_key, rng = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])

    # 1. Core Bellman target (pure SAC)
    target_q = core.compute_td_target(
        actor_state=agent_state.actor_state,
        critic_state=agent_state.critic_state,
        next_observations=next_observations,
        dones=dones,
        rewards=rewards,
        gamma=gamma,
        alpha=alpha,
        rng=value_loss_key,
        recurrent=recurrent,
        reward_scale=reward_scale,
    )

    # 2. Q predictions for diagnostics
    q_preds_for_var = predict_value(
        critic_state=agent_state.critic_state,
        critic_params=agent_state.critic_state.params,
        x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
    )

    # 3. Expert target modifiers (IBRL → blend → MC correction)
    alpha_blend_logged = jnp.zeros(1)
    mc_correction_frac = jnp.zeros(1)
    if target_modifier is not None:
        target_q, alpha_blend_logged, mc_correction_frac = target_modifier(
            target_q,
            agent_state,
            observations,
            actions,
            next_observations,
            dones,
            gamma,
            value_loss_key,
            q_preds_for_var,
        )

    # 4. Core critic loss (MSE against composed target)
    (loss, core_aux), grads = jax.value_and_grad(core.critic_loss_fn, has_aux=True)(
        agent_state.critic_state.params,
        agent_state.critic_state,
        observations,
        actions,
        target_q,
    )

    # 5. Assemble full ValueAuxiliaries with expert diagnostics
    q_pred_min_full = jnp.min(q_preds_for_var, axis=0)
    q_expert_mean = expert_q.mean().flatten() if expert_q is not None else jnp.zeros(1)
    q_gap = (
        (expert_q - q_pred_min_full).mean().flatten()
        if expert_q is not None
        else jnp.zeros(1)
    )

    aux = ValueAuxiliaries(
        critic_loss=core_aux.critic_loss,
        q_pred_min=core_aux.q_pred_min,
        q_expert_mean=q_expert_mean,
        q_gap=q_gap,
        var_preds=core_aux.var_preds,
        alpha_blend=alpha_blend_logged,
        effective_threshold=jnp.zeros(1),
        box_entry_rate=jnp.zeros(1),
        expert_frac_in_buffer=jnp.zeros(1),
        mc_correction_frac=mc_correction_frac,
        phi_star_q_gap_ood=jnp.zeros(1),
    )

    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    return agent_state.replace(rng=rng, critic_state=updated_critic_state), aux


# ---------------------------------------------------------------------------
# Policy update — SAC + value constraint
# ---------------------------------------------------------------------------


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    alpha: jax.Array,
    rng: jax.random.PRNGKey,
    raw_observations: Optional[jax.Array] = None,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    a_expert_precomputed: Optional[jax.Array] = None,
    train_frac: Optional[jax.Array] = None,
    expert_critic_params: Optional[Any] = None,
    expert_v_min: Optional[jax.Array] = None,
    expert_v_max: Optional[jax.Array] = None,
    # Composed policy modifiers (replace 6 boolean flags)
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    bc_loss_fn: Optional[Callable] = None,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """SAC actor loss with composable expert modifiers.

    Structure mirrors the critic side: core SAC loss + layered expert additions.
    1. Pre-process: obs_preprocessor (detach expert-action dims)
    2. Core: forward pass → sample → Q eval → α·log π - Q
    3. Modifier: policy_action_transform (residual RL before Q eval)
    4. Modifier: bc_loss_fn (online decaying BC term)
    5. Diagnostics: expert Q gap, L2 distance, behavior KPIs
    """
    _raw_obs = (
        raw_observations if raw_observations is not None else observations[..., :-1]
    )

    # 1. Pre-process: optionally detach expert-action dims in augmented obs
    obs_for_actor = (
        obs_preprocessor(observations) if obs_preprocessor is not None else observations
    )

    # 2. Core forward pass + sample
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=obs_for_actor,
        done=dones,
        recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = pi.sample_and_log_prob(seed=sample_key)
    log_probs = log_probs.sum(-1, keepdims=True)

    policy_std = (
        pi.unsquashed_stddev().mean()
        if isinstance(pi, SquashedNormal)
        else pi.stddev().mean()
    )

    # 3. Action transform modifier (residual RL)
    q_input_actions = (
        policy_action_transform(actions, _raw_obs, a_expert_precomputed)
        if policy_action_transform is not None
        else actions
    )

    # Core Q evaluation and SAC loss
    q_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_states.params,
        x=jnp.concatenate([observations, q_input_actions], axis=-1),
    )
    q_min = jnp.min(q_preds, axis=0)
    loss_actor = alpha * log_probs - q_min

    # 4. Expert diagnostics and BC loss
    needs_expert = expert_policy is not None and use_expert_guidance
    needs_bc = bc_loss_fn is not None and expert_critic_params is not None

    if needs_expert or needs_bc:
        a_expert = (
            a_expert_precomputed
            if a_expert_precomputed is not None
            else jax.lax.stop_gradient(expert_policy(_raw_obs))
        )
    else:
        a_expert = None

    # Expert diagnostics (no gradient effect)
    if needs_expert:
        q_expert_logged, l2_expert_logged, above_expert_frac = (
            compute_expert_diagnostics(
                critic_states,
                observations,
                q_min,
                a_expert,
                pi.distribution.loc,
            )
        )
    else:
        l2_expert_logged = jnp.zeros(())
        q_expert_logged = jnp.zeros(())
        above_expert_frac = jnp.zeros(())

    # Online decaying BC term
    if needs_bc:
        bc_term = bc_loss_fn(
            pi.distribution.loc,
            a_expert,
            critic_states,
            expert_critic_params,
            observations,
            train_frac,
            expert_v_min,
            expert_v_max,
        )
    else:
        bc_term = jnp.zeros(())

    total_loss = loss_actor.mean() + bc_term

    # 5. Behavior KPIs
    altitude_error_val, z_dot_mean_val = compute_behavior_kpis(
        _raw_obs,
        altitude_obs_idx,
        target_obs_idx,
    )

    return total_loss, PolicyAuxiliaries(
        policy_loss=total_loss,
        log_pi=log_probs.mean(),
        policy_std=policy_std,
        q_min=q_min.mean(),
        q_expert=q_expert_logged,
        l2_expert=l2_expert_logged,
        above_expert_frac=above_expert_frac,
        altitude_error=altitude_error_val,
        z_dot_mean=z_dot_mean_val,
        raw_loss=loss_actor.mean(),
        bc_term=bc_term,
    )


def update_policy(
    agent_state: SACState,
    observations: jax.Array,
    done: Optional[jax.Array],
    recurrent: bool,
    raw_observations: jax.Array,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    a_expert_precomputed: Optional[jax.Array] = None,
    train_frac: Optional[jax.Array] = None,
    # Composed policy modifiers
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    bc_loss_fn: Optional[Callable] = None,
) -> Tuple[SACState, PolicyAuxiliaries, jax.Array]:
    """Returns (new_state, aux, log_probs) — log_probs reused by update_temperature
    to avoid a redundant actor forward pass."""
    rng, policy_key = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])

    (loss, aux), grads = jax.value_and_grad(
        policy_loss_function, has_aux=True, argnums=0
    )(
        agent_state.actor_state.params,
        agent_state.actor_state,
        agent_state.critic_state,
        observations,
        done,
        recurrent,
        alpha,
        policy_key,
        raw_observations=raw_observations,
        expert_policy=expert_policy,
        use_expert_guidance=use_expert_guidance,
        altitude_obs_idx=altitude_obs_idx,
        target_obs_idx=target_obs_idx,
        a_expert_precomputed=a_expert_precomputed,
        train_frac=train_frac,
        expert_critic_params=agent_state.expert_critic_params,
        expert_v_min=agent_state.expert_v_min,
        expert_v_max=agent_state.expert_v_max,
        obs_preprocessor=obs_preprocessor,
        policy_action_transform=policy_action_transform,
        bc_loss_fn=bc_loss_fn,
    )

    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)

    # Recompute log_probs from updated actor for temperature update reuse
    temp_rng, temp_sample_key = jax.random.split(rng)
    pi, _ = get_pi(
        actor_state=updated_actor_state,
        actor_params=updated_actor_state.params,
        obs=observations,
        done=done,
        recurrent=recurrent,
    )
    _, log_probs = pi.sample_and_log_prob(seed=temp_sample_key)
    return (
        agent_state.replace(rng=temp_rng, actor_state=updated_actor_state),
        aux,
        jax.lax.stop_gradient(log_probs),
    )


# ---------------------------------------------------------------------------
# Temperature update with adaptive target entropy
# ---------------------------------------------------------------------------


def temperature_loss_function(
    log_alpha_params: FrozenDict,
    corrected_log_probs: jax.Array,
    effective_target_entropy: jax.Array,
) -> Tuple[jax.Array, TemperatureAuxiliaries]:
    loss, core_aux = core.temperature_loss_fn(
        log_alpha_params,
        corrected_log_probs,
        effective_target_entropy,
    )
    return loss, TemperatureAuxiliaries(
        alpha=core_aux.alpha,
        log_alpha=core_aux.log_alpha,
        effective_target_entropy=core_aux.effective_target_entropy,
    )


def update_temperature(
    agent_state: SACState,
    log_probs: jax.Array,
    effective_target_entropy: jax.Array,
) -> Tuple[SACState, TemperatureAuxiliaries]:
    """Standard SAC temperature update."""
    (loss, aux), grads = jax.value_and_grad(temperature_loss_function, has_aux=True)(
        agent_state.alpha.params,
        log_probs.sum(-1),
        effective_target_entropy,
    )
    new_alpha_state = agent_state.alpha.apply_gradients(grads=grads)
    return agent_state.replace(alpha=new_alpha_state), jax.lax.stop_gradient(aux)


# ---------------------------------------------------------------------------
# Target network update
# ---------------------------------------------------------------------------


def update_target_networks(agent_state: SACState, tau: float) -> SACState:
    return agent_state.replace(
        critic_state=agent_state.critic_state.soft_update(tau=tau)
    )


# ---------------------------------------------------------------------------
# Agent update (one gradient step)
# ---------------------------------------------------------------------------


def update_agent(
    agent_state: SACState,
    _: Any,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    action_dim: int,
    target_entropy: float,
    tau: float,
    num_critic_updates: int = 1,
    reward_scale: float = 1.0,
    additional_transition: Optional[Any] = None,
    transition_mix_fraction: float = 1.0,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    expert_mix_fraction: float = 0.1,
    box_threshold: float = 500.0,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    augment_obs_with_expert_action: bool = False,
    total_timesteps: int = 1,
    target_entropy_far: Optional[float] = None,
    exploration_tau: float = 1.0,
    # Composed modules
    target_modifier: Optional[Callable] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    bc_loss_fn: Optional[Callable] = None,
) -> Tuple[SACState, AuxiliaryLogs]:
    sample_key, expert_sample_key, rng = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=rng)

    # --- Sample from buffer ---
    if buffer is not None and agent_state.collector_state.buffer_state is not None:
        (
            observations,
            terminated,
            truncated,
            next_observations,
            rewards,
            actions,
            raw_observations,
            is_expert,
        ) = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, sample_key
        )
        expert_frac_in_buffer = is_expert.mean()
        original_transition = Transition(
            observations,
            actions,
            rewards,
            terminated,
            truncated,
            next_observations,
            raw_obs=raw_observations,
        )

        if additional_transition is not None and transition_mix_fraction < 1.0:
            len_original = len(observations)
            n_from_buffer = floor(transition_mix_fraction * len_original)
            n_from_online = len_original - n_from_buffer
            additional_transition = jax.tree.map(
                lambda x: jax.random.choice(sample_key, x, shape=(n_from_online,)),
                additional_transition,
            )
            transition = jax.tree.map(
                lambda x, y: (
                    None
                    if (x is None or y is None)
                    else jnp.concatenate([x[:n_from_buffer], y], axis=0)
                ),
                original_transition,
                additional_transition,
                is_leaf=lambda x: x is None,
            )
        else:
            transition = original_transition

    elif additional_transition is not None:
        transition = additional_transition
        expert_frac_in_buffer = jnp.zeros(())
    else:
        raise ValueError("Either buffer or additional_transition must be provided.")

    # --- Expert batch mixing ---
    if expert_mix_fraction > 0.0 and expert_policy is not None:
        (
            exp_obs,
            exp_terminated,
            exp_truncated,
            exp_next_obs,
            exp_rewards,
            exp_actions,
            exp_raw_obs,
            _,
        ) = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, expert_sample_key
        )

        n_total = transition.obs.shape[0]
        n_expert = floor(expert_mix_fraction * n_total)
        n_online = n_total - n_expert

        def _cat(a, b):
            if a is None or b is None:
                return a
            return jnp.concatenate([a[:n_online], b[:n_expert]], axis=0)

        transition = Transition(
            obs=_cat(transition.obs, exp_obs),
            action=_cat(transition.action, exp_actions),
            reward=_cat(transition.reward, exp_rewards),
            terminated=_cat(transition.terminated, exp_terminated),
            truncated=_cat(transition.truncated, exp_truncated),
            next_obs=_cat(transition.next_obs, exp_next_obs),
            raw_obs=_cat(transition.raw_obs, exp_raw_obs),
        )

    dones = jnp.logical_or(transition.terminated, transition.truncated)

    # --- Obs augmentation: append a_expert to obs and next_obs ---
    # Must happen before any network call (critic, actor, policy loss).
    # raw_obs gives the env observations without train_frac, which is what
    # expert_policy expects. For next_obs we strip the last dim (train_frac).
    if augment_obs_with_expert_action and expert_policy is not None:
        _raw = (
            transition.raw_obs
            if transition.raw_obs is not None
            else transition.obs[..., :-1]
        )
        _raw_next = transition.next_obs[..., :-1]  # strip train_frac
        aug_obs = augment_obs_if_needed(transition.obs, _raw, expert_policy, True)
        aug_next_obs = augment_obs_if_needed(
            transition.next_obs, _raw_next, expert_policy, True
        )
        transition = transition.replace(obs=aug_obs, next_obs=aug_next_obs)

    # --- Pre-compute Q(s, a_expert) and a_expert once for critic logging + policy ---
    # Avoids computing expert_policy twice (once here, once inside policy_loss_function)
    expert_q = None
    a_expert_precomputed = None
    needs_expert = expert_policy is not None and (
        use_expert_guidance
        or policy_action_transform is not None
        or bc_loss_fn is not None
    )
    if needs_expert:
        _raw = (
            transition.raw_obs
            if transition.raw_obs is not None
            else transition.obs[..., :-1]
        )
        a_expert_precomputed = jax.lax.stop_gradient(expert_policy(_raw))
        # transition.obs is already augmented at this point if augment_obs_with_expert_action
        expert_q = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.critic_state.params,
                    x=jnp.concatenate([transition.obs, a_expert_precomputed], axis=-1),
                ),
                axis=0,
            )
        )

    # --- φ* OOD quality: |Q_φ*(s,π*) - Q_φ(s,π*)| on training batch ---
    # Measures how much frozen φ* disagrees with the live critic on expert actions.
    # Non-zero only when MC pretrain has been run (expert_critic_params is not None).
    phi_star_q_gap_ood = jnp.zeros(())
    if (
        agent_state.expert_critic_params is not None
        and expert_q is not None
        and a_expert_precomputed is not None
    ):
        q_phi_star = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.expert_critic_params,
                    x=jnp.concatenate([transition.obs, a_expert_precomputed], axis=-1),
                ),
                axis=0,
            )
        )
        phi_star_q_gap_ood = jnp.abs(
            q_phi_star - jax.lax.stop_gradient(expert_q)
        ).mean()

    # --- Critic updates ---
    def critic_update_step(carry, _):
        agent_state = carry
        agent_state, aux_value = update_value_functions(
            observations=transition.obs,
            actions=transition.action,
            next_observations=transition.next_obs,
            rewards=transition.reward,
            dones=dones,
            agent_state=agent_state,
            recurrent=recurrent,
            gamma=gamma,
            reward_scale=reward_scale,
            expert_q=expert_q,
            target_modifier=target_modifier,
        )
        return agent_state, aux_value

    agent_state, aux_value_seq = jax.lax.scan(
        critic_update_step, agent_state, None, length=num_critic_updates
    )
    aux_value = jax.tree.map(lambda x: x[-1], aux_value_seq)

    # --- Policy update — returns log_probs for temperature reuse ---
    train_frac = agent_state.collector_state.timestep / total_timesteps
    new_agent_state, aux_policy, policy_log_probs = update_policy(
        observations=transition.obs,
        done=dones,
        agent_state=agent_state,
        recurrent=recurrent,
        raw_observations=transition.raw_obs,
        expert_policy=expert_policy,
        use_expert_guidance=use_expert_guidance,
        altitude_obs_idx=altitude_obs_idx,
        target_obs_idx=target_obs_idx,
        a_expert_precomputed=a_expert_precomputed,
        train_frac=train_frac,
        obs_preprocessor=obs_preprocessor,
        policy_action_transform=policy_action_transform,
        bc_loss_fn=bc_loss_fn,
    )
    agent_state = jax.lax.cond(
        agent_state.collector_state.timestep >= policy_update_start,
        lambda: new_agent_state,
        lambda: agent_state,
    )

    # --- Distance-modulated entropy target ---
    if target_entropy_far is not None and transition.raw_obs is not None:
        raw_distance = jnp.abs(
            transition.raw_obs[..., altitude_obs_idx]
            - transition.raw_obs[..., target_obs_idx]
        ).mean()
        distance_frac = jnp.clip(raw_distance / box_threshold, 0.0, 1.0)
        effective_target_entropy = (
            target_entropy * (1.0 - distance_frac) + target_entropy_far * distance_frac
        )
    else:
        effective_target_entropy = jnp.asarray(target_entropy)

    # --- Temperature update — reuses log_probs, no redundant actor forward pass ---
    new_agent_state_temp, aux_temperature = update_temperature(
        agent_state,
        log_probs=policy_log_probs,
        effective_target_entropy=effective_target_entropy,
    )
    agent_state = jax.lax.cond(
        agent_state.collector_state.timestep >= alpha_update_start,
        lambda: new_agent_state_temp,
        lambda: agent_state,
    )

    agent_state = update_target_networks(agent_state, tau=tau)

    # --- EDGE diagnostics (computed on training batch) ---
    edge_aux = compute_edge_diagnostics(
        aux_value.q_gap,
        aux_value.q_pred_min,
        exploration_tau,
        expert_frac_in_buffer,
    )

    # Direct field access instead of to_state_dict — avoids serialization overhead
    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=ValueAuxiliaries(
            critic_loss=aux_value.critic_loss.flatten(),
            q_pred_min=aux_value.q_pred_min.flatten(),
            q_expert_mean=aux_value.q_expert_mean.flatten(),
            q_gap=aux_value.q_gap.flatten(),
            var_preds=aux_value.var_preds.flatten(),
            alpha_blend=aux_value.alpha_blend.flatten(),
            effective_threshold=jnp.zeros(()),
            box_entry_rate=jnp.zeros(()),
            expert_frac_in_buffer=jnp.atleast_1d(expert_frac_in_buffer),
            mc_correction_frac=aux_value.mc_correction_frac.flatten(),
            phi_star_q_gap_ood=jnp.atleast_1d(phi_star_q_gap_ood),
        ),
        edge=edge_aux,
        phi_refresh=PhiRefreshAuxiliaries(
            loss_before=jnp.zeros(1),
            loss_after=jnp.zeros(1),
            expert_buffer_size=jnp.zeros(1),
        ),
    )
    return agent_state, aux


# ---------------------------------------------------------------------------
# Training iteration
# ---------------------------------------------------------------------------


def training_iteration(
    agent_state: SACState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    buffer: BufferType,
    agent_config: SACConfig,
    action_dim: int,
    total_timesteps: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: int = 1000,
    horizon: int = 10000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
    n_epochs: int = 1,
    transition_mix_fraction: float = 1.0,
    expert_policy: Optional[Callable] = None,  # used for training
    eval_expert_policy: Optional[Callable] = None,  # used for eval logging only
    use_expert_guidance: bool = True,
    action_scale: float = 1.0,
    early_termination_condition: Optional[Callable] = None,
    num_critic_updates: int = 1,
    expert_mix_fraction: float = 0.1,
    box_threshold: float = 500.0,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    augment_obs_with_expert_action: bool = False,
    detach_obs_aug_action: bool = False,
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    exploration_tau: float = 1.0,
    target_entropy_far: Optional[float] = None,
    # Composed modules (replace boolean flags)
    action_pipeline: Optional[Callable] = None,
    target_modifier: Optional[Callable] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    bc_loss_fn: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    runtime_maintenance: Optional[Callable] = None,
    # API compat
    imitation_coef: float = 0.0,
    distance_to_stable: Callable = lambda x: 1.0,
    imitation_coef_offset: float = 0.0,
) -> tuple[SACState, None]:
    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_config.learning_starts)

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        buffer=buffer,
        uniform=uniform,
        action_pipeline=action_pipeline,
    )

    agent_state, transition = collect_scan_fn(agent_state, None)
    timestep = agent_state.collector_state.timestep

    def do_update(agent_state):
        # Periodic φ* refresh via composed runtime maintenance
        _zero_phi = PhiRefreshAuxiliaries(
            loss_before=jnp.zeros(1),
            loss_after=jnp.zeros(1),
            expert_buffer_size=jnp.zeros(1),
        )
        if runtime_maintenance is not None:
            agent_state, phi_refresh_aux = runtime_maintenance(agent_state)
        else:
            phi_refresh_aux = _zero_phi

        update_scan_fn = partial(
            update_agent,
            buffer=buffer,
            recurrent=recurrent,
            gamma=agent_config.gamma,
            action_dim=action_dim,
            target_entropy=agent_config.target_entropy,
            tau=agent_config.tau,
            reward_scale=agent_config.reward_scale,
            additional_transition=(
                transition if transition_mix_fraction < 1.0 else None
            ),
            transition_mix_fraction=transition_mix_fraction,
            expert_policy=expert_policy,
            use_expert_guidance=use_expert_guidance,
            policy_update_start=policy_update_start,
            alpha_update_start=alpha_update_start,
            num_critic_updates=num_critic_updates,
            expert_mix_fraction=expert_mix_fraction,
            box_threshold=box_threshold,
            altitude_obs_idx=altitude_obs_idx,
            target_obs_idx=target_obs_idx,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            total_timesteps=total_timesteps,
            target_entropy_far=target_entropy_far,
            exploration_tau=exploration_tau,
            target_modifier=target_modifier,
            obs_preprocessor=obs_preprocessor,
            policy_action_transform=policy_action_transform,
            bc_loss_fn=bc_loss_fn,
        )
        agent_state, aux = jax.lax.scan(
            update_scan_fn, agent_state, xs=None, length=n_epochs
        )
        aux = jax.tree.map(lambda x: x[-1].reshape((1,)), aux)
        aux = aux.replace(
            value=ValueAuxiliaries(
                critic_loss=aux.value.critic_loss.flatten(),
                q_pred_min=aux.value.q_pred_min.flatten(),
                q_expert_mean=aux.value.q_expert_mean.flatten(),
                q_gap=aux.value.q_gap.flatten(),
                var_preds=aux.value.var_preds.flatten(),
                alpha_blend=aux.value.alpha_blend.flatten(),
                effective_threshold=aux.value.effective_threshold.flatten(),
                box_entry_rate=aux.value.box_entry_rate.flatten(),
                expert_frac_in_buffer=aux.value.expert_frac_in_buffer.flatten(),
                mc_correction_frac=aux.value.mc_correction_frac.flatten(),
                phi_star_q_gap_ood=aux.value.phi_star_q_gap_ood.flatten(),
            ),
            edge=EDGEAuxiliaries(
                value_gap=aux.edge.value_gap.flatten(),
                p_expert_mean=aux.edge.p_expert_mean.flatten(),
                expert_action_fraction=aux.edge.expert_action_fraction.flatten(),
            ),
            # Override the zeros from update_agent with the actual refresh diagnostics
            phi_refresh=PhiRefreshAuxiliaries(
                loss_before=phi_refresh_aux.loss_before.flatten(),
                loss_after=phi_refresh_aux.loss_after.flatten(),
                expert_buffer_size=phi_refresh_aux.expert_buffer_size.flatten(),
            ),
        )
        return agent_state, aux

    def fill_with_nan(dataclass):
        nan = jnp.ones(1) * jnp.nan
        result = {}
        for field in fields(dataclass):
            sub = field.type
            if hasattr(sub, "__dataclass_fields__"):
                result[field.name] = fill_with_nan(sub)
            else:
                result[field.name] = nan
        return dataclass(**result)

    def skip_update(agent_state):
        return agent_state, fill_with_nan(AuxiliaryLogs)

    agent_state, aux = jax.lax.cond(
        timestep >= agent_config.learning_starts,
        do_update,
        skip_update,
        operand=agent_state,
    )

    # For obs-augmented runs, wrap the actor's apply_fn so that evaluate_and_log
    # (and evaluate.py inside it) sees 12-dim obs transparently — no changes needed
    # to evaluate.py or log.py.
    if augment_obs_with_expert_action and expert_policy is not None:
        _orig_apply_fn = agent_state.actor_state.apply_fn

        def _augmented_apply_fn(params, obs, *args, **kwargs):
            _raw = obs[..., :-1]  # strip train_frac
            _a_exp = jax.lax.stop_gradient(expert_policy(_raw))
            _aug = jnp.concatenate([obs[..., :-1], _a_exp, obs[..., -1:]], axis=-1)
            return _orig_apply_fn(params, _aug, *args, **kwargs)

        _eval_agent_state = agent_state.replace(
            actor_state=agent_state.actor_state.replace(apply_fn=_augmented_apply_fn)
        )
    else:
        _eval_agent_state = agent_state

    _eval_agent_state, metrics_to_log = evaluate_and_log(
        _eval_agent_state,
        aux,
        index,
        mode,
        env_args,
        num_episode_test,
        recurrent,
        lstm_hidden_size,
        log,
        verbose,
        log_fn,
        log_frequency,
        total_timesteps,
        expert_policy=eval_expert_policy,
        action_scale=action_scale,
        early_termination_condition=early_termination_condition,
        train_frac=agent_state.collector_state.train_time_fraction,
        eval_action_transform=eval_action_transform,
    )
    # Keep the original agent_state (with original apply_fn) for training
    agent_state = agent_state.replace(
        eval_rng=_eval_agent_state.eval_rng,
        n_logs=_eval_agent_state.n_logs,
    )

    return agent_state, metrics_to_log


# ---------------------------------------------------------------------------
# Training factory
# ---------------------------------------------------------------------------


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: SACConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
    eval_expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    early_termination_condition: Optional[Callable] = None,
    residual: bool = False,
    fixed_alpha: bool = False,
    num_critics: int = 2,
    expert_buffer_n_steps: int = 20_000,
    num_critic_updates: int = 1,
    expert_mix_fraction: float = 0.1,
    box_threshold: float = 500.0,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    # MC critic pretraining (replaces Bellman pretraining)
    use_mc_critic_pretrain: bool = False,
    mc_pretrain_n_mc_steps: int = 10_000,
    mc_pretrain_n_mc_episodes: int = 100,
    mc_pretrain_n_steps: int = 5_000,
    # Online critic light pre-regression (requires MC critic pretrain)
    use_online_critic_light_pretrain: bool = True,
    online_critic_pretrain_steps: int = 500,
    online_critic_pretrain_lr_scale: float = 0.1,
    # Bellman critic pretraining (legacy fallback, mutually exclusive with MC)
    use_bellman_critic_pretrain: bool = False,
    # Expert-guided policy loss terms
    augment_obs_with_expert_action: bool = False,
    detach_obs_aug_action: bool = False,
    # Train-fraction conditioning: append timestep/total_timesteps to obs
    use_train_frac: bool = False,
    # Update start thresholds
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    # Blended Bellman target (replaces potential-based shaping)
    use_critic_blend: bool = False,
    critic_warmup_frac: float = 0.15,
    # Value-threshold box (v_min/v_max inferred from MC pretraining)
    use_box: bool = False,
    # Online decaying BC term (active for all MC pretrain runs unless disabled)
    use_online_bc: bool = True,
    bc_coef: float = 1.0,
    # EDGE (Expert Decayed Guided Exploration): stochastic expert substitution during collection
    use_expert_guided_exploration: bool = False,
    exploration_decay_frac: float = 0.30,
    exploration_tau: float = 1.0,
    exploration_boltzmann: bool = False,
    fixed_exploration_prob: float = 0.5,
    exploration_argmax: bool = False,
    # Residual RL (Johannink et al.): execute clip(a_expert + a_policy, -1, 1)
    use_residual_rl: bool = False,
    # PID policy: execute expert action directly (no actor used for env interaction)
    use_pid_policy: bool = False,
    # Distance-modulated entropy target (None = disabled)
    target_entropy_far: Optional[float] = None,
    # Online MC correction for high-variance critic states (None = disabled)
    mc_variance_threshold: Optional[float] = None,
    # Periodic self-consistent φ* refresh via expert-flagged buffer transitions
    use_phi_refresh: bool = False,
    phi_refresh_interval: int = 500,
    phi_refresh_steps: int = 20,
    # IBRL bootstrap: use max(Q_policy, Q_expert) for TD target to be consistent
    # with argmax action-selection policy (exploration_argmax=True / EDGE).
    ibrl_bootstrap: bool = False,
    # Pre-collected MC data: (obs, action, mc_return) JAX arrays.
    # When provided, the in-run expert rollout + MC-return computation is skipped.
    mc_preloaded_data: Optional[Tuple] = None,
    # PID actor: actor network predicts PID gains instead of raw actions.
    pid_actor_config=None,
    # --- Composable hook overrides (None = build from flags above) ---
    action_pipeline: Optional[Callable] = None,
    target_modifier: Optional[Callable] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    runtime_maintenance: Optional[Callable] = None,
):
    """
    SAC training factory.

    expert_policy:      used for training (warmup seeding, critic
                        pre-training). Pass None for true vanilla SAC.
    eval_expert_policy: used ONLY for eval logging (expert bias metric).
                        Always passed regardless of whether training uses expert.
                        Defaults to expert_policy if not set explicitly.
    """
    # If no separate eval policy provided, fall back to the training policy
    # (which may be None for vanilla SAC — in that case no expert bias logged)
    _eval_expert_policy = (
        eval_expert_policy if eval_expert_policy is not None else expert_policy
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    if logging_config is not None:
        start_async_logging()

    @partial(jax.jit)
    def train(key, index: Optional[int] = None):
        init_key, expert_key = jax.random.split(key)

        agent_state = init_SAC(
            key=init_key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=buffer,
            expert_policy=expert_policy,
            max_timesteps=total_timesteps if use_train_frac else None,
            num_critics=num_critics,
            expert_buffer_n_steps=(
                expert_buffer_n_steps if expert_policy is not None else 0
            ),
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            pid_actor_config=pid_actor_config,
        )

        _box_v_min = jnp.array(0.0)
        _box_v_max = jnp.array(0.0)

        if expert_policy is not None and use_mc_critic_pretrain:
            expert_critic_state = get_initialized_critic(
                key=expert_key,
                env_config=env_args,
                critic_optimizer_config=critic_optimizer_args,
                network_config=network_args,
                num_critics=num_critics,
                max_timesteps=total_timesteps if use_train_frac else None,
                extra_obs_dim=(
                    get_action_dim(env_args.env, env_args.env_params)
                    if augment_obs_with_expert_action
                    else 0
                ),
            )
            _preloaded = mc_preloaded_data  # None or (obs, action, mc) JAX arrays
            (
                agent_state,
                frozen_expert_params,
                mc_obs_batched,
                mc_action_batched,
                mc_aux,
                expert_critic_state_trained,
            ) = pretrain_critic_mc(
                agent_state=agent_state,
                expert_critic_state=expert_critic_state,
                expert_policy=expert_policy,
                mode=mode,
                env_args=env_args,
                recurrent=network_args.lstm_hidden_size is not None,
                gamma=agent_config.gamma,
                reward_scale=agent_config.reward_scale,
                n_mc_steps=mc_pretrain_n_mc_steps,
                n_mc_episodes=mc_pretrain_n_mc_episodes,
                n_steps=mc_pretrain_n_steps,
                max_timesteps=total_timesteps if use_train_frac else None,
                augment_obs_with_expert_action=augment_obs_with_expert_action,
                preloaded_obs=_preloaded[0] if _preloaded is not None else None,
                preloaded_action=_preloaded[1] if _preloaded is not None else None,
                preloaded_mc=_preloaded[2] if _preloaded is not None else None,
            )
            agent_state = agent_state.replace(
                expert_critic_params=frozen_expert_params,
                expert_v_min=mc_aux.v_min,
                expert_v_max=mc_aux.v_max,
                # Keep φ* optimizer state alive for periodic refresh (None when disabled)
                expert_critic_state=expert_critic_state_trained
                if use_phi_refresh
                else None,
            )
            if use_box:
                _box_v_min = mc_aux.v_min
                _box_v_max = mc_aux.v_max
            jax.debug.print(
                "[MC pretrain] loss: {i:.4f} -> {f:.4f}  |  "
                "Q(s,a*) mean={qm:.1f}  min={qn:.1f}  max={qx:.1f}",
                i=mc_aux.initial_loss,
                f=mc_aux.final_loss,
                qm=mc_aux.q_expert_mean,
                qn=mc_aux.q_expert_min,
                qx=mc_aux.q_expert_max,
            )

            if use_online_critic_light_pretrain:
                agent_state = pretrain_critic_online_light(
                    agent_state,
                    mc_obs_batched,
                    mc_action_batched,
                    n_steps=online_critic_pretrain_steps,
                    lr_scale=online_critic_pretrain_lr_scale,
                )
                jax.debug.print(
                    "[Online critic light pretrain] done ({n} steps, lr_scale={s})",
                    n=online_critic_pretrain_steps,
                    s=online_critic_pretrain_lr_scale,
                )

        if expert_policy is not None and use_bellman_critic_pretrain:
            agent_state = pretrain_critic_bellman(
                agent_state=agent_state,
                recurrent=network_args.lstm_hidden_size is not None,
                gamma=agent_config.gamma,
                reward_scale=agent_config.reward_scale,
                buffer=buffer,
                n_steps=mc_pretrain_n_steps,
                update_value_fn=update_value_functions,
                update_target_fn=update_target_networks,
            )
            jax.debug.print(
                "[Bellman pretrain] done ({n} steps)", n=mc_pretrain_n_steps
            )

        cloning_parameters, pre_train_n_steps = get_cloning_args(
            cloning_args, total_timesteps
        )
        if pre_train_n_steps > 0:
            agent_state = get_pre_trained_agent(
                agent_state,
                expert_policy,
                expert_key,
                env_args,
                cloning_args,
                mode,
                agent_config,
                actor_optimizer_args,
                critic_optimizer_args,
            )

        num_updates = total_timesteps // env_args.n_envs
        _, action_shape = get_state_action_shapes(env_args.env)

        _valid_cloning_params = {
            k: v
            for k, v in cloning_parameters.items()
            if k
            in (
                "n_epochs",
                "transition_mix_fraction",
                "imitation_coef",
                "distance_to_stable",
                "imitation_coef_offset",
            )
        }

        # Compose hooks: if the caller supplied an override, use it;
        # otherwise build the default from the legacy boolean flags.
        _action_pipeline = (
            action_pipeline
            if action_pipeline is not None
            else make_action_pipeline(
                expert_policy=expert_policy,
                recurrent=network_args.lstm_hidden_size is not None,
                env_args=env_args,
                use_box=use_box,
                box_v_min=_box_v_min,
                box_v_max=_box_v_max,
                use_expert_guided_exploration=use_expert_guided_exploration,
                exploration_decay_frac=exploration_decay_frac,
                exploration_tau=exploration_tau,
                exploration_boltzmann=exploration_boltzmann,
                exploration_argmax=exploration_argmax,
                fixed_exploration_prob=fixed_exploration_prob,
                use_residual_rl=use_residual_rl,
                use_pid_policy=use_pid_policy,
                augment_obs_with_expert_action=augment_obs_with_expert_action,
                total_timesteps=total_timesteps,
            )
        )

        _target_modifier = (
            target_modifier
            if target_modifier is not None
            else make_target_modifier(
                ibrl_bootstrap=ibrl_bootstrap,
                use_critic_blend=use_critic_blend,
                critic_warmup_frac=critic_warmup_frac,
                mc_variance_threshold=mc_variance_threshold,
                expert_policy=expert_policy,
                total_timesteps=total_timesteps,
                augment_obs_with_expert_action=augment_obs_with_expert_action,
                recurrent=network_args.lstm_hidden_size is not None,
            )
        )

        _obs_preprocessor = (
            obs_preprocessor
            if obs_preprocessor is not None
            else make_policy_obs_preprocessor(
                augment_obs_with_expert_action,
                detach_obs_aug_action,
                action_shape[0],
            )
        )
        _policy_action_transform = (
            policy_action_transform
            if policy_action_transform is not None
            else make_policy_action_transform(
                use_residual_rl,
                expert_policy,
            )
        )
        _bc_loss_fn = make_bc_loss_fn(
            use_online_bc,
            bc_coef,
            critic_warmup_frac,
            expert_policy,
        )

        _eval_action_transform = (
            eval_action_transform
            if eval_action_transform is not None
            else make_eval_action_transform(
                use_residual_rl=use_residual_rl,
                use_pid_policy=use_pid_policy,
            )
        )

        _runtime_maintenance = (
            runtime_maintenance
            if runtime_maintenance is not None
            else make_runtime_maintenance(
                use_phi_refresh=use_phi_refresh,
                phi_refresh_interval=phi_refresh_interval,
                phi_refresh_steps=phi_refresh_steps,
                gamma=agent_config.gamma,
                reward_scale=agent_config.reward_scale,
                expert_policy=expert_policy,
                buffer=buffer,
            )
        )

        training_iteration_scan_fn = partial(
            training_iteration,
            buffer=buffer,
            recurrent=network_args.lstm_hidden_size is not None,
            action_dim=action_shape[0],
            agent_config=agent_config,
            mode=mode,
            env_args=env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=log,
            total_timesteps=total_timesteps,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            horizon=(logging_config.horizon if logging_config is not None else None),
            expert_policy=expert_policy,
            eval_expert_policy=_eval_expert_policy,
            use_expert_guidance=use_expert_guidance,
            early_termination_condition=early_termination_condition,
            num_critic_updates=num_critic_updates,
            expert_mix_fraction=expert_mix_fraction,
            box_threshold=box_threshold,
            altitude_obs_idx=altitude_obs_idx,
            target_obs_idx=target_obs_idx,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            policy_update_start=policy_update_start,
            alpha_update_start=alpha_update_start,
            exploration_tau=exploration_tau,
            target_entropy_far=target_entropy_far,
            action_pipeline=_action_pipeline,
            target_modifier=_target_modifier,
            obs_preprocessor=_obs_preprocessor,
            policy_action_transform=_policy_action_transform,
            bc_loss_fn=_bc_loss_fn,
            eval_action_transform=_eval_action_transform,
            runtime_maintenance=_runtime_maintenance,
            **_valid_cloning_params,
        )

        agent_state, out = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=agent_state,
            xs=None,
            length=num_updates,
        )
        return agent_state, out

    return train
