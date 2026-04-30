"""Expert-guidance modifiers for SAC — composable, stackable.

Each function is a standalone modifier that transforms inputs/outputs of
the core SAC algorithm. They compose at the call site:

Critic-side target modifiers:
    target_q = core.compute_td_target(...)
    target_q = ibrl_modify_target(target_q, ...)      # optional
    target_q = blend_modify_target(target_q, ...)[0]   # optional
    target_q = mc_correction_modify_target(target_q, ...)[0]  # optional

Policy-side modifiers:
    obs = detach_obs_expert_dims(obs, ...)             # optional
    actions = residual_action_transform(actions, ...)  # optional
    bc_loss = compute_online_bc_loss(...)              # optional additive

Diagnostics:
    compute_expert_diagnostics(...)
    compute_behavior_kpis(...)

Value-threshold box:
    see exploration.py
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from ajax.networks.networks import predict_value

# ---------------------------------------------------------------------------
# Critic-side target modifiers
# ---------------------------------------------------------------------------


def ibrl_modify_target(
    target_q: jax.Array,
    min_q_target_from_core: jax.Array,
    critic_state,
    next_observations: jax.Array,
    next_expert_actions: jax.Array,
) -> jax.Array:
    """IBRL: max(Q_policy, Q_expert) so value function matches argmax policy."""
    q_targets_expert = predict_value(
        critic_state=critic_state,
        critic_params=critic_state.target_params,
        x=jnp.concatenate((next_observations, next_expert_actions), axis=-1),
    )
    min_q_target_expert = jnp.min(q_targets_expert, axis=0, keepdims=False)
    gap = jnp.maximum(min_q_target_expert - min_q_target_from_core, 0.0)
    return target_q + gap


def blend_modify_target(
    target_q: jax.Array,
    v_expert_next: jax.Array,
    alpha_blend: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Blended Bellman: (1-alpha)*y_bellman + alpha*V*(s')."""
    blended = (1.0 - alpha_blend) * target_q + alpha_blend * v_expert_next
    return blended, alpha_blend.mean().flatten()


def mc_correction_modify_target(
    target_q: jax.Array,
    critic_state,
    critic_params_mc,
    observations: jax.Array,
    actions: jax.Array,
    q_var: jax.Array,
    mc_variance_threshold: float,
) -> Tuple[jax.Array, jax.Array]:
    """Replace high-variance Bellman targets with MC-pretrained oracle estimate."""
    uncertain_mask = q_var > mc_variance_threshold
    mc_correction_frac = uncertain_mask.mean().reshape(1)
    q_mc_target = jnp.min(
        predict_value(
            critic_state=critic_state,
            critic_params=critic_params_mc,
            x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
        ),
        axis=0,
    )
    target_q = jnp.where(
        uncertain_mask[..., None],
        jax.lax.stop_gradient(q_mc_target),
        target_q,
    )
    return target_q, mc_correction_frac


# ---------------------------------------------------------------------------
# Policy-side modifiers
# ---------------------------------------------------------------------------


def augment_obs_if_needed(
    observations: jax.Array,
    raw_observations: jax.Array,
    expert_policy,
    augment: bool,
) -> jax.Array:
    """Append expert action to obs at runtime. Layout: [env_obs | a_expert | train_frac]."""
    if not augment or expert_policy is None:
        return observations
    a_expert = jax.lax.stop_gradient(expert_policy(raw_observations))
    return jnp.concatenate(
        [observations[..., :-1], a_expert, observations[..., -1:]], axis=-1
    )


def detach_obs_expert_dims(
    observations: jax.Array,
    action_dim: int,
) -> jax.Array:
    """Stop-gradient through expert-action dims in augmented obs.

    Layout: [env_obs | a_expert | train_frac].
    The actor can read the expert hint but gradients don't flow through it.
    """
    return jnp.concatenate(
        [
            observations[..., : -(action_dim + 1)],
            jax.lax.stop_gradient(observations[..., -(action_dim + 1) : -1]),
            observations[..., -1:],
        ],
        axis=-1,
    )


def residual_action_transform(
    actions: jax.Array,
    a_expert: jax.Array,
    scale: float = 1.0,
) -> jax.Array:
    """Residual RL: executed action is clip(a_expert + scale * a_pi, -1, 1).

    The critic was trained on (s, executed_action) tuples, so the policy
    gradient must flow through Q(s, executed_action), not Q(s, a_pi).
    ``scale`` follows Johannink et al. 2019; values < 1 keep the initial
    behaviour close to the expert when the policy is randomly initialised.
    """
    return jnp.clip(a_expert + scale * actions, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def compute_expert_diagnostics(
    critic_state,
    observations: jax.Array,
    q_min: jax.Array,
    a_expert: jax.Array,
    pi_loc: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Expert policy diagnostics: Q(s, a_expert), ||pi - a_expert||^2, above-expert fraction.

    Returns (q_expert_mean, l2_expert_mean, above_expert_frac).
    """
    q_expert = jnp.min(
        predict_value(
            critic_state=critic_state,
            critic_params=critic_state.params,
            x=jnp.concatenate([observations, a_expert], axis=-1),
        ),
        axis=0,
    )
    above_expert_frac = jnp.mean((q_min >= q_expert).astype(jnp.float32))
    l2_expert = jnp.sum((jnp.tanh(pi_loc) - a_expert) ** 2, axis=-1, keepdims=True)
    return q_expert.mean(), l2_expert.mean(), above_expert_frac


def compute_online_bc_loss(
    pi_loc: jax.Array,
    a_expert: jax.Array,
    critic_state,
    expert_critic_params,
    observations: jax.Array,
    train_frac: jax.Array,
    critic_warmup_frac: float,
    expert_v_min: jax.Array,
    expert_v_max: jax.Array,
    bc_coef: float,
) -> jax.Array:
    """Online decaying BC term — value-weighted, warmup-decaying.

    Decays to zero after train_frac exceeds critic_warmup_frac.
    Value-weighting ensures BC is strongest in high-value states.
    """
    bc_weight = jnp.maximum(1.0 - train_frac / critic_warmup_frac, 0.0)
    v_star = jax.lax.stop_gradient(
        jnp.min(
            predict_value(
                critic_state=critic_state,
                critic_params=expert_critic_params,
                x=jnp.concatenate([observations, a_expert], axis=-1),
            ),
            axis=0,
        )
    )
    v_weight = (
        jnp.clip(
            (v_star - expert_v_min) / (expert_v_max - expert_v_min + 1e-6),
            0.0,
            1.0,
        )
        ** 2
    )
    mu = jnp.tanh(pi_loc)
    return (
        bc_coef
        * bc_weight
        * (
            v_weight
            * jnp.sum(
                (mu - jax.lax.stop_gradient(a_expert)) ** 2,
                axis=-1,
                keepdims=True,
            )
        ).mean()
    )


def compute_behavior_kpis(
    raw_obs: jax.Array,
    altitude_obs_idx: int,
    target_obs_idx: int,
) -> Tuple[jax.Array, jax.Array]:
    """Behavior KPIs from raw env observations.

    Returns (altitude_error, z_dot_mean).
    """
    altitude_error = jnp.abs(
        raw_obs[..., altitude_obs_idx] - raw_obs[..., target_obs_idx]
    ).mean()
    z_dot_mean = jnp.abs(raw_obs[..., 2]).mean()
    return altitude_error, z_dot_mean
