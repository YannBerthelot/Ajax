"""Exploration strategies for SAC — composable, stackable.

EDGE (Expert Decayed Guided Exploration): value-gap-gated expert action
substitution during collection, decaying over training.

Three gating modes (compose at init time by selecting the gate function):
    edge_argmax_gate:    deterministic — use expert when Q(s,pi*) > Q(s,pi)
    edge_boltzmann_gate: adaptive — p = decay * sigmoid(gap / (tau * |Q|))
    edge_fixed_gate:     constant — p = decay * fixed_prob

Box (value-threshold): override with expert action when V*(s) > threshold,
with curriculum that raises the threshold over training.

Both compose at the call site in collect_experience:
    action = policy_action
    action = edge_override(action, ...)        # optional
    action = box_action_override(action, ...)   # optional
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from ajax.networks.networks import predict_value

# ---------------------------------------------------------------------------
# EDGE — Expert Decayed Guided Exploration
# ---------------------------------------------------------------------------


def edge_compute_value_gap(
    obs: jax.Array,
    policy_action: jax.Array,
    expert_action: jax.Array,
    critic_state,
    critic_params,
) -> Tuple[jax.Array, jax.Array]:
    """Compute Q(s,pi*) - Q(s,pi).  Positive means expert is still better.

    Returns (gap, q_policy) — q_policy is reused by the Boltzmann gate.
    """
    q_policy = jnp.min(
        predict_value(
            critic_state=critic_state,
            critic_params=critic_params,
            x=jnp.concatenate([obs, policy_action], axis=-1),
        ),
        axis=0,
    )
    q_expert = jnp.min(
        predict_value(
            critic_state=critic_state,
            critic_params=critic_params,
            x=jnp.concatenate([obs, expert_action], axis=-1),
        ),
        axis=0,
    )
    return q_expert - q_policy, q_policy


def edge_compute_decay(
    timestep: jax.Array,
    total_timesteps: int,
    decay_frac: float,
) -> jax.Array:
    """Linear decay: 1 -> 0 over decay_frac of training.

    decay_frac=0.0 means never decay (gate stays active for full training).
    """
    if decay_frac == 0.0:
        return jnp.ones(())
    train_frac = timestep / total_timesteps
    return jnp.maximum(1.0 - train_frac / decay_frac, 0.0)


def edge_argmax_gate(
    gap: jax.Array,
    decay: jax.Array,
    rng: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Deterministic: use expert whenever gap > 0 and decay > 0.

    Returns (use_expert_mask, unchanged_rng).
    """
    return (gap > 0.0) & (decay > 0.0), rng


def edge_boltzmann_gate(
    gap: jax.Array,
    decay: jax.Array,
    rng: jax.Array,
    q_policy: jax.Array,
    tau: float,
) -> Tuple[jax.Array, jax.Array]:
    """Adaptive: p = decay * sigmoid(gap / (tau * |Q|)).

    Returns (use_expert_mask, updated_rng).
    """
    q_scale = jax.lax.stop_gradient(jnp.abs(q_policy).mean() + 1e-6)
    p_expert = decay * jax.nn.sigmoid(gap / (tau * q_scale))
    rng, key = jax.random.split(rng)
    return jax.random.uniform(key, shape=p_expert.shape) < p_expert, rng


def edge_fixed_gate(
    gap: jax.Array,
    decay: jax.Array,
    rng: jax.Array,
    fixed_prob: float,
) -> Tuple[jax.Array, jax.Array]:
    """Fixed probability: p = decay * fixed_prob.

    Returns (use_expert_mask, updated_rng).
    """
    p = decay * fixed_prob
    rng, key = jax.random.split(rng)
    return jax.random.uniform(key, shape=p.shape) < p, rng


# ---------------------------------------------------------------------------
# Box — value-threshold expert override
# ---------------------------------------------------------------------------


def box_compute_threshold(
    v_min: jax.Array,
    v_max: jax.Array,
    train_frac: jax.Array,
) -> jax.Array:
    """Curriculum threshold: v_min + (v_max - v_min) * train_frac."""
    return v_min + (v_max - v_min) * train_frac


def box_compute_state(
    obs: jax.Array,
    raw_obs: jax.Array,
    expert_policy,
    critic_state,
    expert_critic_params,
    threshold: jax.Array,
    last_in_box: Optional[jax.Array],
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute box membership and entry bonus.

    Returns (in_box, entry_bonus, v_box).
    """
    a_exp = jax.lax.stop_gradient(expert_policy(raw_obs))
    v_box = jnp.min(
        predict_value(
            critic_state=critic_state,
            critic_params=expert_critic_params,
            x=jnp.concatenate([obs, a_exp], axis=-1),
        ),
        axis=0,
    )
    in_box = v_box > threshold

    if last_in_box is None:
        last_in_box = jnp.zeros_like(in_box)

    entry_bonus = jnp.where(
        (last_in_box < 0.5) & (in_box > 0.5),
        v_box,
        jnp.zeros_like(v_box),
    )
    return in_box, entry_bonus, v_box


def box_action_override(
    action: jax.Array,
    expert_action: jax.Array,
    in_box: jax.Array,
) -> jax.Array:
    """Override with expert action inside the value box."""
    return jnp.where(in_box, expert_action, action)


def box_modify_reward_done(
    reward: jax.Array,
    terminated: jax.Array,
    entry_bonus: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Add entry bonus to reward and mark as terminal on entry."""
    reward = reward + entry_bonus[..., 0]
    terminated = jnp.logical_or(
        terminated.astype(bool), entry_bonus[..., 0] > 0
    ).astype(terminated.dtype)
    return reward, terminated


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@struct.dataclass
class EDGEAuxiliaries:
    """EDGE diagnostics computed on the training batch."""

    value_gap: jax.Array  # Q(s,pi*) - Q(s,pi): gate signal
    p_expert_mean: jax.Array  # sigmoid(gap / (tau*|Q_pi|)): Boltzmann p before decay
    expert_action_fraction: jax.Array  # fraction of batch with is_expert=1


def compute_edge_diagnostics(
    q_gap: jax.Array,
    q_pred_min: jax.Array,
    exploration_tau: float,
    expert_frac_in_buffer: jax.Array,
) -> EDGEAuxiliaries:
    """EDGE diagnostics on the training batch (same distribution as collection)."""
    q_scale = jax.lax.stop_gradient(jnp.abs(q_pred_min).mean() + 1e-6)
    p_expert = jax.nn.sigmoid(q_gap / (exploration_tau * q_scale))
    return EDGEAuxiliaries(
        value_gap=q_gap.flatten(),
        p_expert_mean=jnp.atleast_1d(p_expert.mean()),
        expert_action_fraction=jnp.atleast_1d(expert_frac_in_buffer),
    )
