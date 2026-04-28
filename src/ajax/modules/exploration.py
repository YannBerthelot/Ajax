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
# Quality-aware (LCB) gate — uses inter-critic disagreement to weight choice
# ---------------------------------------------------------------------------


def edge_compute_lcb_scores(
    obs: jax.Array,
    policy_action: jax.Array,
    expert_action: jax.Array,
    critic_state,
    critic_params,
    beta: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Lower-confidence-bound scores for each candidate action.

    Returns (score_expert, score_policy, q_policy_mean, mu_actor,
    mu_expert, sigma_actor, sigma_expert).

    score(a) = Q_min(s,a) - beta * (Q_max(s,a) - Q_min(s,a))

    With twin (or n) critics, Q_max - Q_min approximates epistemic
    disagreement. LCB penalises uncertain candidates: low-disagreement
    (well-estimated) actions get scored close to their min-Q; high-
    disagreement (OOD) actions get a hefty penalty. The action_pipeline
    consumer typically gates on (score_e > score_p), optionally
    stochastically via a softmax over the difference.

    The trailing four returns are diagnostic per-batch summaries
    (mean/std of the critic ensemble for each candidate action) used
    for live telemetry — they are not consumed by the gate itself.
    """
    q_e = predict_value(
        critic_state=critic_state,
        critic_params=critic_params,
        x=jnp.concatenate([obs, expert_action], axis=-1),
    )
    q_p = predict_value(
        critic_state=critic_state,
        critic_params=critic_params,
        x=jnp.concatenate([obs, policy_action], axis=-1),
    )
    q_min_e = jnp.min(q_e, axis=0)
    q_max_e = jnp.max(q_e, axis=0)
    q_min_p = jnp.min(q_p, axis=0)
    q_max_p = jnp.max(q_p, axis=0)
    score_e = q_min_e - beta * (q_max_e - q_min_e)
    score_p = q_min_p - beta * (q_max_p - q_min_p)
    # Diagnostics (mean/std across the critic ensemble axis 0).
    mu_p = jnp.mean(q_p, axis=0)
    mu_e = jnp.mean(q_e, axis=0)
    sigma_p = jnp.std(q_p, axis=0)
    sigma_e = jnp.std(q_e, axis=0)
    return score_e, score_p, q_min_p, mu_p, mu_e, sigma_p, sigma_e


def edge_lcb_gate(
    score_expert: jax.Array,
    score_policy: jax.Array,
    rng: jax.Array,
    temperature: float,
) -> Tuple[jax.Array, jax.Array]:
    """Stochastic Boltzmann gate over LCB scores.

    p(use_expert) = sigmoid((score_e - score_p) / temperature).

    Hard threshold (argmax) creates a self-fulfilling expert preference:
    expert always wins → buffer all expert → critic never learns Q for
    policy actions → expert keeps winning. The stochastic version ensures
    policy occasionally gets sampled, breaking the inertia.

    Temperature → 0 recovers argmax; → ∞ recovers uniform mixing.
    """
    delta = (score_expert - score_policy) / jnp.maximum(temperature, 1e-6)
    p_expert = jax.nn.sigmoid(delta)
    rng, key = jax.random.split(rng)
    use_expert = jax.random.uniform(key, shape=p_expert.shape) < p_expert
    return use_expert, rng


def edge_compute_thompson_stats(
    obs: jax.Array,
    policy_action: jax.Array,
    expert_action: jax.Array,
    critic_state,
    critic_params,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-action Q mean / std across critic ensemble.

    Returns (mu_e, sigma_e, mu_p, sigma_p, mu_p) where the last item keeps
    the call-signature parity with edge_compute_lcb_scores (q_policy summary
    consumed downstream as the critic-side training signal).
    """
    q_e = predict_value(
        critic_state=critic_state,
        critic_params=critic_params,
        x=jnp.concatenate([obs, expert_action], axis=-1),
    )
    q_p = predict_value(
        critic_state=critic_state,
        critic_params=critic_params,
        x=jnp.concatenate([obs, policy_action], axis=-1),
    )
    mu_e = jnp.mean(q_e, axis=0)
    sigma_e = jnp.std(q_e, axis=0)
    mu_p = jnp.mean(q_p, axis=0)
    sigma_p = jnp.std(q_p, axis=0)
    return mu_e, sigma_e, mu_p, sigma_p, mu_p


def edge_thompson_gate(
    mu_e: jax.Array,
    sigma_e: jax.Array,
    mu_p: jax.Array,
    sigma_p: jax.Array,
    rng: jax.Array,
    temperature: float = 1.0,
    epsilon_floor: float = 0.0,
) -> Tuple[jax.Array, jax.Array]:
    """Thompson-sampling gate, with optional epsilon floor on policy picks.

    Treat each candidate's Q as Gaussian(mu, (temperature * sigma)^2),
    draw one sample per side, pick whichever is larger. Equivalent to
    p(use_expert) = Phi((mu_e - mu_p) / (T * sqrt(sigma_e^2 + sigma_p^2)))
    in expectation, but stochastic per-step (which is what we want for
    exploration: the gate itself injects noise rather than deferring to a
    fixed rule).

    Symmetric: high uncertainty on either side widens the gate, never
    biasing it. Confident estimates dominate. With both sigmas → 0 we
    recover deterministic argmax.

    epsilon_floor: with this probability, force the policy regardless of
    the Thompson outcome. Required when the expert dominates by many
    sigmas (Phi -> 1 -> Thompson never picks the policy -> the critic
    never sees policy actions in its updates -> Q for policy actions
    stays stale -> the gate can never flip even if the policy improves).
    Set carefully on brittle envs: too high crashes the agent, too low
    starves the policy of evaluation data. ~0.001-0.05 is a sane range.
    """
    rng, key_e, key_p, key_floor = jax.random.split(rng, 4)
    scale = jnp.maximum(temperature, 1e-6)
    q_tilde_e = mu_e + scale * sigma_e * jax.random.normal(key_e, mu_e.shape)
    q_tilde_p = mu_p + scale * sigma_p * jax.random.normal(key_p, mu_p.shape)
    use_expert_thompson = q_tilde_e > q_tilde_p
    if epsilon_floor > 0.0:
        force_policy = jax.random.uniform(key_floor, mu_e.shape) < epsilon_floor
        use_expert = jnp.where(force_policy, jnp.zeros_like(use_expert_thompson),
                               use_expert_thompson)
    else:
        use_expert = use_expert_thompson
    return use_expert, rng


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
    # Live gating telemetry (per-step batch means from the latest
    # collect_experience call, NOT from the replay batch). Useful for
    # studying gate dynamics over training; NaN for vanilla SAC.
    live_expert_frac: jax.Array
    live_q_advantage: jax.Array       # mean(mu_actor - mu_expert), LCB/Thompson only
    live_critic_sigma_actor: jax.Array
    live_critic_sigma_expert: jax.Array


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
        # Inner-update diag returns NaN; the outer aggregator overwrites
        # these with the actual values from collector_state.last_*.
        live_expert_frac=jnp.array([jnp.nan]),
        live_q_advantage=jnp.array([jnp.nan]),
        live_critic_sigma_actor=jnp.array([jnp.nan]),
        live_critic_sigma_expert=jnp.array([jnp.nan]),
    )
