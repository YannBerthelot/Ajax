"""Pure SAC algorithm — no expert guidance, no exploration, no pretraining.

This module contains the core Soft Actor-Critic functions that compose
the base algorithm. Expert features (IBRL, critic blend, online BC,
residual RL, obs augmentation) are layered on top in train_SAC.py.

Design principle: "compose at init time, not JIT time."
  - Core functions take pre-computed values (e.g. target_q), not feature flags.
  - Expert modules modify inputs/outputs of these core functions.
  - No static_argnames for feature selection — eliminates recompilation.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from ajax.agents.SAC.utils import SquashedNormal
from ajax.environments.interaction import get_pi
from ajax.networks.networks import predict_value

# ---------------------------------------------------------------------------
# Auxiliary dataclasses (core diagnostics only)
# ---------------------------------------------------------------------------


@struct.dataclass
class CoreCriticAux:
    critic_loss: jax.Array
    q_pred_min: jax.Array
    var_preds: jax.Array


@struct.dataclass
class CoreActorAux:
    policy_loss: jax.Array
    log_pi: jax.Array
    policy_std: jax.Array
    q_min: jax.Array


@struct.dataclass
class CoreTemperatureAux:
    alpha: jax.Array
    log_alpha: jax.Array
    effective_target_entropy: jax.Array


# ---------------------------------------------------------------------------
# Temperature parameter
# ---------------------------------------------------------------------------


def create_alpha_train_state(
    learning_rate: float = 3e-4,
    alpha_init: float = 1.0,
) -> TrainState:
    log_alpha = jnp.log(alpha_init)
    params = FrozenDict({"log_alpha": log_alpha})
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate),
    )
    return TrainState.create(
        apply_fn=lambda params: jnp.exp(params["log_alpha"]),
        params=params,
        tx=tx,
    )


# ---------------------------------------------------------------------------
# TD target computation (pure Bellman — modifiers compose on top)
# ---------------------------------------------------------------------------


def compute_td_target(
    actor_state,
    critic_state,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    rng: jax.Array,
    recurrent: bool,
    reward_scale: float = 1.0,
) -> jax.Array:
    """Pure SAC Bellman target: r + γ(1-d)(min Q_target(s', π(s')) - α log π).

    Returns stop-gradient'd target. Expert modules can modify this
    (IBRL, critic blend, MC correction) before it enters the critic loss.
    """
    rewards = rewards * reward_scale

    next_pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_state.params,
        obs=next_observations,
        done=dones,
        recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    next_actions, log_probs = next_pi.sample_and_log_prob(seed=sample_key)
    log_probs = log_probs.sum(-1, keepdims=True)

    q_targets = predict_value(
        critic_state=critic_state,
        critic_params=critic_state.target_params,
        x=jnp.concatenate((next_observations, next_actions), axis=-1),
    )
    min_q_target = jnp.min(q_targets, axis=0, keepdims=False)

    target = rewards + gamma * (1.0 - dones) * (min_q_target - alpha * log_probs)
    return jax.lax.stop_gradient(target)


# ---------------------------------------------------------------------------
# Critic loss (MSE against pre-computed target)
# ---------------------------------------------------------------------------


def critic_loss_fn(
    critic_params: FrozenDict,
    critic_state,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
) -> Tuple[jax.Array, CoreCriticAux]:
    """MSE critic loss against a pre-computed target.

    The target_q is computed by compute_td_target + optional modifiers,
    and must already be stop_gradient'd.
    """
    q_preds = predict_value(
        critic_state=critic_state,
        critic_params=critic_params,
        x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
    )
    var_preds = q_preds.var(axis=0, keepdims=True)

    total_loss = jnp.mean((q_preds - target_q) ** 2)
    q_pred_min = jnp.min(q_preds, axis=0)

    return total_loss, CoreCriticAux(
        critic_loss=total_loss,
        q_pred_min=q_pred_min.mean().flatten(),
        var_preds=var_preds.mean().flatten(),
    )


# ---------------------------------------------------------------------------
# Actor loss (pure SAC: α log π - Q)
# ---------------------------------------------------------------------------


def actor_loss_fn(
    actor_params: FrozenDict,
    actor_state,
    critic_state,
    observations: jax.Array,
    dones: Optional[jax.Array],
    alpha: jax.Array,
    rng: jax.Array,
    recurrent: bool,
) -> Tuple[jax.Array, Tuple[CoreActorAux, jax.Array, jax.Array]]:
    """Pure SAC actor loss: α·log π(a|s) - min Q(s, a).

    Returns (loss, (aux, actions, log_probs)).
    actions and log_probs are exposed so expert modules (residual RL,
    online BC) can use them without recomputing the forward pass.
    """
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=observations,
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

    q_preds = predict_value(
        critic_state=critic_state,
        critic_params=critic_state.params,
        x=jnp.concatenate([observations, actions], axis=-1),
    )
    q_min = jnp.min(q_preds, axis=0)
    loss = (alpha * log_probs - q_min).mean()

    aux = CoreActorAux(
        policy_loss=loss,
        log_pi=log_probs.mean(),
        policy_std=policy_std,
        q_min=q_min.mean(),
    )
    return loss, (aux, actions, log_probs)


# ---------------------------------------------------------------------------
# Temperature loss (standard SAC dual gradient)
# ---------------------------------------------------------------------------


def temperature_loss_fn(
    log_alpha_params: FrozenDict,
    log_probs: jax.Array,
    target_entropy: jax.Array,
) -> Tuple[jax.Array, CoreTemperatureAux]:
    """Standard SAC temperature loss: log(α) · (-log π - H_target)."""
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    loss = (log_alpha * jax.lax.stop_gradient(-log_probs - target_entropy)).mean()
    return loss, CoreTemperatureAux(
        alpha=alpha,
        log_alpha=log_alpha,
        effective_target_entropy=target_entropy,
    )


# ---------------------------------------------------------------------------
# Target network soft update
# ---------------------------------------------------------------------------


def soft_update_target_params(params, target_params, tau: float):
    """Polyak averaging: target ← τ·params + (1-τ)·target."""
    return jax.tree.map(
        lambda p, tp: tau * p + (1.0 - tau) * tp,
        params,
        target_params,
    )
