"""The Q-critics' regression to a held target, and its logs: shared by every
Q-critic agent (SAC and its descendants, TD3, IQL, CrossQ)."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.agents.recurrent import RecurrentCarries, q_values


@struct.dataclass
class CoreCriticAux:
    critic_loss: jax.Array
    q_pred_min: jax.Array
    var_preds: jax.Array
    q_gap: jax.Array  # mean (max - min) over the critics: |Q1 - Q2| for two


def critic_loss_fn(
    critic_params: FrozenDict,
    critic_state,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, CoreCriticAux]:
    """MSE critic loss against a pre-computed target.

    The target_q is computed by compute_td_target + optional modifiers,
    and must already be stop_gradient'd.
    """
    q_preds = q_values(
        critic_state,
        critic_params,
        observations,
        jax.lax.stop_gradient(actions),
        carries,
    )
    return q_regression_loss(q_preds, target_q)


def q_regression_loss(
    q_preds: jax.Array, target_q: jax.Array
) -> Tuple[jax.Array, CoreCriticAux]:
    """The critics' mean squared error to the held ``target_q`` and its
    logs, from their predictions ``q_preds`` (the ensemble axis first),
    however the agent computed them: :func:`critic_loss_fn`'s pass or
    CrossQ's joint one."""
    var_preds = q_preds.var(axis=0, keepdims=True)

    total_loss = jnp.mean((q_preds - target_q) ** 2)
    q_pred_min = jnp.min(q_preds, axis=0)

    return total_loss, CoreCriticAux(
        critic_loss=total_loss,
        q_pred_min=q_pred_min.mean().flatten(),
        var_preds=var_preds.mean().flatten(),
        q_gap=(jnp.max(q_preds, axis=0) - q_pred_min).mean().flatten(),
    )
