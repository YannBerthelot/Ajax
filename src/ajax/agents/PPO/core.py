"""PPO's clipped-surrogate maths (Schulman et al., 2017), which its
descendants (APO) import rather than copy. Its minibatch epochs are
:mod:`ajax.agents.minibatch`'s."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from ajax.deprecation import moved_names
from ajax.schedule import at


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_probs: jax.Array
    old_log_probs: jax.Array
    clip_fraction: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    policy_std: jax.Array
    # The loss builds these metrics before the gradient exists; the update
    # fills it in after the step.
    grad_norm: jax.Array | float = jnp.nan


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    predictions: jax.Array
    targets: jax.Array
    explained_variance: jax.Array
    grad_norm: jax.Array | float = jnp.nan  # filled in after the step, as the policy's


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics


def clipped_surrogate(
    ratio: jax.Array, advantages: jax.Array, clip_coef: float
) -> tuple[jax.Array, jax.Array]:
    """The per-sample clipped surrogate loss ``-min(r A, clip(r) A)`` and
    the fraction of samples whose ratio is clipped."""
    loss = -jnp.minimum(
        ratio * advantages,
        jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages,
    )
    return loss, (jnp.abs(ratio - 1) > clip_coef).mean()


def resolve_clip_coef(agent_config: Any, timestep: int | jax.Array) -> Any:
    """The clip range, a constant or a schedule of the timestep."""
    return at(agent_config.clip_range, timestep)


__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
        ("resolve_num_minibatches", "run_epochs", "rollout_actions"),
        "ajax.agents.minibatch",
    )
    | dict.fromkeys(
        ("approximate_kl", "explained_variance", "mean_policy_std"),
        "ajax.agents.diagnostics",
    )
    | dict.fromkeys(
        ("policy_entropy", "recompute_log_prob"), "ajax.networks.distributions"
    ),
)
