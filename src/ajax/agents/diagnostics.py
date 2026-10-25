"""Training diagnostics an on-policy update logs, not part of its maths:
shared by PPO and APO."""

from typing import Any

import distrax
import jax
import jax.numpy as jnp

from ajax.networks.distributions import latent


def approximate_kl(log_ratio: jax.Array) -> jax.Array:
    """The mean of (r - 1) - log r, r the probability ratio: an
    estimate of KL(old || new) that is never negative (Schulman's "k3"), as
    CleanRL's ppo.py and SB3's PPO log it."""
    return jnp.mean((jnp.exp(log_ratio) - 1) - log_ratio)


def explained_variance(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """1 - Var[targets - predictions] / Var[targets], as CleanRL and SB3
    log it: 1 for a perfect critic, 0 for one no better than the targets'
    mean; NaN when the targets do not vary."""
    target_variance = jnp.var(targets)
    return jnp.where(
        target_variance == 0,
        jnp.nan,
        1 - jnp.var(targets - predictions) / target_variance,
    )


def mean_policy_std(pi: Any) -> jax.Array:
    """The mean standard deviation of a continuous policy's Gaussian, before
    any squashing; NaN for a discrete policy, which has none."""
    if isinstance(pi, distrax.Categorical):
        return jnp.asarray(jnp.nan)
    return latent(pi).stddev().mean()
