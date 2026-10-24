"""PPO's clipped-surrogate maths (Schulman et al., 2017), which its
descendants (APO) import rather than copy. Its minibatch epochs are
:mod:`ajax.agents.minibatch`'s."""

from typing import Any, Optional

import distrax
import jax
import jax.numpy as jnp
from flax import struct

from ajax.deprecation import moved_names
from ajax.networks.distributions import entropy, latent, log_prob_from_raw
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


def recompute_log_prob(
    pi: Any, actions: jax.Array, raw_actions: Optional[jax.Array]
) -> jax.Array:
    """The current policy's log-probability of the stored actions, shaped
    like :func:`ajax.agents.minibatch.rollout_actions`'s. A squashed policy recomputes it from the
    stored pre-tanh sample, never through arctanh, which is unstable as
    ``|action| -> 1`` (:func:`log_prob_from_raw`); an unsquashed one's
    pre-tanh sample is its action."""
    if isinstance(pi, distrax.Categorical):
        return jnp.expand_dims(pi.log_prob(actions.squeeze(-1)), -1)
    if raw_actions is not None:
        return log_prob_from_raw(pi, raw_actions).sum(-1, keepdims=True)
    return pi.log_prob(actions).sum(-1, keepdims=True)


def policy_entropy(pi: Any, rng: Optional[jax.Array] = None) -> jax.Array:
    """The per-sample entropy bonus: the joint action's entropy, summed over
    the action dimensions as CleanRL's ``ppo_continuous_action.py`` and
    baselines' ppo2 sum it (a mean would scale ``ent_coef`` by 1 / d). A
    squashed policy's is sampled with ``rng`` (:meth:`Squash.entropy`)."""
    if isinstance(pi, distrax.Categorical):
        return pi.entropy()
    return entropy(pi, rng).sum(-1)


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


def approximate_kl(log_ratio: jax.Array) -> jax.Array:
    """The mean of ``(r - 1) - log r``, ``r`` the probability ratio: an
    estimate of KL(old || new) that is never negative (Schulman's "k3"), as
    CleanRL's ``ppo.py`` and SB3's PPO log it."""
    return jnp.mean((jnp.exp(log_ratio) - 1) - log_ratio)


def explained_variance(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """``1 - Var[targets - predictions] / Var[targets]``, as CleanRL and SB3
    log it: 1 for a perfect critic, 0 for one no better than the targets'
    mean; NaN when the targets do not vary."""
    var_targets = jnp.var(targets)
    return jnp.where(
        var_targets == 0, jnp.nan, 1 - jnp.var(targets - predictions) / var_targets
    )


def mean_policy_std(pi: Any) -> jax.Array:
    """The mean standard deviation of a continuous policy's Gaussian, before
    any squashing; NaN for a discrete policy, which has none."""
    if isinstance(pi, distrax.Categorical):
        return jnp.asarray(jnp.nan)
    return latent(pi).stddev().mean()


def resolve_clip_coef(agent_config: Any, timestep: int | jax.Array) -> Any:
    """The clip range, a constant or a schedule of the timestep."""
    return at(agent_config.clip_range, timestep)


__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
        ("resolve_num_minibatches", "run_epochs", "rollout_actions"),
        "ajax.agents.minibatch",
    ),
)
