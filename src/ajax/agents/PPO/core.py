"""PPO's clipped-surrogate maths and minibatch epochs (Schulman et al.,
2017), which its descendants (APO) import rather than copy."""

from typing import Any, Callable, Optional

import distrax
import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents.SAC.utils import SquashedNormal
from ajax.state import Transition


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_probs: jax.Array
    old_log_probs: jax.Array
    clip_fraction: jax.Array
    entropy: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    predictions: jax.Array
    targets: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


def rollout_actions(rollout: Transition) -> tuple[jax.Array, jax.Array]:
    """The rollout's actions and their log-probabilities, each with a
    trailing axis: a discrete action gains one, a continuous action's
    per-dimension log-probabilities are summed."""
    assert rollout.log_prob is not None  # an on-policy rollout carries it
    action, log_prob = rollout.action, rollout.log_prob
    actions = jnp.expand_dims(action, -1) if jnp.ndim(action) < 3 else action
    log_probs = (
        jnp.expand_dims(log_prob, -1)
        if jnp.ndim(log_prob) < 3
        else log_prob.sum(-1, keepdims=True)
    )
    return actions, log_probs


def recompute_log_prob(
    pi: Any, actions: jax.Array, raw_actions: Optional[jax.Array]
) -> jax.Array:
    """The current policy's log-probability of the stored actions, shaped
    like :func:`rollout_actions`'s. A squashed policy recomputes it from the
    stored pre-tanh sample, never through arctanh, which is unstable as
    ``|action| -> 1`` (:meth:`SquashedNormal.log_prob_from_raw`)."""
    if isinstance(pi, distrax.Categorical):
        return jnp.expand_dims(pi.log_prob(actions.squeeze(-1)), -1)
    if isinstance(pi, SquashedNormal) and raw_actions is not None:
        return pi.log_prob_from_raw(raw_actions)
    return pi.log_prob(actions).sum(-1, keepdims=True)


def policy_entropy(pi: Any, rng: Optional[jax.Array] = None) -> jax.Array:
    """The per-sample entropy bonus. A squashed policy's is, with ``rng``,
    brax's one-sample estimate of the executed action's entropy (the latent
    Gaussian's plus the tanh log-det-Jacobian at a sample, so its gradient
    reaches the mean); without, the latent Gaussian's."""
    if isinstance(pi, SquashedNormal):
        if rng is not None:
            return pi.effective_entropy(rng, num_samples=1)
        return pi.unsquashed_entropy()
    return pi.entropy()


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


def resolve_num_minibatches(agent_config: Any) -> int:
    """``num_minibatches`` when positive (brax); otherwise Ajax's legacy
    ``max(batch_size, n_steps) // min(batch_size, n_steps)``."""
    if agent_config.num_minibatches > 0:
        return agent_config.num_minibatches
    sizes = (agent_config.batch_size, agent_config.n_steps)
    return max(sizes) // min(sizes)


def resolve_clip_coef(agent_config: Any, timestep: jax.Array) -> Any:
    """The clip range, a constant or a schedule of the timestep."""
    clip_range = agent_config.clip_range
    return clip_range(timestep) if callable(clip_range) else clip_range


def run_epochs(
    agent_state: Any,
    minibatches: Any,
    step: Callable[[Any, Any], tuple[Any, Any]],
    n_epochs: int,
) -> tuple[Any, Any]:
    """``n_epochs`` passes of ``step(agent_state, minibatch) -> (agent_state,
    aux)`` over ``minibatches`` (minibatch axis leading); the metrics
    averaged over every epoch and minibatch."""

    def epoch(agent_state: Any, _: Any) -> tuple[Any, Any]:
        return jax.lax.scan(step, agent_state, minibatches)

    agent_state, aux = jax.lax.scan(epoch, agent_state, None, length=n_epochs)
    return agent_state, jax.tree.map(jnp.mean, aux)
