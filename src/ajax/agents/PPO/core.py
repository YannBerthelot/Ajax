"""PPO's clipped-surrogate maths and minibatch epochs (Schulman et al.,
2017), which its descendants (APO) import rather than copy."""

from typing import Any, Callable, Optional

import distrax
import jax
import jax.numpy as jnp
from flax import struct

from ajax.networks.distributions import entropy, log_prob_from_raw
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


def resolve_num_minibatches(agent_config: Any) -> int:
    """``num_minibatches`` when positive (brax); otherwise Ajax's legacy
    ``max(batch_size, n_steps) // min(batch_size, n_steps)``."""
    if agent_config.num_minibatches > 0:
        return agent_config.num_minibatches
    sizes = (agent_config.batch_size, agent_config.n_steps)
    return max(sizes) // min(sizes)


def resolve_clip_coef(agent_config: Any, timestep: int | jax.Array) -> Any:
    """The clip range, a constant or a schedule of the timestep."""
    clip_range = agent_config.clip_range
    return clip_range(timestep) if callable(clip_range) else clip_range


def run_epochs(
    agent_state: Any,
    key: jax.Array,
    minibatches: Callable[[jax.Array], Any],
    step: Callable[[Any, Any], tuple[Any, Any]],
    n_epochs: int,
) -> tuple[Any, Any]:
    """``n_epochs`` passes of ``step(agent_state, minibatch) -> (agent_state,
    aux)``, each over its own partition ``minibatches(epoch_key)`` of the
    rollout (minibatch axis leading); the metrics averaged over every epoch
    and minibatch.

    Each epoch reshuffles, as the PPO references do: OpenAI baselines' ppo2
    and CleanRL's ``ppo.py`` shuffle the sample indices inside their epoch
    loop, and APO's official code (``xtma/apo``,
    ``apo/algos/apg/appo.py``, ``APPO.optimize_agent``) draws
    ``iterate_mb_idxs(..., shuffle=True)`` once per epoch.
    """

    def epoch(agent_state: Any, epoch_key: jax.Array) -> tuple[Any, Any]:
        return jax.lax.scan(step, agent_state, minibatches(epoch_key))

    epoch_keys = jax.random.split(key, n_epochs)
    agent_state, aux = jax.lax.scan(epoch, agent_state, epoch_keys)
    return agent_state, jax.tree.map(jnp.mean, aux)
