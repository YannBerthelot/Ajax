"""Soft Actor-Critic's maths: the one source for SAC and its descendants.

SAC (train_SAC.py) layers its expert features on top of these functions;
ASAC, REDQ and AVG import them rather than keep copies (the lineage rule,
CONTRIBUTING.md). Functions take pre-computed values (e.g. ``target_q``),
not feature flags.
"""

from typing import Any, Callable, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from ajax.agents import temperature
from ajax.agents.loop import derive_extensions_key, gradient_step
from ajax.agents.recurrent import (
    RecurrentCarries,
    actor_dist,
    q_values,
    stored_actor_carry_dim,
)
from ajax.controller import make_scored_action
from ajax.deprecation import moved_names
from ajax.environments.interaction import init_collector_state
from ajax.extensions.base import ExtensionStack
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import BufferType

S = TypeVar("S", bound=temperature.SoftActorCriticState)


def init_soft_actor_critic(
    init_key: jax.Array,
    collector_key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: Optional[BufferType] = None,
    *,
    num_critics: int,
    window_size: int = 10,
    stored_state: bool = False,
    collector_extras: Optional[dict] = None,
) -> Tuple[Any, Any, Any]:
    """A soft actor-critic's squashed-Gaussian actor, ``num_critics``
    Q-critics and collector (writing to ``buffer`` when given); the
    ``collector_extras`` reach the collector's builder."""
    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=num_critics,
    )
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=env_args.mode,
        buffer=buffer,
        window_size=window_size,
        actor_carry_dim=stored_actor_carry_dim(network_args.memory, stored_state),
        **(collector_extras or {}),
    )
    return actor_state, critic_state, collector_state


def sample_next_actions(
    actor_state: Any,
    next_observations: jax.Array,
    key: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    executed: Optional[Callable[[jax.Array, Any], jax.Array]] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """The bootstrap actions ``a' ~ pi(.|s')`` and their log-probabilities,
    summed over the action dimensions; with a residual policy, the executed
    ones the critics score, ``executed(a', next_a_expert)``."""
    next_pi = actor_dist(
        actor_state, actor_state.params, next_observations, carries, bootstrap=True
    )
    next_actions, log_probs = next_pi.sample_and_log_prob(seed=key)
    if executed is not None:
        next_actions = executed(next_actions, next_a_expert)
    return next_actions, log_probs.sum(-1, keepdims=True)


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
    executed: Optional[Callable[[jax.Array, Any], jax.Array]] = None,
    next_a_expert: Optional[jax.Array] = None,
    carries: Optional[RecurrentCarries] = None,
) -> jax.Array:
    """Pure SAC Bellman target: r + γ(1-d)(min Q_target(s', π(s')) - α log π).

    With a residual policy, executed maps the sampled next actions to the
    executed ones, from the expert's stored next actions next_a_expert
    (ajax.controller.executed_fn): the bootstrap is queried where the
    critic was trained, clip(a_expert(s') + scale * pi(s'), -1, 1). Without
    it the target Q is out of distribution and training diverges.

    The carries (sequence replay) are set exactly when recurrent. Returns
    the target with no gradient; the target extensions (IBRL, CriticBlend,
    MCVarianceCorrection) may change it before the critic loss reads it.
    """
    del recurrent
    sample_key, rng = jax.random.split(rng)
    next_actions, log_probs = sample_next_actions(
        actor_state, next_observations, sample_key, carries, executed, next_a_expert
    )
    q_targets = q_values(
        critic_state,
        critic_state.target_params,
        next_observations,
        next_actions,
        carries,
        bootstrap=True,
    )
    min_q_target = jnp.min(q_targets, axis=0, keepdims=False)
    target = soft_bellman_target(
        min_q_target, log_probs, rewards, dones, gamma, alpha, reward_scale
    )
    return jax.lax.stop_gradient(target)


def soft_bellman_target(
    next_q: jax.Array,
    next_log_probs: jax.Array,
    rewards: jax.Array,
    dones: jax.Array,
    gamma: float,
    alpha: jax.Array,
    reward_scale: float = 1.0,
) -> jax.Array:
    """SAC's Bellman target ``r + gamma (1 - d)(Q' - alpha log pi(a'|s'))``
    from the bootstrap value ``next_q`` (the critics' reduced value at
    ``(s', a')``) and ``a'``'s log-probabilities, however the agent computed
    them: a target critic's pass (:func:`compute_td_target`) or CrossQ's
    joint pass."""
    rewards = rewards * reward_scale
    return rewards + gamma * (1.0 - dones) * (next_q - alpha * next_log_probs)


def soft_policy_loss(
    actor_params: FrozenDict,
    actor_state,
    critic_state,
    observations: jax.Array,
    alpha: jax.Array,
    rng: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    q_reduce: Callable[..., jax.Array] = jnp.min,
    action_transform: Optional[Callable[[jax.Array], jax.Array]] = None,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, Any]]:
    """``mean(alpha log pi(a|s) - Q(s, a))``, ``a ~ pi(.|s)``, ``Q`` the
    critics' values reduced by ``q_reduce`` over the ensemble axis, at
    ``action_transform(a)`` when given (residual RL scores the executed
    action).

    Returns the loss and ``(log_probs, q, pi)``. In sequence mode the
    fresh actions are queried at the critic's head, its memory reading the
    actions taken (:func:`~ajax.agents.recurrent.q_values`).
    """
    pi = actor_dist(actor_state, actor_params, observations, carries)
    return soft_policy_objective(
        pi, critic_state, observations, alpha, rng, carries, q_reduce, action_transform
    )


def soft_policy_objective(
    pi: Any,
    critic_state,
    observations: jax.Array,
    alpha: jax.Array,
    rng: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    q_reduce: Callable[..., jax.Array] = jnp.min,
    action_transform: Optional[Callable[[jax.Array], jax.Array]] = None,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, Any]]:
    """:func:`soft_policy_loss` on the policy ``pi`` at ``observations``,
    however the agent computed it (CrossQ's actor in training)."""
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = pi.sample_and_log_prob(seed=sample_key)
    if action_transform is not None:
        actions = action_transform(actions)
    q = q_reduce(
        q_values(critic_state, critic_state.params, observations, actions, carries),
        axis=0,
    )
    log_probs = log_probs.sum(-1, keepdims=True)
    assert log_probs.shape == q.shape, f"{log_probs.shape} != {q.shape}"
    return (alpha * log_probs - q).mean(), (log_probs, q, pi)


def soft_actor_step(
    agent_state: S,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
    q_reduce: Callable[..., jax.Array] = jnp.min,
    a_expert: Optional[jax.Array] = None,
    actions: Optional[jax.Array] = None,
    executed: Optional[Callable[[jax.Array, Any], jax.Array]] = None,
) -> Tuple[S, Tuple[jax.Array, jax.Array, jax.Array], dict]:
    """One actor step on :func:`soft_policy_loss` plus the extensions'
    actor-loss terms, which also read ``a_expert``, the expert's stored
    actions at the rows (None without an expert), and ``actions``, the
    rows' own. With a residual policy the critics score ``executed(a,
    a_expert)`` (:func:`ajax.controller.executed_actions`). Returns the state,
    ``(loss, log_probs, q)``, the SAC loss before the extensions' terms,
    and the extensions' metrics."""
    agent_state, policy_key = agent_state.next_keys()
    alpha = temperature.get_alpha(agent_state)
    actor_state = agent_state.actor_state
    transform = make_scored_action(executed, a_expert)

    def loss_fn(params: FrozenDict) -> Tuple[jax.Array, Any]:
        loss, (log_probs, q, pi) = soft_policy_loss(
            params,
            actor_state,
            agent_state.critic_state,
            observations,
            alpha,
            policy_key,
            carries,
            q_reduce,
            transform,
        )
        actor_batch = {
            "observations": observations,
            "raw_observations": raw_observations,
            "pi_mean": pi.mean(),
            "actor_params": params,
            "actor_state": actor_state,
            "a_expert": a_expert,
            "actions": actions,
            "critic_state": agent_state.critic_state,
            "agent_loss": loss,
        }
        extra, metrics = extension_stack.compute_extensions_actor_loss(
            agent_state,
            actor_batch,
            agent_state.collector_state.timestep,
            derive_extensions_key(policy_key),
            total_timesteps,
        )
        return loss + extra, ((loss, log_probs, q), metrics)

    actor_state, (out, metrics), _ = gradient_step(actor_state, loss_fn)
    return agent_state.replace(actor_state=actor_state), out, metrics


def temperature_loss_fn(
    log_alpha_params: FrozenDict,
    log_probs: jax.Array,
    target_entropy: jax.Array,
) -> Tuple[jax.Array, temperature.TemperatureAuxiliaries]:
    """Standard SAC temperature loss: log(α) · (-log π - H_target)."""
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    loss = (log_alpha * jax.lax.stop_gradient(-log_probs - target_entropy)).mean()
    return loss, temperature.TemperatureAuxiliaries(
        alpha=alpha,
        log_alpha=log_alpha,
        effective_target_entropy=target_entropy,
    )


def alpha_temperature_loss_fn(
    log_alpha_params: FrozenDict,
    log_probs: jax.Array,
    target_entropy: jax.Array,
) -> Tuple[jax.Array, temperature.TemperatureAuxiliaries]:
    """``alpha (H - H_target)``, ``H = -mean log pi``: the temperature's loss
    in CrossQ's authors' SBX fork (adityab/CrossQ@a318a26,
    ``sbx/sac/sac.py:414-416``), alpha in place of log alpha, so its gradient
    is alpha times :func:`temperature_loss_fn`'s (SBX later took the log)."""
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    loss = alpha * jax.lax.stop_gradient(-log_probs.mean() - target_entropy)
    return loss, temperature.TemperatureAuxiliaries(
        alpha=alpha,
        log_alpha=log_alpha,
        effective_target_entropy=target_entropy,
    )


def update_temperature(
    agent_state: S,
    log_probs: jax.Array,
    target_entropy: jax.Array,
    loss_fn: Callable[..., Tuple[jax.Array, temperature.TemperatureAuxiliaries]] = (
        temperature_loss_fn
    ),
) -> Tuple[S, temperature.TemperatureAuxiliaries]:
    """The 0.4.0 form of ajax.agents.temperature.update_temperature, SAC's
    temperature loss by default (0.5.0 removes it)."""
    return temperature.update_temperature(
        agent_state, log_probs, target_entropy, loss_fn
    )


_new_homes = {
    "CoreCriticAux": "ajax.agents.critics",
    "TemperatureAuxiliaries": "ajax.agents.temperature",
    "create_alpha_train_state": "ajax.agents.temperature",
    "critic_loss_fn": "ajax.agents.critics:compute_mean_critic_loss",
    "update_target_networks": "ajax.agents.update_steps",
}
__getattr__ = moved_names(__name__, _new_homes)
