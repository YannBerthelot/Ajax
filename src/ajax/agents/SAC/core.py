"""Soft Actor-Critic's maths: the one source for SAC and its descendants.

SAC (train_SAC.py) layers its expert features on top of these functions;
ASAC, REDQ and AVG import them rather than keep copies (the lineage rule,
CONTRIBUTING.md). Functions take pre-computed values (e.g. ``target_q``),
not feature flags.
"""

from typing import Any, Callable, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from ajax.agents.loop import gradient_step
from ajax.agents.recurrent import (
    RecurrentCarries,
    actor_dist,
    q_values,
    stored_actor_carry_dim,
)
from ajax.agents.SAC.state import SoftACState
from ajax.environments.interaction import init_collector_state
from ajax.environments.utils import check_env_is_gymnax
from ajax.extensions.base import ExtensionStack
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import BufferType

S = TypeVar("S", bound=SoftACState)

# ---------------------------------------------------------------------------
# Auxiliary dataclasses (core diagnostics only)
# ---------------------------------------------------------------------------


@struct.dataclass
class CoreCriticAux:
    critic_loss: jax.Array
    q_pred_min: jax.Array
    var_preds: jax.Array


@struct.dataclass
class TemperatureAuxiliaries:
    alpha: jax.Array
    log_alpha: jax.Array
    effective_target_entropy: jax.Array


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def create_alpha_train_state(
    learning_rate: float = 3e-4,
    alpha_init: float = 1.0,
) -> TrainState:
    log_alpha = jnp.log(alpha_init)
    params = FrozenDict({"log_alpha": log_alpha})
    # No gradient clipping on the dual variable. log_alpha's gradient is
    # the scalar (H_target - H_pi); clipping it to a constant norm erases
    # the magnitude (and hence the target) from the update whenever the
    # policy entropy is more than the clip away from the target, turning
    # the step into pure sign-descent at ``learning_rate`` per update.
    # Reference SAC uses unclipped Adam here.
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=lambda params: jnp.exp(params["log_alpha"]),
        params=params,
        tx=tx,
    )


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
    pid_actor_config: Any = None,
    network_extras: Optional[dict] = None,
    collector_extras: Optional[dict] = None,
) -> Tuple[Any, Any, Any]:
    """A soft actor-critic's squashed-Gaussian actor, ``num_critics``
    Q-critics and collector (writing to ``buffer`` when given); the
    ``*_extras`` reach the network and collector builders."""
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
        pid_actor_config=pid_actor_config,
        **(network_extras or {}),
    )
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode="gymnax" if check_env_is_gymnax(env_args.env) else "brax",
        buffer=buffer,
        window_size=window_size,
        actor_carry_dim=stored_actor_carry_dim(network_args.memory, stored_state),
        **(collector_extras or {}),
    )
    return actor_state, critic_state, collector_state


# ---------------------------------------------------------------------------
# TD target computation (pure Bellman — modifiers compose on top)
# ---------------------------------------------------------------------------


def sample_next_actions(
    actor_state: Any,
    next_observations: jax.Array,
    key: jax.Array,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, jax.Array]:
    """The bootstrap actions ``a' ~ pi(.|s')`` and their log-probabilities,
    summed over the action dimensions."""
    next_pi = actor_dist(
        actor_state, actor_state.params, next_observations, carries, bootstrap=True
    )
    next_actions, log_probs = next_pi.sample_and_log_prob(seed=key)
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
    next_action_transform=None,
    next_a_expert: Optional[jax.Array] = None,
    carries: Optional[RecurrentCarries] = None,
) -> jax.Array:
    """Pure SAC Bellman target: r + γ(1-d)(min Q_target(s', π(s')) - α log π).

    ``next_action_transform`` is an optional callable applied to the
    sampled next action before the target critic sees it. Used by
    residual RL so the bootstrap is queried at the same residual-
    transformed action distribution the critic was trained on:
    ``a_target = clip(a_expert(s_{t+1}) + scale * pi(s_{t+1}), -1, 1)``.
    Without this the target Q is OOD and training diverges.

    ``carries`` (sequence replay) is set exactly when ``recurrent``.
    Returns stop-gradient'd target. Expert modules can further modify
    this (IBRL, critic blend, MC correction) before it enters the
    critic loss.
    """
    del recurrent
    rewards = rewards * reward_scale
    sample_key, rng = jax.random.split(rng)
    next_actions, log_probs = sample_next_actions(
        actor_state, next_observations, sample_key, carries
    )
    if next_action_transform is not None:
        next_actions = next_action_transform(
            next_actions,
            next_observations,
            next_a_expert,
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
    var_preds = q_preds.var(axis=0, keepdims=True)

    total_loss = jnp.mean((q_preds - target_q) ** 2)
    q_pred_min = jnp.min(q_preds, axis=0)

    return total_loss, CoreCriticAux(
        critic_loss=total_loss,
        q_pred_min=q_pred_min.mean().flatten(),
        var_preds=var_preds.mean().flatten(),
    )


# ---------------------------------------------------------------------------
# Actor loss (α log π - Q)
# ---------------------------------------------------------------------------


def soft_policy_loss(
    actor_params: FrozenDict,
    actor_state,
    critic_state,
    observations: jax.Array,
    alpha: jax.Array,
    rng: jax.Array,
    carries: Optional[RecurrentCarries] = None,
    q_reduce: Callable[..., jax.Array] = jnp.min,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, Any]]:
    """``mean(alpha log pi(a|s) - Q(s, a))``, ``a ~ pi(.|s)``, ``Q`` the
    critics' values reduced by ``q_reduce`` over the ensemble axis.

    Returns the loss and ``(log_probs, q, pi)``. In sequence mode the
    fresh actions are queried at the critic's head, its memory reading the
    actions taken (:func:`~ajax.agents.recurrent.q_values`).
    """
    pi = actor_dist(actor_state, actor_params, observations, carries)
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = pi.sample_and_log_prob(seed=sample_key)
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
) -> Tuple[S, Tuple[jax.Array, jax.Array, jax.Array]]:
    """One actor step on :func:`soft_policy_loss` plus the extensions'
    actor-loss terms. Returns the state and ``(loss, log_probs, q)``, the
    SAC loss before the extensions' terms."""
    rng, policy_key = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    actor_state = agent_state.actor_state

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
        )
        actor_batch = {
            "observations": observations,
            "raw_observations": raw_observations,
            "pi_mean": pi.mean(),
            "actor_params": params,
            "actor_state": actor_state,
        }
        extra = extension_stack.fold_actor_loss(
            agent_state,
            actor_batch,
            agent_state.collector_state.timestep,
            policy_key,
            total_timesteps,
        )
        return loss + extra, (loss, log_probs, q)

    actor_state, out = gradient_step(actor_state, loss_fn)
    return agent_state.replace(rng=rng, actor_state=actor_state), out


# ---------------------------------------------------------------------------
# Temperature (standard SAC dual gradient) and target networks
# ---------------------------------------------------------------------------


def temperature_loss_fn(
    log_alpha_params: FrozenDict,
    log_probs: jax.Array,
    target_entropy: jax.Array,
) -> Tuple[jax.Array, TemperatureAuxiliaries]:
    """Standard SAC temperature loss: log(α) · (-log π - H_target)."""
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    loss = (log_alpha * jax.lax.stop_gradient(-log_probs - target_entropy)).mean()
    return loss, TemperatureAuxiliaries(
        alpha=alpha,
        log_alpha=log_alpha,
        effective_target_entropy=target_entropy,
    )


def update_temperature(
    agent_state: S,
    log_probs: jax.Array,
    target_entropy: jax.Array,
) -> Tuple[S, TemperatureAuxiliaries]:
    """One dual-gradient step of ``log_alpha`` towards ``target_entropy``,
    ``log_probs`` per action dimension (summed here)."""
    (_, aux), grads = jax.value_and_grad(temperature_loss_fn, has_aux=True)(
        agent_state.alpha.params,
        log_probs.sum(-1),
        target_entropy,
    )
    new_alpha_state = agent_state.alpha.apply_gradients(grads=grads)
    return agent_state.replace(alpha=new_alpha_state), jax.lax.stop_gradient(aux)


def update_target_networks(agent_state: S, tau: float) -> S:
    """Polyak-average the target critics towards the critics."""
    return agent_state.replace(
        critic_state=agent_state.critic_state.soft_update(tau=tau)
    )
