"""The Q-critics' regression to a target no gradient flows back through,
and its logs: shared by every Q-critic agent (SAC and its descendants,
TD3, IQL, CrossQ); and its discrete-action counterpart, the Q-network's
regression at the taken action, shared by the value agents (DQN, PQN)."""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict

from ajax.agents.recurrent import RecurrentCarries, q_values
from ajax.agents.update_steps import gradient_step
from ajax.extensions.base import ExtensionStack
from ajax.networks.q_networks import predict_q
from ajax.state import LoadedTrainState


@struct.dataclass
class CoreCriticAux:
    critic_loss: jax.Array
    q_pred_min: jax.Array
    var_preds: jax.Array
    q_gap: jax.Array  # mean (max - min) over the critics: |Q1 - Q2| for two


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array
    q_pred_mean: jax.Array
    abs_td_error: jax.Array
    # The loss builds these metrics before the gradient exists; the step
    # fills it in.
    grad_norm: jax.Array | float = jnp.nan


def compute_mean_critic_loss(
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
    return compute_q_regression_loss(q_preds, target_q)


def critic_value_readers(
    critic_state: LoadedTrainState, observations: jax.Array, actions: jax.Array
) -> tuple[Callable, Callable]:
    """on_target's predictions and target_value_at for an ensemble of Q
    critics: their mean prediction at the batch's rows (observations,
    actions) at the step's parameters, which the loss regresses, and their
    mean value at given observations and actions at the target's (the
    step's where there are no target critics). No gradient flows back, and
    each is traced only where an extension reads it."""

    def mean_value(params: FrozenDict, obs: jax.Array, taken: jax.Array) -> jax.Array:
        values = q_values(critic_state, params, obs, taken)
        return jax.lax.stop_gradient(jnp.mean(values, axis=0))

    target_params = critic_state.target_params
    if target_params is None:
        target_params = critic_state.params

    def predictions() -> jax.Array:
        return mean_value(critic_state.params, observations, actions)

    def target_value_at(obs: jax.Array, taken: jax.Array) -> jax.Array:
        return mean_value(target_params, obs, taken)

    return predictions, target_value_at


def q_network_value_readers(
    q_state: LoadedTrainState, observations: jax.Array, actions: jax.Array
) -> tuple[Callable, Callable]:
    """critic_value_readers for a Q-network over discrete actions (DQN's):
    its value at the action indices taken, at the step's parameters and at
    the target's (the step's where it has none)."""

    def value(params: FrozenDict, obs: jax.Array, taken: jax.Array) -> jax.Array:
        values = predict_q(q_state, params, obs)
        chosen = jnp.take_along_axis(values, taken.astype(jnp.int32), axis=-1)
        return jax.lax.stop_gradient(chosen)

    target_params = q_state.target_params
    if target_params is None:
        target_params = q_state.params

    def predictions() -> jax.Array:
        return value(q_state.params, observations, actions)

    def target_value_at(obs: jax.Array, taken: jax.Array) -> jax.Array:
        return value(target_params, obs, taken)

    return predictions, target_value_at


def compute_q_regression_loss(
    q_preds: jax.Array, target_q: jax.Array
) -> Tuple[jax.Array, CoreCriticAux]:
    """The critics' mean squared error to target_q, which no gradient
    flows back through (the callers stop it), and its logs, from their
    predictions q_preds (the ensemble axis first),
    however the agent computed them: compute_mean_critic_loss's pass or
    CrossQ's joint one."""
    prediction_variance = q_preds.var(axis=0, keepdims=True)

    total_loss = jnp.mean((q_preds - target_q) ** 2)
    q_pred_min = jnp.min(q_preds, axis=0)

    return total_loss, CoreCriticAux(
        critic_loss=total_loss,
        q_pred_min=q_pred_min.mean().flatten(),
        var_preds=prediction_variance.mean().flatten(),
        q_gap=(jnp.max(q_preds, axis=0) - q_pred_min).mean().flatten(),
    )


def compute_mse_td_loss(q_taken: jax.Array, target_q: jax.Array) -> jax.Array:
    """Mean-squared TD error."""
    return jnp.mean((q_taken - target_q) ** 2)


def make_huber_td_loss(delta: float = 1.0) -> Callable:
    """Build a Huber TD-loss callable.

    Huber loss is quadratic within delta of zero error and linear
    beyond it: its gradient is the TD error clipped to [-delta, delta],
    the original DQN's clip (google-deepmind/dqn @9d9b1d1
    NeuralQLearner.lua:224-226, clip_delta=1 in run_gpu:31), DQN's
    default at delta = 1.
    """

    def compute_huber_td_loss(q_taken: jax.Array, target_q: jax.Array) -> jax.Array:
        return jnp.mean(optax.huber_loss(q_taken, target_q, delta=delta))

    return compute_huber_td_loss


#: DQN's default TD loss; one function, so jit's static-argument cache hits
#: across runs.
compute_huber_1_td_loss = make_huber_td_loss(1.0)


def compute_q_loss(
    q_params,
    q_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    loss_fn: Callable,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """TD loss between Q(s, a) at the taken action and a pre-computed target.

    loss_fn(q_taken, target_q) -> scalar selects the error norm
    (compute_huber_1_td_loss, DQN's default, or compute_mse_td_loss).
    """
    q_all = predict_q(q_state, q_params, observations)
    # actions is (batch, 1) int32 -> gather one Q-value per row.
    q_taken = jnp.take_along_axis(q_all, actions, axis=-1)
    loss = loss_fn(q_taken, target_q)
    aux = ValueAuxiliaries(
        critic_loss=loss.flatten(),
        target_q=target_q.mean().flatten(),
        q_pred_mean=q_taken.mean().flatten(),
        abs_td_error=jnp.abs(q_taken - target_q).mean().flatten(),
    )
    return loss, aux


def q_gradient_step(
    agent_state: Any,
    observations: jax.Array,
    actions: jax.Array,
    targets: jax.Array,
    td_loss_fn: Callable,
    extension_stack: ExtensionStack,
    key: Optional[jax.Array],
    total_timesteps: int,
) -> Tuple[LoadedTrainState, ValueAuxiliaries, dict]:
    """One step of the Q-network (agent_state.actor_state) down the TD
    loss plus the extensions' critic-loss terms, folded on key (drawn
    for them alone; None without extensions). Returns the stepped
    Q-network, the loss's metrics, with the gradients' norm, and the
    extensions' metrics."""
    q_state = agent_state.actor_state

    def total_td_loss(params: Any) -> Tuple[jax.Array, Any]:
        loss, aux = compute_q_loss(
            params, q_state, observations, actions, targets, td_loss_fn
        )
        metrics: dict = {}
        if extension_stack:
            assert key is not None
            batch = {
                "observations": observations,
                "actions": actions,
                "targets": targets,
                # At the parameters being differentiated, so the term has a
                # gradient.
                "q_state": q_state.replace(params=params),
            }
            extra, metrics = extension_stack.compute_extensions_critic_loss(
                agent_state,
                batch,
                agent_state.collector_state.timestep,
                key,
                total_timesteps,
            )
            loss = loss + extra
        return loss, (aux, metrics)

    q_state, (aux, metrics), grad_norm = gradient_step(q_state, total_td_loss)
    return q_state, aux.replace(grad_norm=grad_norm), metrics
