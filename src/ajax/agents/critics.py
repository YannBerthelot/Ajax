"""The Q-critics' regression to a held target, and its logs: shared by every
Q-critic agent (SAC and its descendants, TD3, IQL, CrossQ); and its
discrete-action counterpart, the Q-network's regression at the taken action,
shared by the value agents (DQN, PQN)."""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict

from ajax.agents.recurrent import RecurrentCarries, q_values
from ajax.agents.steps import gradient_step
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


def mse_td_loss(q_taken: jax.Array, target_q: jax.Array) -> jax.Array:
    """Mean-squared TD error."""
    return jnp.mean((q_taken - target_q) ** 2)


def make_huber_td_loss(delta: float = 1.0) -> Callable:
    """Build a Huber TD-loss callable.

    Huber loss is quadratic within ``delta`` of zero error and linear
    beyond it: its gradient is the TD error clipped to ``[-delta, delta]``,
    the original DQN's clip (google-deepmind/dqn @9d9b1d1
    NeuralQLearner.lua:224-226, ``clip_delta=1`` in run_gpu:31), DQN's
    default at ``delta = 1``.
    """

    def huber_td_loss(q_taken: jax.Array, target_q: jax.Array) -> jax.Array:
        return jnp.mean(optax.huber_loss(q_taken, target_q, delta=delta))

    return huber_td_loss


#: DQN's default TD loss; one function, so jit's static-argument cache hits
#: across runs.
huber_1_td_loss = make_huber_td_loss(1.0)


def q_loss_fn(
    q_params,
    q_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    loss_fn: Callable,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """TD loss between Q(s, a) at the taken action and a pre-computed target.

    ``loss_fn(q_taken, target_q) -> scalar`` selects the error norm
    (``huber_1_td_loss``, DQN's default, or ``mse_td_loss``).
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
    """One step of the Q-network (``agent_state.actor_state``) down the TD
    loss plus the extensions' critic-loss terms, folded on ``key`` (drawn
    for them alone; ``None`` without extensions). Returns the stepped
    Q-network, the loss's metrics, with the gradients' norm, and the
    extensions' metrics."""
    q_state = agent_state.actor_state

    def loss_fn(params: Any) -> Tuple[jax.Array, Any]:
        loss, aux = q_loss_fn(
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
            extra, metrics = extension_stack.fold_critic_loss(
                agent_state,
                batch,
                agent_state.collector_state.timestep,
                key,
                total_timesteps,
            )
            loss = loss + extra
        return loss, (aux, metrics)

    q_state, (aux, metrics), grad_norm = gradient_step(q_state, loss_fn)
    return q_state, aux.replace(grad_norm=grad_norm), metrics
