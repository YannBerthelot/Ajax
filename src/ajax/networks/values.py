"""Reading a critic: what it takes as input (the observation normalised,
the action in its units) and its values on rows and on sequences, and the
value estimates an evaluation compares (state_value, action_value).
"""

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from ajax.environments.obs_norm import apply_obs_norm
from ajax.networks.memory import Carry
from ajax.state import LoadedTrainState, network_variables


def action_value_input(
    obs: jax.Array, action: jax.Array, previous_action: Optional[jax.Array] = None
) -> jax.Array:
    """A Q-critic's input: ``(obs, action)``, or, given the previous
    action, the recurrent critic's ``(obs, previous_action, action)``: its
    memory reads the observation and the previous action, the current
    action joins at its head (``Critic.query_dim``)."""
    if previous_action is None:
        return jnp.concatenate([obs, action], axis=-1)
    return jnp.concatenate([obs, previous_action, action], axis=-1)


def in_critic_units(critic_state: LoadedTrainState, actions: jax.Array) -> jax.Array:
    """``actions`` as the critic reads them: ``critic_state.action_units``'
    affine map of the agent's [-1, 1] actions, or the actions themselves."""
    units = getattr(critic_state, "action_units", None)
    if units is None:
        return actions
    offset, scale = (jnp.asarray(u, actions.dtype) for u in units)
    return offset + scale * actions


def _apply_critic_action_units(
    critic_state: LoadedTrainState, x: jax.Array
) -> jax.Array:
    """The action slice of a Q-critic's input ``concat(obs, action)`` in the
    critic's units (:func:`in_critic_units`): every caller passes the
    agent's [-1, 1] actions."""
    units = getattr(critic_state, "action_units", None)
    if units is None:
        return x
    k = len(units[0])
    return jnp.concatenate(
        [x[..., :-k], in_critic_units(critic_state, x[..., -k:])], axis=-1
    )


def _apply_critic_obs_norm(critic_state: LoadedTrainState, x: jax.Array) -> jax.Array:
    """Normalise the obs slice of a critic input (obs or concat(obs, action));
    the action slice stays raw (``apply_obs_norm``). No-op when
    normalisation is disabled."""
    return apply_obs_norm(x, getattr(critic_state, "obs_norm_info", None))


def critic_input(critic_state: LoadedTrainState, x: jax.Array) -> jax.Array:
    """``x`` (obs, or ``concat(obs, action)`` with the agent's action) as a
    feedforward critic reads it: its observations normalised, its action in
    its units."""
    return _apply_critic_action_units(
        critic_state, _apply_critic_obs_norm(critic_state, x)
    )


def predict_value_sequence(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
    resets: jax.Array,
    initial_hidden: Carry,
    *,
    quantiles: bool = False,
) -> Tuple[jax.Array, Carry]:
    """Run a recurrent critic over a time-major sequence (``quantiles``: a
    distributional critic's quantiles, last axis, in place of the values).

    Args:
        x: (T, B, features) critic input (obs, or concat(obs, action)).
        resets: (T, B) episode-start flags aligned with ``x`` (resets[t]
            means x[t] is the first observation of a new episode).
        initial_hidden: carry valid for x[0]; leaves are (num, B, hidden)
            for critic ensembles.

    Returns:
        (values, final_hidden): values (num, T, B, 1); final_hidden is the
        carry after consuming the whole sequence.
    """
    x = _apply_critic_obs_norm(critic_state, x)
    return critic_state.apply_fn(
        network_variables(critic_state, critic_params),
        x,
        hidden_state=initial_hidden,
        done=resets,
        method="quantiles" if quantiles else None,
    )


def phi_star_critic_state(critic_state: LoadedTrainState) -> LoadedTrainState:
    """The critic state the frozen expert critic phi*'s parameters
    (``expert_critic_params``) are applied with: the online critic's
    network without the agent's observation statistics, as MC pretraining
    and PhiRefresh fit phi* on the observations as stored."""
    if getattr(critic_state, "obs_norm_info", None) is None:
        return critic_state
    return critic_state.replace(obs_norm_info=None)


def predict_value(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
    *,
    quantiles: bool = False,
) -> jax.Array:
    """The critic's values at ``x``: obs, or ``concat(obs, action)`` with the
    agent's [-1, 1] action, which a critic with ``action_units`` reads in its
    units (TD3's, in the env's). ``quantiles``: a distributional critic's
    quantiles (last axis) in place of their mean, the value."""
    x = critic_input(critic_state, x)
    variables = network_variables(critic_state, critic_params)
    if quantiles:
        return critic_state.apply_fn(variables, x, method="quantiles")
    return critic_state.apply_fn(variables, x)


def get_policy_updating_batch_statistics(
    actor_state: LoadedTrainState, params: Any, obs: jax.Array
) -> Tuple[Any, Optional[Any]]:
    """The actor's distribution at obs in training mode, and its batch
    normalisation's running statistics after this pass (apply_train).

    In training mode a batch normalisation layer (CrossQ's batch
    renormalisation) normalises with the batch's own statistics and updates
    its running averages; acting and the targets use the running averages
    instead. A feedforward actor only."""
    obs = apply_obs_norm(obs, getattr(actor_state, "obs_norm_info", None))
    return actor_state.apply_train(params, obs)


def predict_value_updating_batch_statistics(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
) -> Tuple[jax.Array, Optional[FrozenDict]]:
    """The critic's values at x in training mode, and its batch
    normalisation's running statistics after this pass
    (LoadedTrainState.apply_train); see get_policy_updating_batch_statistics
    for what training mode changes."""
    return critic_state.apply_train(critic_params, critic_input(critic_state, x))


def state_value(agent_state: Any, obs: jax.Array, action: jax.Array) -> jax.Array:
    """``V(obs)`` of the agent's critic, its ensemble's mean (``action``
    unused): an evaluation's value estimate (:class:`ajax.evaluate.ValueEstimate`)."""
    del action
    critic = agent_state.critic_state
    return predict_value(critic, critic.params, obs).mean(0).squeeze(-1)


def action_value(agent_state: Any, obs: jax.Array, action: jax.Array) -> jax.Array:
    """``Q(obs, action)`` of the agent's critic, its ensemble's mean: an
    evaluation's value estimate (:class:`ajax.evaluate.ValueEstimate`)."""
    critic = agent_state.critic_state
    x = action_value_input(obs, action.astype(obs.dtype))
    return predict_value(critic, critic.params, x).mean(0).squeeze(-1)
