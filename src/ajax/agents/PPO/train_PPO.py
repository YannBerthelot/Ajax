"""PPO's old train module: its names moved to :mod:`ajax.agents.PPO.PPO` and
the shared modules, from which they resolve until 0.5.0.

It keeps the losses' 0.4.0 forms, value_loss_function and
policy_loss_function, until 0.5.0. PPO.py's compute_value_loss and
compute_policy_loss read a minibatch instead, which a network with memory
replays (ajax.agents.recurrent)."""

from typing import Any, Callable, Optional, Tuple

import jax
from flax.core import FrozenDict

from ajax.agents.PPO.PPO import compute_policy_loss, compute_value_loss
from ajax.deprecation import moved_names, warn_former_parameters
from ajax.state import LoadedTrainState


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    recurrent: bool,
    vf_coef: float = 1.0,
    extra_loss_fn: Optional[Callable] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[Any, dict]]:
    """compute_value_loss with its 0.4.0 parameters: a recurrent critic
    replays the observations from initial_hidden, resets flagging the ones
    that start an episode."""
    warn_former_parameters(
        f"{__name__}.value_loss_function", "ajax.agents.PPO.PPO.compute_value_loss"
    )
    minibatch: dict[str, Any] = {"obs": observations}
    if recurrent:
        minibatch.update({"resets": resets, "critic_hidden": initial_hidden})
    return compute_value_loss(
        critic_params, critic_states, minibatch, value_targets, vf_coef, extra_loss_fn
    )


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    recurrent: bool,
    clip_coef: float,
    ent_coef: float,
    advantage_normalization: bool,
    extra_loss_fn: Optional[Callable] = None,
    raw_actions: Optional[jax.Array] = None,
    entropy_rng: Optional[jax.Array] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, Tuple[Any, dict]]:
    """compute_policy_loss with its 0.4.0 parameters: a recurrent actor
    replays the observations from initial_hidden, resets flagging the ones
    that start an episode."""
    warn_former_parameters(
        f"{__name__}.policy_loss_function", "ajax.agents.PPO.PPO.compute_policy_loss"
    )
    minibatch: dict[str, Any] = {
        "obs": observations,
        "actions": actions,
        "log_probs": log_probs,
        "raw_actions": raw_actions,
    }
    if recurrent:
        minibatch.update({"resets": resets, "actor_hidden": initial_hidden})
    return compute_policy_loss(
        actor_params,
        actor_state,
        minibatch,
        gae,
        clip_coef,
        ent_coef,
        advantage_normalization,
        extra_loss_fn,
        entropy_rng,
    )


__getattr__ = moved_names(
    __name__,
    {
        "init_PPO": "ajax.agents.PPO.PPO:PPO.init",
        "update_agent": "ajax.agents.PPO.PPO:PPO.update",
        "make_train": "ajax.agents.families:OnPolicyAgent.make_train",
        "Geometry": "ajax.agents.minibatch:RolloutSplit",
        "minibatch_geometry": "ajax.agents.minibatch:choose_rollout_split",
        "default_bptt_length": "ajax.agents.minibatch",
        "run_epochs": "ajax.agents.minibatch",
        "clip_global_norm": "ajax.agents.update_steps",
        "joint_step": "ajax.agents.update_steps",
    },
)
