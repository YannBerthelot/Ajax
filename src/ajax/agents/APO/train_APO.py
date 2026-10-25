"""APO's training, now in :mod:`ajax.agents.APO.APO` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. ``init_APO``, ``update_agent``,
``update_networks`` and ``make_train`` became methods with other
parameters: their old names raise TypeError with the method's call form.

policy_loss_function is APO.py's compute_policy_loss, and
value_loss_function, kept here until 0.5.0, is compute_value_loss with its
0.4.0 parameter names."""

from typing import Any, Optional, Tuple

import jax
from flax.core import FrozenDict

from ajax.agents.APO.APO import compute_value_loss
from ajax.deprecation import moved_names, warn_former_parameters
from ajax.state import LoadedTrainState


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    nu: float,
    b: float | jax.Array,
    vf_coef: float = 1.0,
    valid: Optional[jax.Array] = None,
) -> Tuple[jax.Array, Any]:
    """compute_value_loss with its 0.4.0 parameter names (b is the value
    bias)."""
    warn_former_parameters(
        f"{__name__}.value_loss_function", "ajax.agents.APO.APO.compute_value_loss"
    )
    return compute_value_loss(
        critic_params, critic_states, observations, value_targets, nu, b, vf_coef, valid
    )


_new_homes = dict.fromkeys(
    ("TORCH_CLIP_EPS", "ValueAuxiliaries", "valid_mean"), "ajax.agents.APO.APO"
)
_new_homes.update(
    {
        "policy_loss_function": "ajax.agents.APO.APO:compute_policy_loss",
        "init_APO": "ajax.agents.APO.APO:APO.init",
        "update_agent": "ajax.agents.APO.APO:APO.update",
        "update_networks": "ajax.agents.APO.APO:APO.minibatch_step",
        "make_train": "ajax.agents.families:OnPolicyAgent.make_train",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
