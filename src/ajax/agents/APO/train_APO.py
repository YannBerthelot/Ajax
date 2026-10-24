"""APO's training, now in :mod:`ajax.agents.APO.APO` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. ``init_APO``, ``update_agent``,
``update_networks`` and ``make_train`` became methods with other
parameters: their old names raise TypeError with the method's call form."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(
    (
        "TORCH_CLIP_EPS",
        "ValueAuxiliaries",
        "policy_loss_function",
        "valid_mean",
        "value_loss_function",
    ),
    "ajax.agents.APO.APO",
)
_new_homes.update(
    {
        "init_APO": "ajax.agents.APO.APO:APO.init",
        "update_agent": "ajax.agents.APO.APO:APO.update",
        "update_networks": "ajax.agents.APO.APO:APO.minibatch_step",
        "make_train": "ajax.agents.families:OnPolicyAgent.make_train",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
