"""REDQ's training, now in :mod:`ajax.agents.REDQ.REDQ` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. init_REDQ, update_agent,
update_value_functions, update_policy and make_train became methods with
other parameters: their old names raise TypeError with the method's call
form. value_loss_function is compute_critic_loss."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(
    (
        "AuxiliaryLogs",
        "FIRST_ALPHA",
        "PolicyAuxiliaries",
        "ValueAuxiliaries",
        "compute_redq_td_target",
        "q_ensemble_divergence",
        "read_temperature",
    ),
    "ajax.agents.REDQ.REDQ",
)
_new_homes.update(
    {
        "value_loss_function": "ajax.agents.REDQ.REDQ:compute_critic_loss",
        "init_REDQ": "ajax.agents.REDQ.REDQ:REDQ.init",
        "update_agent": "ajax.agents.REDQ.REDQ:REDQ.update",
        "update_value_functions": "ajax.agents.REDQ.REDQ:REDQ.update_critic",
        "update_policy": "ajax.agents.REDQ.REDQ:REDQ.update_actor",
        "make_train": "ajax.agents.families:ReplayAgent.make_train",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
