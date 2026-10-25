"""ASAC's training, now in :mod:`ajax.agents.ASAC.ASAC` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. init_ASAC, update_agent,
update_value_functions, update_policy and make_train became methods with
other parameters: their old names raise TypeError with the method's call
form. value_loss_function is compute_critic_loss."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(
    (
        "AuxiliaryLogs",
        "PolicyAuxiliaries",
        "ThetaAuxiliaries",
        "ValueAuxiliaries",
        "compute_asac_td_target",
        "uniform_log_prob",
        "update_theta",
    ),
    "ajax.agents.ASAC.ASAC",
)
_new_homes.update(
    {
        "value_loss_function": "ajax.agents.ASAC.ASAC:compute_critic_loss",
        "init_ASAC": "ajax.agents.ASAC.ASAC:ASAC.init",
        "update_agent": "ajax.agents.ASAC.ASAC:ASAC.update",
        "update_value_functions": "ajax.agents.ASAC.ASAC:ASAC.update_critic",
        "update_policy": "ajax.agents.ASAC.ASAC:ASAC.update_actor",
        "make_train": "ajax.agents.families:ReplayAgent.make_train",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
