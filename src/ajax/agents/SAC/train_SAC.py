"""SAC's training, now in :mod:`ajax.agents.SAC.SAC` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. init_SAC, update_agent,
update_value_functions, update_policy and make_train became methods with
other parameters: their old names raise TypeError with the method's call
form."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(
    ("AuxiliaryLogs", "PolicyAuxiliaries", "ValueAuxiliaries"), "ajax.agents.SAC.SAC"
)
_new_homes.update(
    {
        "init_SAC": "ajax.agents.SAC.SAC:SAC.init",
        "update_agent": "ajax.agents.SAC.SAC:SAC.update",
        "update_value_functions": "ajax.agents.SAC.SAC:SAC.update_critic",
        "update_policy": "ajax.agents.SAC.SAC:SAC.update_actor",
        "make_train": "ajax.agents.families:ReplayAgent.make_train",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
