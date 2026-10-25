"""TQC's training, now in :mod:`ajax.agents.TQC.TQC` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. init_TQC, update_agent,
update_value_functions, update_policy and make_train became methods with
other parameters: their old names raise TypeError with the method's call
form."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(
    (
        "AuxiliaryLogs",
        "PolicyAuxiliaries",
        "ValueAuxiliaries",
        "compute_tqc_td_target",
        "quantile_huber_loss",
        "truncated_quantiles",
    ),
    "ajax.agents.TQC.TQC",
)
_new_homes.update(
    {
        "init_TQC": "ajax.agents.TQC.TQC:TQC.init",
        "update_agent": "ajax.agents.TQC.TQC:TQC.update",
        "update_value_functions": "ajax.agents.TQC.TQC:TQC.update_critic",
        "update_policy": "ajax.agents.TQC.TQC:TQC.update_actor",
        "make_train": "ajax.agents.families:ReplayAgent.make_train",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
