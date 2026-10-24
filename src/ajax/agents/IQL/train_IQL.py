"""IQL's training, now in :mod:`ajax.agents.IQL.IQL` (its update and
equations) and :mod:`ajax.agents.families` (its loop); the old path imports
the names with a FutureWarning until 0.5.0. ``init_IQL``, ``update_agent``
and ``make_train`` became methods, with the agent's signatures."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
        ("Auxiliaries", "AuxiliaryLogs", "expectile_loss"), "ajax.agents.IQL.IQL"
    )
    | {
        "init_IQL": "ajax.agents.IQL.IQL:IQL.init",
        "update_agent": "ajax.agents.IQL.IQL:IQL.update",
        "make_train": "ajax.agents.families:ReplayAgent.make_train",
    },
)
