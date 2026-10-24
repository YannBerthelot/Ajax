"""CrossQ's training, now in :mod:`ajax.agents.CrossQ.CrossQ` (its update)
and :mod:`ajax.agents.families` (its loop); the old path imports the names
with a FutureWarning until 0.5.0. ``init_CrossQ``, ``update_agent``,
``update_critics``, ``update_actor_and_temperature`` and ``make_train``
became methods, with the agent's signatures."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__,
    dict.fromkeys(("AuxiliaryLogs", "PolicyAuxiliaries"), "ajax.agents.CrossQ.CrossQ")
    | {
        "init_CrossQ": "ajax.agents.CrossQ.CrossQ:CrossQ.init",
        "update_agent": "ajax.agents.CrossQ.CrossQ:CrossQ.update",
        "update_critics": "ajax.agents.CrossQ.CrossQ:CrossQ.critic_update",
        "update_actor_and_temperature": (
            "ajax.agents.CrossQ.CrossQ:CrossQ.actor_update"
        ),
        "make_train": "ajax.agents.families:ReplayAgent.make_train",
    },
)
