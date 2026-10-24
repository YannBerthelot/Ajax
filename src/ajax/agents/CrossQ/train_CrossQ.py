"""CrossQ's training, now in :mod:`ajax.agents.CrossQ.CrossQ` (its update)
and :mod:`ajax.agents.families` (its loop); the old path imports the names
with a FutureWarning until 0.5.0. ``init_CrossQ``, ``update_agent``,
``update_critics``, ``update_actor_and_temperature`` and ``make_train``
became methods with other parameters: their old names raise TypeError with
the method's call form."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(
    ("AuxiliaryLogs", "PolicyAuxiliaries"), "ajax.agents.CrossQ.CrossQ"
)
_new_homes.update(
    {
        "init_CrossQ": "ajax.agents.CrossQ.CrossQ:CrossQ.init",
        "update_agent": "ajax.agents.CrossQ.CrossQ:CrossQ.update",
        "update_critics": "ajax.agents.CrossQ.CrossQ:CrossQ.update_critic",
        "update_actor_and_temperature": (
            "ajax.agents.CrossQ.CrossQ:CrossQ.update_actor"
        ),
        "make_train": "ajax.agents.families:ReplayAgent.make_train",
    }
)
__getattr__ = moved_names(__name__, _new_homes)
