"""SAC's state and config, now in :mod:`ajax.agents.SAC.SAC`, and a soft
actor-critic's state, in :mod:`ajax.agents.temperature`; their old path
imports them with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

_new_homes = dict.fromkeys(("SACConfig", "SACState"), "ajax.agents.SAC.SAC")
_new_homes.update({"SoftACState": "ajax.agents.temperature:SoftActorCriticState"})
__getattr__ = moved_names(__name__, _new_homes)
