"""ASAC's state and config, now in :mod:`ajax.agents.ASAC.ASAC`; their old
path imports them with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__, dict.fromkeys(("ASACConfig", "ASACState"), "ajax.agents.ASAC.ASAC")
)
