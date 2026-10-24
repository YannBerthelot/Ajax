"""APO's state and config, now in :mod:`ajax.agents.APO.APO`; their old
path imports them with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__, dict.fromkeys(("APOConfig", "APOState"), "ajax.agents.APO.APO")
)
