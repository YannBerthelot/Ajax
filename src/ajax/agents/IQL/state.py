"""IQL's state and config, now in :mod:`ajax.agents.IQL.IQL`; their old
path imports them with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__, dict.fromkeys(("IQLConfig", "IQLState"), "ajax.agents.IQL.IQL")
)
