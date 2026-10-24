"""TD3's state and config, now in :mod:`ajax.agents.TD3.TD3`; their old
path imports them with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__, dict.fromkeys(("TD3Config", "TD3State"), "ajax.agents.TD3.TD3")
)
