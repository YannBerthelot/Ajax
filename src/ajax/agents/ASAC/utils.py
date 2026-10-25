"""ASAC's termination penalty, now in :mod:`ajax.agents.ASAC.ASAC`; the old
path imports its names with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
        (
            "compute_episode_termination_penalty",
            "get_episode_termination_penalized_rewards",
        ),
        "ajax.agents.ASAC.ASAC",
    ),
)
