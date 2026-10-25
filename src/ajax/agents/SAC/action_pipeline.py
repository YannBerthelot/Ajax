"""SAC's action pipeline, now in :mod:`ajax.agents.action_pipeline`, a shared
home; the old path imports its names with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
        ("ACTION_SLOTS", "check_action_extensions", "make_action_pipeline"),
        "ajax.agents.action_pipeline",
    ),
)
