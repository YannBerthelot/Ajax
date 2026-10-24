"""TD3's networks, now in :mod:`ajax.networks.deterministic`; their old
path imports them with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
        (
            "DeterministicActor",
            "DeterministicPIDActor",
            "get_initialized_td3_actor_critic",
        ),
        "ajax.networks.deterministic",
    ),
)
