"""DQN's Q-networks, now in :mod:`ajax.networks.q_networks`; their old
path imports them with a FutureWarning until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__,
    dict.fromkeys(
        (
            "GreedyQPolicy",
            "_build_q_encoder",
            "QNetwork",
            "DuelingQNetwork",
            "predict_q",
            "q_network_value",
            "get_initialized_q_network",
        ),
        "ajax.networks.q_networks",
    ),
)
