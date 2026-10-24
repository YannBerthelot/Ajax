"""PQN's old state module: its names moved to :mod:`ajax.agents.PQN.PQN`,
from which they resolve until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__, dict.fromkeys(("PQNConfig", "PQNState"), "ajax.agents.PQN.PQN")
)
