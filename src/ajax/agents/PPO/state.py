"""PPO's old state module: its names moved to :mod:`ajax.agents.PPO.PPO`,
from which they resolve until 0.5.0."""

from ajax.deprecation import moved_names

__getattr__ = moved_names(
    __name__, dict.fromkeys(("PPOConfig", "PPOState"), "ajax.agents.PPO.PPO")
)
