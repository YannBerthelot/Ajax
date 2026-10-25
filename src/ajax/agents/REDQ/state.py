from flax import struct

from ajax.agents.SAC.state import SACConfig
from ajax.agents.temperature import SoftActorCriticState


@struct.dataclass(kw_only=True)
class REDQState(SoftActorCriticState):
    """REDQ's state: a soft actor-critic's, nothing more."""


@struct.dataclass(kw_only=True)
class REDQConfig(SACConfig):
    """SAC's hyperparameters plus REDQ's ensemble ones."""

    num_critics: int = 10
    subset_size: int = 2
    num_critic_updates: int = 20
