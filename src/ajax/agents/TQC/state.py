from flax import struct

from ajax.agents.SAC.SAC import SACConfig
from ajax.agents.temperature import SoftActorCriticState


@struct.dataclass(kw_only=True)
class TQCState(SoftActorCriticState):
    """TQC's state: a soft actor-critic's, nothing more."""


@struct.dataclass(kw_only=True)
class TQCConfig(SACConfig):
    """SAC's hyperparameters plus TQC's quantile critics'."""

    num_critics: int = 5
    n_quantiles: int = 25
    top_quantiles_to_drop_per_net: int = 2
