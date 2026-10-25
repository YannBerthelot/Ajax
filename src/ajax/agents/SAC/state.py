from typing import Any, Optional

from flax import struct

from ajax.agents.recurrent import RecurrentReplayConfig
from ajax.agents.temperature import SoftActorCriticState
from ajax.deprecation import moved_names
from ajax.state import LoadedTrainState

# A soft actor-critic's state moved to the temperature's shared home; the
# old name serves it until 0.5.0.
__getattr__ = moved_names(
    __name__, {"SoftACState": "ajax.agents.temperature:SoftActorCriticState"}
)


@struct.dataclass(kw_only=True)
class SACState(SoftActorCriticState):
    """SAC's state, with the frozen expert critic phi* its extensions fill."""

    expert_critic_params: Optional[Any] = None
    expert_v_min: Optional[Any] = None
    expert_v_max: Optional[Any] = None
    # Mutable φ* state for periodic self-consistent refresh (None when refresh disabled)
    expert_critic_state: Optional[LoadedTrainState] = None


@struct.dataclass(kw_only=True)
class SACConfig(RecurrentReplayConfig):
    """SAC's hyperparameters (the wrapper sets every one)."""

    gamma: float
    target_entropy: float
    tau: float
    learning_starts: int
    reward_scale: float
