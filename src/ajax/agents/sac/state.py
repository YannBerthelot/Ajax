from typing import Optional

from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@struct.dataclass
class SACState(BaseAgentState):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    alpha: LoadedTrainState  # Temperature parameter
    evarest_alpha: LoadedTrainState  # Evarest parameter


@struct.dataclass
class SACConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    target_entropy: float
    tau: float = 0.005
    learning_starts: int = 100
    reward_scale: float = 5.0
    init_evarest_alpha: Optional[float] = None
    evarest_alpha_learning_rate: Optional[float] = None
