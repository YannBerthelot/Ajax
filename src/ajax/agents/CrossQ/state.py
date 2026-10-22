from flax import struct

from ajax.agents.SAC.state import SoftACState
from ajax.state import BaseAgentConfig


@struct.dataclass(kw_only=True)
class CrossQState(SoftACState):
    """CrossQ's state: a soft actor-critic's, its critics without targets."""


@struct.dataclass(kw_only=True)
class CrossQConfig(BaseAgentConfig):
    """CrossQ's hyperparameters (the wrapper sets every one)."""

    gamma: float
    target_entropy: float
    learning_starts: int
    reward_scale: float
    policy_delay: int
