from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@struct.dataclass(kw_only=True)
class IQLState(BaseAgentState):
    """IQL's actor, twin Q critic (with its target), and state-value network."""

    value_state: LoadedTrainState


@struct.dataclass(kw_only=True)
class IQLConfig(BaseAgentConfig):
    expectile: float = 0.7
    temperature: float = 3.0
    gamma: float = 0.99
    tau: float = 0.005
    learning_starts: int = 0
    reward_scale: float = 1.0
