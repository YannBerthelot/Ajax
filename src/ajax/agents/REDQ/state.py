from flax import struct
from jax.tree_util import Partial as partial

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@partial(struct.dataclass, kw_only=True)
class REDQState(BaseAgentState):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    alpha: LoadedTrainState  # Temperature parameter


@struct.dataclass
class REDQConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    gamma: float
    target_entropy: float
    tau: float = 0.005
    learning_starts: int = 100
    reward_scale: float = 5.0
    num_critics: int = 10
    subset_size: int = 2
    num_critic_updates: int = 20
