from flax import struct
from jax.tree_util import Partial as partial

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@partial(struct.dataclass, kw_only=True)
class ASACState(BaseAgentState):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    alpha: LoadedTrainState  # Temperature parameter
    episode_termination_penalty: float
    theta: float


@struct.dataclass
class ASACConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    target_entropy: float
    tau: float = 0.005
    learning_starts: int = 100
    reward_scale: float = 5.0
    p_0: float = 10.0
    # Recurrent (memory) training only. Sampled sequences have length
    # burn_in + sequence_length + 1: the first burn_in steps warm the
    # carries from zero under stop_gradient (R2D2-style), the next
    # sequence_length steps are trained on, and the final step provides
    # the bootstrap next-observations.
    burn_in: int = 8
    sequence_length: int = 16
