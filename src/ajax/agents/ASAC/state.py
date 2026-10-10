from flax import struct
from jax.tree_util import Partial as partial

from ajax.agents.recurrent import RecurrentReplayConfig
from ajax.agents.SAC.state import SoftACState


@partial(struct.dataclass, kw_only=True)
class ASACState(SoftACState):
    """ASAC's state: a soft actor-critic's plus the reward rate and the
    termination penalty."""

    episode_termination_penalty: float
    theta: float


@partial(struct.dataclass, kw_only=True)
class ASACConfig(RecurrentReplayConfig):
    """ASAC's hyperparameters (the wrapper sets every one); no discount,
    the criterion is the average reward."""

    target_entropy: float
    tau: float
    learning_starts: int
    reward_scale: float
    p_0: float
