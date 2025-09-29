from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState

PPOState = BaseAgentState


@struct.dataclass
class PPOConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    gamma: float = 0.99
    ent_coef: float = 0.0
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
