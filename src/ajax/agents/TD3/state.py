from flax import struct
from jax.tree_util import Partial as partial

from ajax.state import BaseAgentConfig, BaseAgentState


@partial(struct.dataclass, kw_only=True)
class TD3State(BaseAgentState):
    """TD3 carries only actor + critic + collector (no temperature, no expert state)."""


@partial(struct.dataclass, kw_only=True)
class TD3Config(BaseAgentConfig):
    gamma: float
    tau: float = 0.005
    learning_starts: int = 100
    reward_scale: float = 1.0
    num_critics: int = 2
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    exploration_noise: float = 0.1
    # Recurrent (memory) training only; see ajax.agents.recurrent.
    burn_in: int = 8
    sequence_length: int = 16
    # R2D2 stored-state replay: read actor carries back from the buffer
    # instead of burning them in from zero (Kapturowski et al. 2019).
    stored_state: bool = False
