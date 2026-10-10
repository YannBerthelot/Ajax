from typing import Any, Optional

from flax import struct
from jax.tree_util import Partial as partial

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@partial(struct.dataclass, kw_only=True)
class SACState(BaseAgentState):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    alpha: LoadedTrainState  # Temperature parameter
    expert_critic_params: Optional[Any] = None
    expert_v_min: Optional[Any] = None
    expert_v_max: Optional[Any] = None
    # Mutable φ* state for periodic self-consistent refresh (None when refresh disabled)
    expert_critic_state: Optional[LoadedTrainState] = None


@partial(struct.dataclass, kw_only=True)
class SACConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    gamma: float
    target_entropy: float
    tau: float = 0.005
    learning_starts: int = 100
    reward_scale: float = 5.0
    # Recurrent (memory) training only; see ajax.agents.recurrent.
    burn_in: int = 8
    sequence_length: int = 16
    # R2D2 stored-state replay: read actor carries back from the buffer
    # instead of burning them in from zero (Kapturowski et al. 2019).
    stored_state: bool = False
