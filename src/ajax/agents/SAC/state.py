from typing import Any, Optional

from flax import struct
from jax.tree_util import Partial as partial

from ajax.agents.recurrent import RecurrentReplayConfig
from ajax.state import BaseAgentState, LoadedTrainState


@partial(struct.dataclass, kw_only=True)
class SoftACState(BaseAgentState):
    """A soft actor-critic's state: the base one plus the temperature."""

    alpha: LoadedTrainState  # log-temperature, params["log_alpha"]


@partial(struct.dataclass, kw_only=True)
class SACState(SoftACState):
    """SAC's state, with the frozen expert critic phi* its extensions fill."""

    expert_critic_params: Optional[Any] = None
    expert_v_min: Optional[Any] = None
    expert_v_max: Optional[Any] = None
    # Mutable φ* state for periodic self-consistent refresh (None when refresh disabled)
    expert_critic_state: Optional[LoadedTrainState] = None


@partial(struct.dataclass, kw_only=True)
class SACConfig(RecurrentReplayConfig):
    """SAC's hyperparameters (the wrapper sets every one)."""

    gamma: float
    target_entropy: float
    tau: float
    learning_starts: int
    reward_scale: float
