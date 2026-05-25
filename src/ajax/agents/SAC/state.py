from typing import Any, Optional

from flax import struct
from jax.tree_util import Partial as partial

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@partial(struct.dataclass, kw_only=True)
class SACState(BaseAgentState):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    alpha: LoadedTrainState  # Temperature parameter
    lambda_param: float
    expert_critic_params: Optional[Any] = None
    expert_v_min: Optional[Any] = None
    expert_v_max: Optional[Any] = None
    # Mutable φ* state for periodic self-consistent refresh (None when refresh disabled)
    expert_critic_state: Optional[LoadedTrainState] = None
    # Optional safety V-head (used by SafeSAC via init_transform/auxiliary_update hooks)
    safety_critic_state: Optional[LoadedTrainState] = None
    # Optional auxiliary representation module (VAE decoder + classifier for
    # Approach A; RSSM cell + prior/posterior/decoder for Approach B). Holds
    # the aux module's apply_fn / params / optimizer state; the critic's
    # encoder is shaped by aux gradients via a separate small-LR path that
    # also lives here (the aux module owns its own optimizer, the critic's
    # encoder is driven via the existing safety_critic_state's small-LR tx).
    aux_state: Optional[LoadedTrainState] = None


@partial(struct.dataclass, kw_only=True)
class SACConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    gamma: float
    target_entropy: float
    tau: float = 0.005
    learning_starts: int = 100
    reward_scale: float = 5.0
