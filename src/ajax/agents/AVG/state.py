import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@struct.dataclass
class NormalizationInfo:
    value: jnp.array
    count: jnp.array
    mean: jnp.array
    mean_2: jnp.array


one = jnp.ones(1)


@partial(struct.dataclass, kw_only=True)
class AVGState(BaseAgentState):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    alpha: LoadedTrainState  # Temperature parameter
    reward: NormalizationInfo
    gamma: NormalizationInfo
    G_return: NormalizationInfo
    scaling_coef: jnp.ndarray


@struct.dataclass
class AVGConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    gamma: float
    target_entropy: float
    learning_starts: int = 0
    reward_scale: float = 1
    num_critics: int = 1  # to switch from single to double-q (or more if you want)
    batch_size: int = (
        1  # AVG only considers batch size of 1, but we keep it for experimentation
    )
