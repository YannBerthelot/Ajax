import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from ajax.agents.SAC.state import SoftACState
from ajax.state import BaseAgentConfig


@struct.dataclass
class NormalizationInfo:
    value: jnp.array
    count: jnp.array
    mean: jnp.array
    mean_2: jnp.array


@partial(struct.dataclass, kw_only=True)
class AVGState(SoftACState):
    """AVG's state: a soft actor-critic's plus the TD-error scale statistics."""

    reward: NormalizationInfo
    gamma: NormalizationInfo
    G_return: NormalizationInfo
    scaling_coef: jnp.ndarray


@partial(struct.dataclass, kw_only=True)
class AVGConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    gamma: float
    target_entropy: float
    learning_starts: int = 0
    reward_scale: float = 1
    num_critics: int = 1  # to switch from single to double-q (or more if you want)
