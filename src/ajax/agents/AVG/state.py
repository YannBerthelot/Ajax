import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from ajax.agents.SAC.state import SoftACState
from ajax.deprecation import moved_names
from ajax.state import BaseAgentConfig


@struct.dataclass
class RunningMoments:
    """The latest ``value`` of one scalar stream (AVG's reward, discount or
    squared return) and its running ``count``, ``mean`` and ``mean_2``;
    not the observation normaliser's :class:`ajax.types.NormalizationInfo`."""

    value: jnp.array
    count: jnp.array
    mean: jnp.array
    mean_2: jnp.array


@partial(struct.dataclass, kw_only=True)
class AVGState(SoftACState):
    """AVG's state: a soft actor-critic's plus the TD-error scale statistics."""

    reward: RunningMoments
    gamma: RunningMoments
    G_return: RunningMoments
    scaling_coef: jnp.ndarray


@partial(struct.dataclass, kw_only=True)
class AVGConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    gamma: float
    target_entropy: float
    learning_starts: int = 0
    reward_scale: float = 1
    num_critics: int = 1  # to switch from single to double-q (or more if you want)


__getattr__ = moved_names(__name__, NormalizationInfo=RunningMoments)
