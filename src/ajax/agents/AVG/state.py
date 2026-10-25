import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents.temperature import SoftActorCriticState
from ajax.state import BaseAgentConfig
from ajax.types import Replaceable


@struct.dataclass
class RunningMoments(Replaceable):
    """The latest ``value`` of one scalar stream (AVG's reward, discount or
    squared return) and its running ``count``, ``mean`` and ``mean_2``;
    not the observation normaliser's :class:`ajax.types.NormalizationInfo`."""

    value: jax.Array
    count: jax.Array
    mean: jax.Array
    mean_2: jax.Array


@struct.dataclass(kw_only=True)
class AVGState(SoftActorCriticState):
    """AVG's state: a soft actor-critic's plus the TD-error scale statistics."""

    reward: RunningMoments
    gamma: RunningMoments
    G_return: RunningMoments
    scaling_coef: jnp.ndarray


@struct.dataclass(kw_only=True)
class AVGConfig(BaseAgentConfig):
    """AVG's hyperparameters."""

    gamma: float
    learning_starts: int = 0
    reward_scale: float = 1
    num_critics: int = 1
