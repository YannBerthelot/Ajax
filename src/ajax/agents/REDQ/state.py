from flax import struct
from jax.tree_util import Partial as partial

from ajax.agents.SAC.state import SACConfig, SoftACState


@partial(struct.dataclass, kw_only=True)
class REDQState(SoftACState):
    """REDQ's state: a soft actor-critic's, nothing more."""


@partial(struct.dataclass, kw_only=True)
class REDQConfig(SACConfig):
    """SAC's hyperparameters plus REDQ's ensemble ones."""

    num_critics: int = 10
    subset_size: int = 2
    num_critic_updates: int = 20
    # SVGD-style function-space kernel repulsion coefficient on the
    # critic ensemble. 0.0 disables it (vanilla REDQ).
    repulsion_coef: float = 0.0
