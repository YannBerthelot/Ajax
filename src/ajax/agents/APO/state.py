import jax.numpy as jnp
from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState


@struct.dataclass(kw_only=True)
class APOState(BaseAgentState):
    average_reward: float
    b: float


@struct.dataclass
class APOConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    ent_coef: float = 0.0
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    alpha: float = 0.1
    gamma: float = jnp.nan
    nu: float = 0.1
    # When set (>0), use this as the number of minibatches per epoch
    # directly -- decouples num_minibatches from batch_size, matching
    # brax PPO's surface. When 0 (legacy default), the old formula
    # ``max(batch_size, n_steps) // min(...)`` is used (couples the
    # two; requires ``n_steps % num_minibatches == 0``).
    num_minibatches: int = 0
