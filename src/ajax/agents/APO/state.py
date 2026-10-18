from typing import Optional

import jax.numpy as jnp
from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState


@struct.dataclass(kw_only=True)
class APOState(BaseAgentState):
    average_reward: float
    b: float


@struct.dataclass
class APOConfig(BaseAgentConfig):
    """APO's hyperparameters."""

    # The official code's entropy_loss_coeff (APO's constructor).
    ent_coef: float = 0.01
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    alpha: float = 0.1
    gamma: float = jnp.nan
    nu: float = 0.1
    # When > 0, decouples num_minibatches from batch_size (brax-style).
    num_minibatches: int = 0
    vf_coef: float = 1.0
    # The global-norm clip of both networks' gradients at once
    # (train_APO.update_networks); None, no clip.
    max_grad_norm: Optional[float] = 10.0
