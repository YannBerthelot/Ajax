from typing import ClassVar, Optional

from flax import struct

from ajax.schedule import Schedule
from ajax.state import BaseAgentConfig, BaseAgentState


@struct.dataclass(kw_only=True)
class APOState(BaseAgentState):
    average_reward: float
    b: float

    #: Its networks' gradients are clipped together (steps.joint_step): a
    #: checkpoint from when each clipped its own restores (ajax.checkpoint).
    joint_gradient_clip: ClassVar[bool] = True
    #: How to restore a checkpoint of APO's former default network, added to
    #: a shape mismatch's error (ajax.checkpoint). Expand-then-contract
    #: (CLAUDE.md, refactoring rule 10): delete it once no such checkpoint
    #: remains.
    restore_hint: ClassVar[str] = (
        '. APO\'s default hidden layers were ("128", "tanh", "128", "tanh")'
        " before they became its reference code's two of 64 units: an APO"
        ' checkpoint saved then restores into APO(..., actor_architecture=("128",'
        ' "tanh", "128", "tanh"), critic_architecture=("128", "tanh", "128",'
        ' "tanh")), its other settings unchanged.'
    )


@struct.dataclass
class APOConfig(BaseAgentConfig):
    """APO's hyperparameters."""

    # The official code's entropy_loss_coeff (APO's constructor).
    ent_coef: float = 0.01
    clip_range: Schedule = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    alpha: float = 0.1
    nu: float = 0.1
    # When > 0, decouples num_minibatches from batch_size (brax-style).
    num_minibatches: int = 0
    vf_coef: float = 1.0
    # The global-norm clip of both networks' gradients at once
    # (train_APO.update_networks); None, no clip.
    max_grad_norm: Optional[float] = 10.0
