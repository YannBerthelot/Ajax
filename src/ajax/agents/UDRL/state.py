from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState

UDRLState = BaseAgentState


@struct.dataclass
class UDRLConfig(BaseAgentConfig):
    """Hyperparameters for Upside-Down RL (Schmidhuber, 2019)."""

    gamma: float = 1.0
    n_steps: int = 64
    batch_size: int = 64
    n_epochs: int = 4
    command_return_init: float = 1.0
    command_horizon_init: float = 100.0
    bc_loss_type: str = "nll"  # "nll" or "mse"
