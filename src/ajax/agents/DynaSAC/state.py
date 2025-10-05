from flax import struct

from ajax.agents.AVG.state import AVGConfig, AVGState
from ajax.agents.SAC.state import SACConfig, SACState


@struct.dataclass
class DynaSACState:
    """The agent properties to be carried over iterations of environment interaction and updates"""

    primary: SACState
    secondary: AVGState


@struct.dataclass
class DynaSACConfig:
    """The agent properties to be carried over iterations of environment interaction and updates"""

    primary: SACConfig
    secondary: AVGConfig
    SAC_length: int
    avg_length: int
    dyna_tau: float
    dyna_factor: float
    n_avg_agents: int
    actor_distillation_lr: float
    critic_distillation_lr: float
    n_distillation_samples: int
    alpha_polyak_primary_to_secondary: float = 1e-3
    initial_alpha_polyak_secondary_to_primary: float = 1e-3
    final_alpha_polyak_secondary_to_primary: float = 1e-3
    transition_mix_fraction: float = 0.5
    transfer_mode: str = "copy"
