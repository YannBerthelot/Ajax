from flax import struct

from ajax.agents.AVG.state import AVGConfig, AVGState
from ajax.agents.sac.state import SACConfig, SACState


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
    sac_length: int
    avg_length: int
    dyna_tau: float
    dyna_factor: float
