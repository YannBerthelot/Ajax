from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState


@struct.dataclass(kw_only=True)
class DAggerState(BaseAgentState):
    """DAgger carries its actor and collector; its critic is unused."""


@struct.dataclass(kw_only=True)
class DAggerConfig(BaseAgentConfig):
    """DAgger's iterations (Ross, Gordon and Bagnell, 2011, Alg. 3.1).

    Iteration ``i`` (from 1) spans ``iteration_steps`` timesteps, over all
    envs; its rollouts let the expert act with probability ``beta_decay **
    (i - 1)``, so ``beta_decay=0`` is the paper's parameter-free
    ``β_i = I(i = 1)``.
    """

    beta_decay: float = 0.0
    iteration_steps: int = 500
