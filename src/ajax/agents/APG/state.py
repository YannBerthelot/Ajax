from typing import Optional

from flax import struct
from jax.tree_util import Partial as partial

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@partial(struct.dataclass, kw_only=True)
class APGState(BaseAgentState):
    """APG carries no value function: ``critic_state`` is ``None``.

    Keyword-only so the narrowed default can sit before the base class's
    required fields.
    """

    critic_state: Optional[LoadedTrainState] = None  # type: ignore[assignment]


@partial(struct.dataclass, kw_only=True)
class APGConfig(BaseAgentConfig):
    """Hyper-parameters of the analytic-policy-gradient trainer.

    ``horizon`` is the closed-loop rollout length ``N`` differentiated
    through per update. Every update samples ``n_envs`` systems (one per
    parallel env) and back-propagates the mean undiscounted return.
    """

    horizon: int = 100
