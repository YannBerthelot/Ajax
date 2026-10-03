"""TD-MPC2 (Hansen et al., ICLR 2024), paper-era ``nicklashansen/tdmpc2@5f6fade``.

Milestone M2: the world model (:mod:`.networks`), the static configuration and
learner state (:mod:`.state`) and one training update as pure functions
(:mod:`.core`). The planner and the agent class come in later milestones; see
``docs/world_models/DESIGN.md`` §4.
"""

from ajax.agents.TDMPC2.core import (
    TDMPC2Batch,
    UpdateNoise,
    create_update_state,
    discount_from_episode_length,
    draw_update_noise,
    update,
)
from ajax.agents.TDMPC2.networks import MODEL_SIZE, resolve_model_size
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2UpdateState

__all__ = [
    "MODEL_SIZE",
    "TDMPC2Batch",
    "TDMPC2Config",
    "TDMPC2UpdateState",
    "UpdateNoise",
    "create_update_state",
    "discount_from_episode_length",
    "draw_update_noise",
    "resolve_model_size",
    "update",
]
