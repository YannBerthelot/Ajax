"""TD-MPC2 (Hansen et al., ICLR 2024), paper-era ``nicklashansen/tdmpc2@5f6fade``.

The world model (:mod:`.networks`), the static configuration and states
(:mod:`.state`), one training update (:mod:`.core`) and the MPPI planner
(:mod:`.planner`) as pure functions; the b67b21c episode replay buffer
(:mod:`.buffer`), the online training loop (:mod:`.train_TDMPC2`) and the
agent class :class:`TDMPC2` (:mod:`.TDMPC2`). See
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
from ajax.agents.TDMPC2.planner import (
    PlanInfo,
    PlanNoise,
    draw_plan_noise,
    plan,
    plan_from_latent,
)
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2State, TDMPC2UpdateState
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2

__all__ = [
    "MODEL_SIZE",
    "PlanInfo",
    "PlanNoise",
    "TDMPC2",
    "TDMPC2Batch",
    "TDMPC2Config",
    "TDMPC2State",
    "TDMPC2UpdateState",
    "UpdateNoise",
    "create_update_state",
    "discount_from_episode_length",
    "draw_plan_noise",
    "draw_update_noise",
    "plan",
    "plan_from_latent",
    "resolve_model_size",
    "update",
]
