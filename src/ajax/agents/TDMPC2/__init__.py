"""TD-MPC2 (Hansen et al., ICLR 2024), paper-era ``nicklashansen/tdmpc2@5f6fade``.

The world model (:mod:`.networks`), the static configuration and states
(:mod:`.state`), one training update (:mod:`.core`) and the MPPI planner
(:mod:`.planner`) as pure functions; the b67b21c episode replay buffer
(:mod:`.buffer`), the online training loop (:mod:`.train_TDMPC2`) and the
agent class :class:`TDMPC2` (:mod:`.TDMPC2`). Multi-task (M8): the
task-conditioning mechanisms (:mod:`.multitask`), the offline datasets
(:mod:`.dataset`), the offline training loop (:mod:`.train_TDMPC2MultiTask`)
and the agent class :class:`TDMPC2MultiTask` (:mod:`.TDMPC2MultiTask`). See
``docs/world_models/DESIGN.md`` §4 and §7.
"""

from ajax.agents.TDMPC2.state import TDMPC2Config
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.agents.TDMPC2.TDMPC2MultiTask import TDMPC2MultiTask

__all__ = ["TDMPC2", "TDMPC2Config", "TDMPC2MultiTask"]
