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

from ajax.agents.TDMPC2.core import (
    TASK_EMB,
    TaskContext,
    TDMPC2Batch,
    UpdateNoise,
    create_update_state,
    discount_from_episode_length,
    draw_update_noise,
    renorm_task_embedding,
    update,
)
from ajax.agents.TDMPC2.dataset import (
    MultiTaskDataset,
    TaskEpisodes,
    concatenate_episodes,
    export_episodes,
    load_dataset,
    pool_tasks,
    save_dataset,
)
from ajax.agents.TDMPC2.multitask import PAPER_TASK_DIM, TaskSet
from ajax.agents.TDMPC2.networks import MODEL_SIZE, resolve_model_size
from ajax.agents.TDMPC2.planner import (
    PlanInfo,
    PlanNoise,
    draw_plan_noise,
    plan,
    plan_from_latent,
)
from ajax.agents.TDMPC2.state import (
    TDMPC2Config,
    TDMPC2MultiTaskState,
    TDMPC2State,
    TDMPC2UpdateState,
)
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.agents.TDMPC2.TDMPC2MultiTask import TDMPC2MultiTask

__all__ = [
    "MODEL_SIZE",
    "MultiTaskDataset",
    "PAPER_TASK_DIM",
    "PlanInfo",
    "PlanNoise",
    "TASK_EMB",
    "TDMPC2",
    "TDMPC2Batch",
    "TDMPC2Config",
    "TDMPC2MultiTask",
    "TDMPC2MultiTaskState",
    "TDMPC2State",
    "TDMPC2UpdateState",
    "TaskContext",
    "TaskEpisodes",
    "TaskSet",
    "UpdateNoise",
    "concatenate_episodes",
    "create_update_state",
    "discount_from_episode_length",
    "draw_plan_noise",
    "draw_update_noise",
    "export_episodes",
    "load_dataset",
    "plan",
    "plan_from_latent",
    "pool_tasks",
    "renorm_task_embedding",
    "resolve_model_size",
    "save_dataset",
    "update",
]
