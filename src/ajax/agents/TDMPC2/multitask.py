"""TD-MPC2 multi-task mechanisms (M8), paper-era ``nicklashansen/tdmpc2@5f6fade``.

The multi-task machinery of the paper (Hansen et al., ICLR 2024, Sec. 3.3,
App. A; ``docs/world_models/DESIGN.md`` §7; tdmpc2_spec 1.21-1.23, 2.18, 2.20,
3.9, 3.25, 4.15-4.20) as data indexed by task id, built on the single-task
functions (the lineage rule):

* the **task set** (:class:`TaskSet`): per-task observation and action dims,
  episode lengths (agent steps) and discounts, in task-id order. Observations
  are zero-padded *at the end* to the largest dim (``MultitaskWrapper._pad_obs``,
  ``envs/wrappers/multitask.py:44-47``; :func:`pad_observation`); actions
  live in the largest action space, task ``i`` using its ``action_dims[i]``
  *leading* dims (prefix masks, ``world_model.py:21-23``), and the env gets
  that prefix (``multitask.py:55-57``). The discount table holds each task's
  ``discount_from_episode_length(T_i)`` (``tdmpc2.py:32-34``, spec 2.20);
* the **embedding table** ``[num_tasks, task_dim]``, a world-model parameter
  (``wm_params[core.TASK_EMB]``, created by :func:`create_update_state`),
  ``U(-0.02, 0.02)``, ``task_dim = 96`` (paper Table 8-9; deviation T14:
  the code's 64 for mt30 at 5/19/48M only serves the released checkpoints),
  max-norm 1 by look-up-time renorm with write-back
  (:func:`ajax.agents.TDMPC2.core.renorm_task_embedding`);
* the **task context** of a batch or decision
  (:class:`ajax.agents.TDMPC2.core.TaskContext`: task ids and their action
  masks, :meth:`TaskSet.context`), which :func:`update` and :func:`plan`
  pass to the single-task :func:`ajax.agents.TDMPC2.core.update` and
  :func:`ajax.agents.TDMPC2.planner.plan` together with the per-task
  discounts.

Multi-task TD-MPC2 is offline only at the paper era (``train.py:52``; spec
3.25, 4.1): it trains from a dataset (:mod:`ajax.agents.TDMPC2.dataset`) with
the offline trainer (:class:`ajax.agents.TDMPC2.TDMPC2MultiTask.TDMPC2MultiTask`)
and plans only to evaluate, each task in its own env
(:func:`task_planner_policy`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.TDMPC2 import core, planner
from ajax.agents.TDMPC2.state import TDMPC2Config
from ajax.agents.TDMPC2.train_TDMPC2 import planner_policy
from ajax.types import FloatOrCallable

# Paper Table 8-9 (deviation T14: the code's 64 for mt30 at 5/19/48M).
PAPER_TASK_DIM = 96

IntLike = Union[int, jax.Array]


@dataclasses.dataclass(frozen=True)
class TaskSet:
    """The tasks of a multi-task run, in task-id order (static, hashable).

    Build it with :meth:`create`, which derives the discounts from the
    episode lengths as the reference does.

    Attributes:
        obs_dims: each task's observation dim; the model sees them
            zero-padded to :attr:`obs_dim`.
        action_dims: each task's action dim, the leading dims of the
            :attr:`action_dim`-dim padded action space.
        episode_lengths: each task's episode length ``T`` in agent steps.
        discounts: each task's discount (TD targets and planning).
        names: task names (logging).
    """

    obs_dims: tuple[int, ...]
    action_dims: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    discounts: tuple[float, ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        sizes = {
            len(self.obs_dims),
            len(self.action_dims),
            len(self.episode_lengths),
            len(self.discounts),
            len(self.names),
        }
        if len(sizes) != 1 or not self.obs_dims:
            raise ValueError(
                "a task set needs >= 1 task and one obs dim, action dim, episode"
                f" length, discount and name per task, got {self}"
            )
        dims = self.obs_dims + self.action_dims + self.episode_lengths
        if min(dims) < 1:
            raise ValueError(f"dims and episode lengths must be >= 1, got {self}")
        if not all(0.0 < d <= 1.0 for d in self.discounts):
            raise ValueError(f"discounts must be in (0, 1], got {self.discounts}")

    @classmethod
    def create(
        cls,
        obs_dims: Sequence[int],
        action_dims: Sequence[int],
        episode_lengths: Sequence[int],
        *,
        names: Optional[Sequence[str]] = None,
        discount_denom: float = 5.0,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
    ) -> TaskSet:
        """The task set with the reference's per-task discounts.

        ``discount_from_episode_length(T_i, denom, min, max)`` per task
        (``tdmpc2.py:32-49``, spec 2.20; e.g. DMC at action repeat 2,
        ``T = 500``: 0.99). ``names`` default to ``task0``, ``task1``, ...
        """
        lengths = tuple(int(t) for t in episode_lengths)
        return cls(
            obs_dims=tuple(int(d) for d in obs_dims),
            action_dims=tuple(int(d) for d in action_dims),
            episode_lengths=lengths,
            discounts=tuple(
                core.discount_from_episode_length(
                    t, discount_denom, discount_min, discount_max
                )
                for t in lengths
            ),
            names=(
                tuple(f"task{i}" for i in range(len(lengths)))
                if names is None
                else tuple(str(n) for n in names)
            ),
        )

    @property
    def num_tasks(self) -> int:
        return len(self.obs_dims)

    @property
    def obs_dim(self) -> int:
        """The padded observation dim, ``max(obs_dims)``."""
        return max(self.obs_dims)

    @property
    def action_dim(self) -> int:
        """The padded action dim, ``max(action_dims)``; the planner's
        iteration rule uses it (spec 3.1: 6 iterations for mt30 / mt80)."""
        return max(self.action_dims)

    @property
    def action_masks(self) -> np.ndarray:
        """``float32 [num_tasks, action_dim]``: ``M[i, :action_dims[i]] = 1``
        (``world_model.py:21-23``)."""
        dims = np.asarray(self.action_dims)[:, None]
        return (np.arange(self.action_dim)[None] < dims).astype(np.float32)

    @property
    def discount_table(self) -> np.ndarray:
        """``float32 [num_tasks]``, the reference's ``torch.tensor`` of the
        per-task discounts (``tdmpc2.py:32-34``)."""
        return np.asarray(self.discounts, np.float32)

    def context(self, task: IntLike) -> core.TaskContext:
        """The :class:`~ajax.agents.TDMPC2.core.TaskContext` of task ids
        ``task`` (``[B]`` for a batch, a scalar for one decision)."""
        ids = jnp.asarray(task, jnp.int32)
        return core.TaskContext(ids=ids, mask=jnp.asarray(self.action_masks)[ids])

    def discount(self, task: IntLike) -> jax.Array:
        """The discounts of task ids ``task``: ``discount[task]``, float32."""
        return jnp.asarray(self.discount_table)[jnp.asarray(task, jnp.int32)]


def pad_observation(obs: jax.Array, obs_dim: int) -> jax.Array:
    """``obs [..., d]`` zero-padded *at the end* to ``[..., obs_dim]``
    (``MultitaskWrapper._pad_obs``, ``envs/wrappers/multitask.py:44-47``)."""
    pad = obs_dim - obs.shape[-1]
    if pad < 0:
        raise ValueError(f"cannot pad an observation of {obs.shape[-1]} to {obs_dim}")
    if pad == 0:
        return obs
    return jnp.concatenate([obs, jnp.zeros((*obs.shape[:-1], pad), obs.dtype)], -1)


def create_update_state(
    key: jax.Array,
    config: TDMPC2Config,
    tasks: TaskSet,
    *,
    task_dim: int = PAPER_TASK_DIM,
    learning_rate: FloatOrCallable = 3e-4,
    enc_lr_scale: float = 0.3,
    pi_eps: float = 1e-5,
) -> core.TDMPC2UpdateState:
    """:func:`ajax.agents.TDMPC2.core.create_update_state` for ``tasks``:
    padded input widths, the ``[num_tasks, task_dim]`` embedding table in the
    world-model parameters."""
    return core.create_update_state(
        key,
        config,
        tasks.obs_dim,
        tasks.action_dim,
        learning_rate=learning_rate,
        enc_lr_scale=enc_lr_scale,
        pi_eps=pi_eps,
        task_dim=task_dim,
        num_tasks=tasks.num_tasks,
    )


S = TypeVar("S", bound=core.UpdateStateLike)


def update(
    state: S,
    batch: core.TDMPC2Batch,
    task: jax.Array,
    noise: core.UpdateNoise,
    *,
    config: TDMPC2Config,
    tasks: TaskSet,
) -> tuple[S, dict[str, jax.Array]]:
    """One multi-task update: the single-task update conditioned on the
    batch's tasks (``tdmpc2.py:218-290`` with ``task``).

    ``batch`` holds padded observations and actions (zero on each sample's
    invalid dims), ``task [B]`` each slice's task id; the TD targets use
    ``discount[task]`` per sample (``tdmpc2.py:215``); the embedding rows of
    the batch are renormed and written back before the TD target and after
    the world-model step (:func:`ajax.agents.TDMPC2.core.update`). ``noise``
    is drawn for the padded action dim.
    """
    return core.update(
        state,
        batch,
        noise,
        config=config,
        gamma=tasks.discount(task),
        task=tasks.context(task),
    )


def plan(
    wm_params: Any,
    pi_params: Any,
    obs: jax.Array,
    prev_mean: jax.Array,
    t0: Union[bool, jax.Array],
    noise: planner.PlanNoise,
    task: IntLike,
    *,
    config: TDMPC2Config,
    tasks: TaskSet,
    eval_mode: bool,
) -> tuple[jax.Array, jax.Array, planner.PlanInfo]:
    """One MPPI decision for task ``task``: ``act(obs, t0, eval_mode, task)``
    (``tdmpc2.py:70-171``).

    :func:`ajax.agents.TDMPC2.planner.plan` with the task's context and
    discount; ``obs [obs_dim]`` is padded, ``prev_mean [H, A]`` and the
    returned action are in the padded action space (invalid dims exactly 0).
    The embedding row is renormed for the decision, not persisted (T15).
    """
    return planner.plan(
        wm_params,
        pi_params,
        obs,
        prev_mean,
        t0,
        noise,
        config=config,
        gamma=tasks.discount(task),
        eval_mode=eval_mode,
        task=tasks.context(task),
    )


def task_planner_policy(
    carry: jax.Array,
    obs: jax.Array,
    is_first: jax.Array,
    key: jax.Array,
    *,
    wm_params: Any,
    pi_params: Any,
    config: TDMPC2Config,
    tasks: TaskSet,
    task: int,
    eval_mode: bool,
) -> tuple[jax.Array, jax.Array, None]:
    """The planner acting in task ``task``'s own env, for
    :func:`ajax.evaluate.evaluate_policy` (the row-collector protocol).

    The single-task :func:`ajax.agents.TDMPC2.train_TDMPC2.planner_policy`
    with the task's context and discount: ``obs [m, obs_dims[task]]`` is
    zero-padded, each env plans with its own draws and warm start ``carry =
    prev_mean [m, H, A_max]`` (``t0 = is_first``), and the env gets the
    prefix ``action[:, :action_dims[task]]`` (``MultitaskWrapper.step``,
    ``envs/wrappers/multitask.py:55-57``).
    """
    action, prev_mean, _ = planner_policy(
        carry,
        pad_observation(obs, tasks.obs_dim),
        is_first,
        key,
        wm_params=wm_params,
        pi_params=pi_params,
        config=config,
        gamma=tasks.discount(task),
        eval_mode=eval_mode,
        task=tasks.context(task),
    )
    return action[:, : tasks.action_dims[task]], prev_mean, None


__all__ = [
    "PAPER_TASK_DIM",
    "TaskSet",
    "create_update_state",
    "pad_observation",
    "plan",
    "task_planner_policy",
    "update",
]
