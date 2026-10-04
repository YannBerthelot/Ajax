"""Offline multi-task TD-MPC2 datasets: schema, export, pooling, sampling (M8).

The paper's multi-task models train offline on the pooled replay of
single-task agents (Sec. 4.1; tdmpc2_spec 4.18-4.19; ``DESIGN.md`` §7). This
module holds the data side, independent of the trainer:

* :class:`TaskEpisodes`: one task's whole episodes, unpadded (host numpy);
  :func:`export_episodes` reads them from a single-task
  :class:`~ajax.agents.TDMPC2.TDMPC2.TDMPC2` run's episode ring (every
  committed episode, in the order collected), :func:`concatenate_episodes`
  joins runs of the same task (several seeds);
* :class:`MultiTaskDataset`: the pooled, padded dataset the trainer samples,
  built by :func:`pool_tasks` (task ``i`` = the ``i``-th pooled task),
  written to and read from disk by :func:`save_dataset` and
  :func:`load_dataset` (an ``.npz`` of the arrays and a JSON metadata
  string, no pickle).

Schema (``N`` episodes of ``L`` rows; ``L = T + 1`` for whole episodes):

==========  ======================  ==========================================
field       shape                   content
==========  ======================  ==========================================
``obs``     ``[N, L, O_max]`` f32   observations, zero-padded at the end
                                    (``MultitaskWrapper._pad_obs``,
                                    ``envs/wrappers/multitask.py:44-47``)
``action``  ``[N, L, A_max]`` f32   actions, zero on the task's invalid
                                    (trailing) dims (spec 1.23)
``reward``  ``[N, L]`` f32          rewards
``task``    ``[N]`` int32           task id of each episode
==========  ======================  ==========================================

plus, per task, the observation dim, the action dim, the episode length ``T``
in agent steps (which sets its discount and evaluation episodes; it may exceed
``L - 1`` when the rows are segments of longer episodes, as the mt80 data
probably are, spec 4.18) and a name.

Rows are obs-aligned (``DESIGN.md`` §5.2, the layout of the single-task ring,
:mod:`ajax.agents.TDMPC2.buffer`): row ``k`` holds ``o_k``, the action ``a_k``
taken there and the reward ``r_{k-1}`` received on entering ``o_k``. Row 0 is
the reset row, whose reward slot is 0 (no reward led into it); the last row
has no action, its action slot is 0. The reference stores ``(o_k, a_{k-1},
r_{k-1})`` with dummies in row 0 (b67b21c ``online_trainer.py:50-65``) and
drops them when slicing (``buffer.py:26-28``); the slice conversion of
:func:`ajax.agents.TDMPC2.buffer.slice_batch` (obs rows ``s .. s + H``, action
rows ``s .. s + H - 1``, reward rows ``s + 1 .. s + H``) yields the same
transitions from this layout, so the placeholder slots are never read.

Sampling (:meth:`MultiTaskDataset.sample`) is b67b21c's (``buffer.py:60-70``):
an episode uniformly over the *pooled* episodes (``RandomSampler``, so a
task's share of the batch is its share of the episodes, no per-task
balancing; spec 4.19) and a uniform crop of ``H + 1`` rows
(``RandomCropTensorDict``); each slice is single-task.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ajax.agents.TDMPC2.buffer import EpisodeBuffer, EpisodeBufferState, slice_batch
from ajax.agents.TDMPC2.core import TDMPC2Batch

if TYPE_CHECKING:
    from ajax.agents.TDMPC2.TDMPC2 import TDMPC2


@dataclasses.dataclass(frozen=True, eq=False)
class TaskEpisodes:
    """Whole episodes of one task, unpadded (host numpy, obs-aligned rows).

    Attributes:
        obs: ``[N, L, obs_dim]`` float32.
        action: ``[N, L, action_dim]`` float32 (row ``L - 1`` is 0).
        reward: ``[N, L]`` float32 (row 0 is 0).
        episode_length: the task's episode length ``T`` in agent steps.
        name: the task's name.
    """

    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    episode_length: int
    name: str = "task"

    def __post_init__(self) -> None:
        n, rows = self.obs.shape[:2] if self.obs.ndim == 3 else (-1, -1)
        if (
            self.obs.ndim != 3
            or self.action.ndim != 3
            or self.action.shape[:2] != (n, rows)
            or self.reward.shape != (n, rows)
        ):
            raise ValueError(
                "episodes need obs [N, L, O], action [N, L, A] and reward [N, L],"
                f" got {self.obs.shape}, {self.action.shape}, {self.reward.shape}"
            )
        if n < 1 or rows < 2 or self.episode_length < rows - 1:
            raise ValueError(
                f"need >= 1 episode of >= 2 rows and episode_length >= L - 1,"
                f" got {n} episodes of {rows} rows, episode_length"
                f" {self.episode_length}"
            )

    @property
    def num_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def rows(self) -> int:
        """Rows per episode ``L``."""
        return int(self.obs.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[-1])


def _host(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def export_episodes(
    agent: TDMPC2,
    state: Any,
    seed_index: Optional[int] = None,
    *,
    name: str = "task",
    allow_evicted: bool = False,
) -> TaskEpisodes:
    """The committed episodes of a single-task TD-MPC2 run, oldest first.

    Reads ``state.buffer_state`` (the episode ring of ``agent``'s run,
    :mod:`ajax.agents.TDMPC2.buffer`) after the ticks its collector has run:
    every complete episode the ring holds, in the order collected (rounds in
    order, env order within a round; the episode in progress is not
    exported). Its rows are the ring's obs-aligned rows, unchanged.

    Args:
        agent: the :class:`~ajax.agents.TDMPC2.TDMPC2.TDMPC2` that trained
            ``state`` (its ``n_envs`` and episode length).
        state: the run's state, as ``agent.train`` returned it; with a
            leading seed axis, ``seed_index`` picks the seed.
        seed_index: the seed to export, required for a seed-batched state.
        name: the task's name.
        allow_evicted: export what the ring still holds when it wrapped and
            evicted older episodes. By default this raises: a dataset of the
            full training history (``DESIGN.md`` §7) needs ``buffer_size`` at
            least the run length.

    Raises:
        ValueError: on a missing or extraneous ``seed_index``, when no
            episode is complete, or when episodes were evicted (unless
            ``allow_evicted``).
    """
    obs = _host(state.buffer_state.obs)
    action = _host(state.buffer_state.action)
    reward = _host(state.buffer_state.reward)
    rows = _host(state.collector_state.rows)
    if obs.ndim == 4:
        if seed_index is None:
            raise ValueError(
                f"the state holds {obs.shape[0]} seeds: pass seed_index to pick one"
            )
        obs, action, reward = obs[seed_index], action[seed_index], reward[seed_index]
        rows = rows[seed_index]
    elif seed_index is not None:
        raise ValueError("seed_index given, but the state has no seed axis")
    n_envs = agent.env_args.n_envs
    ring = EpisodeBuffer(
        n_envs=n_envs,
        episode_length=agent.agent_episode_length,
        n_rounds=obs.shape[0] // n_envs,
        obs_dim=obs.shape[-1],
        action_dim=action.shape[-1],
    )
    ring.check_layout(EpisodeBufferState(obs=obs, action=action, reward=reward))
    ticks = int(rows) // n_envs
    slots, evicted = ring.committed_slots(ticks - 1)
    if slots.size == 0:
        raise ValueError(
            f"no complete episode to export after {ticks} ticks"
            f" (T = {ring.episode_length})"
        )
    if evicted and not allow_evicted:
        raise ValueError(
            f"the replay ring evicted the {evicted} oldest episodes of the run"
            " (buffer_size below the run length): pass allow_evicted=True to"
            f" export the {slots.size} it holds"
        )
    return TaskEpisodes(
        obs=obs[slots],
        action=action[slots],
        reward=reward[slots],
        episode_length=ring.episode_length,
        name=name,
    )


def concatenate_episodes(episodes: Sequence[TaskEpisodes]) -> TaskEpisodes:
    """The episodes of several runs of one task (e.g. seeds), in order.

    They must agree on the rows, dims, episode length and name.
    """
    if not episodes:
        raise ValueError("nothing to concatenate")
    first = episodes[0]
    for other in episodes[1:]:
        if (
            other.rows,
            other.obs_dim,
            other.action_dim,
            other.episode_length,
            other.name,
        ) != (
            first.rows,
            first.obs_dim,
            first.action_dim,
            first.episode_length,
            first.name,
        ):
            raise ValueError(
                f"cannot concatenate episodes of {other.name!r} (rows {other.rows},"
                f" dims {other.obs_dim}/{other.action_dim}, T"
                f" {other.episode_length}) with {first.name!r} (rows {first.rows},"
                f" dims {first.obs_dim}/{first.action_dim}, T"
                f" {first.episode_length})"
            )
    return TaskEpisodes(
        obs=np.concatenate([e.obs for e in episodes]),
        action=np.concatenate([e.action for e in episodes]),
        reward=np.concatenate([e.reward for e in episodes]),
        episode_length=first.episode_length,
        name=first.name,
    )


@struct.dataclass
class MultiTaskDataset:
    """A pooled multi-task dataset (module docstring for the schema).

    A pytree of the four arrays; the per-task metadata is static. The
    offline trainer passes it to its jitted, seed-vmapped program as one
    unbatched argument (``build_resumable_train``'s ``shared`` input).

    Attributes:
        obs: ``[N, L, O_max]`` float32, zero-padded at the end.
        action: ``[N, L, A_max]`` float32, zero on invalid dims.
        reward: ``[N, L]`` float32.
        task: ``[N]`` int32 task ids.
        obs_dims: per-task observation dims.
        action_dims: per-task action dims (the leading dims).
        episode_lengths: per-task episode lengths ``T`` (agent steps).
        names: per-task names.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    task: jax.Array
    obs_dims: tuple[int, ...] = struct.field(pytree_node=False)
    action_dims: tuple[int, ...] = struct.field(pytree_node=False)
    episode_lengths: tuple[int, ...] = struct.field(pytree_node=False)
    names: tuple[str, ...] = struct.field(pytree_node=False)

    @property
    def num_tasks(self) -> int:
        return len(self.obs_dims)

    @property
    def num_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def rows(self) -> int:
        """Rows per episode ``L``."""
        return int(self.obs.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[-1])

    @property
    def nbytes(self) -> int:
        return sum(
            int(x.nbytes) for x in (self.obs, self.action, self.reward, self.task)
        )

    def episode_counts(self) -> np.ndarray:
        """Episodes per task (host): each task's expected share of a batch."""
        return np.bincount(_host(self.task), minlength=self.num_tasks)

    def check(self) -> None:
        """Raise ``ValueError`` unless the arrays follow the schema (host).

        Shapes, dtypes and metadata agree; every task id is valid and every
        task has episodes; the padding is exactly zero: observation dims
        beyond the task's (``_pad_obs``) and action dims beyond its
        ``action_dims`` (the reference feeds the stored actions unmasked to
        the dynamics, reward and Q, spec 1.23, open question 1.B).
        """
        n, rows = self.num_episodes, self.rows
        sizes = {
            len(self.obs_dims),
            len(self.action_dims),
            len(self.episode_lengths),
            len(self.names),
        }
        if len(sizes) != 1 or not self.obs_dims:
            raise ValueError("one obs dim, action dim, T and name per task")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"task names must be unique, got {self.names}")
        expected = {
            "obs": ((n, rows, max(self.obs_dims)), np.float32),
            "action": ((n, rows, max(self.action_dims)), np.float32),
            "reward": ((n, rows), np.float32),
            "task": ((n,), np.int32),
        }
        for field, (shape, dtype) in expected.items():
            value = getattr(self, field)
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(
                    f"dataset {field} is {value.dtype}{list(value.shape)},"
                    f" expected {np.dtype(dtype)}{list(shape)}"
                )
        if rows < 2 or min(self.episode_lengths) < rows - 1:
            raise ValueError(
                f"episodes of {rows} rows need >= 2 rows and every task's"
                f" T >= L - 1, got T = {self.episode_lengths}"
            )
        task = _host(self.task)
        if np.any((task < 0) | (task >= self.num_tasks)):
            raise ValueError(f"task ids must be in [0, {self.num_tasks})")
        if np.any(self.episode_counts() == 0):
            raise ValueError(
                f"every task needs episodes, got counts {self.episode_counts()}"
            )
        for field, dims in (("obs", self.obs_dims), ("action", self.action_dims)):
            values = _host(getattr(self, field))
            valid = np.arange(values.shape[-1])[None] < np.asarray(dims)[task][:, None]
            if np.any(np.where(valid[:, None, :], 0.0, values) != 0):
                raise ValueError(
                    f"dataset {field} must be zero beyond each task's"
                    f" {field} dims {dims}"
                )

    def sample(
        self, key: jax.Array, batch_size: int, horizon: int
    ) -> tuple[TDMPC2Batch, jax.Array]:
        """``batch_size`` sub-trajectories of ``horizon + 1`` rows and their
        task ids ``[B]``.

        Each draws an episode uniformly over the pooled episodes and an
        offset uniformly in ``[0, L - 1 - H]``, independently (b67b21c
        ``RandomSampler`` + ``RandomCropTensorDict(H + 1)``, ``buffer.py:
        60-70``), then :func:`~ajax.agents.TDMPC2.buffer.slice_batch`; the
        task id is the episode's (``td['task'][0]``, ``buffer.py:28``).
        Jittable; unbatched under a seed ``vmap`` the dataset is gathered
        from, never copied.
        """
        if horizon > self.rows - 1:
            raise ValueError(
                f"horizon {horizon} needs episodes of at least horizon + 1 rows,"
                f" got {self.rows}"
            )
        episode_key, start_key = jax.random.split(key)
        episode = jax.random.randint(episode_key, (batch_size,), 0, self.num_episodes)
        start = jax.random.randint(start_key, (batch_size,), 0, self.rows - horizon)
        store = EpisodeBufferState(obs=self.obs, action=self.action, reward=self.reward)
        return slice_batch(store, episode, start, horizon), self.task[episode]

    def task_episodes(self, task: int) -> TaskEpisodes:
        """Task ``task``'s episodes, unpadded (host): the inverse of
        :func:`pool_tasks` for one task."""
        index = np.flatnonzero(_host(self.task) == task)
        return TaskEpisodes(
            obs=_host(self.obs)[index][..., : self.obs_dims[task]],
            action=_host(self.action)[index][..., : self.action_dims[task]],
            reward=_host(self.reward)[index],
            episode_length=self.episode_lengths[task],
            name=self.names[task],
        )


def _pad(x: np.ndarray, size: int) -> np.ndarray:
    pad = [(0, 0)] * (x.ndim - 1) + [(0, size - x.shape[-1])]
    return np.pad(x.astype(np.float32), pad)


def pool_tasks(tasks: Sequence[TaskEpisodes]) -> MultiTaskDataset:
    """The multi-task dataset of ``tasks``, task ``i`` being ``tasks[i]``.

    Observations and actions are zero-padded at the end to the largest dims
    (``MultitaskWrapper``, spec 1.23, 4.16); episodes keep their order,
    task after task. Every task's episodes must have the same number of rows
    (cutting longer episodes into segments, as the mt80 data probably did,
    spec 4.18, is not done here). Names must be unique. The arrays are
    placed on the default device once.
    """
    if not tasks:
        raise ValueError("pool at least one task")
    rows = {t.rows for t in tasks}
    if len(rows) != 1:
        raise ValueError(
            "every task's episodes must have the same number of rows, got"
            f" {[t.rows for t in tasks]}"
        )
    obs_dim = max(t.obs_dim for t in tasks)
    action_dim = max(t.action_dim for t in tasks)
    dataset = MultiTaskDataset(
        obs=jnp.asarray(np.concatenate([_pad(t.obs, obs_dim) for t in tasks])),
        action=jnp.asarray(np.concatenate([_pad(t.action, action_dim) for t in tasks])),
        reward=jnp.asarray(
            np.concatenate([t.reward.astype(np.float32) for t in tasks])
        ),
        task=jnp.asarray(
            np.concatenate(
                [np.full(t.num_episodes, i, np.int32) for i, t in enumerate(tasks)]
            )
        ),
        obs_dims=tuple(t.obs_dim for t in tasks),
        action_dims=tuple(t.action_dim for t in tasks),
        episode_lengths=tuple(t.episode_length for t in tasks),
        names=tuple(t.name for t in tasks),
    )
    dataset.check()
    return dataset


#: Identifies the file format of :func:`save_dataset` (bump the version on
#: any change of the stored fields).
DATASET_FORMAT = "ajax.tdmpc2.multitask_dataset"
DATASET_FORMAT_VERSION = 1


def save_dataset(dataset: MultiTaskDataset, path: str) -> None:
    """Write ``dataset`` to ``path`` (an uncompressed NumPy ``.npz``).

    The file holds the four arrays under their field names and ``meta``, a
    JSON string with the format name and version and the per-task metadata
    (obs dims, action dims, episode lengths, names): plain arrays and text,
    readable without pickle (:func:`load_dataset`). The dataset is checked
    first (:meth:`MultiTaskDataset.check`) and the file is written to a
    temporary name and renamed, so an interrupted save leaves no partial
    file under ``path``.
    """
    dataset.check()
    meta = {
        "format": DATASET_FORMAT,
        "version": DATASET_FORMAT_VERSION,
        "obs_dims": list(dataset.obs_dims),
        "action_dims": list(dataset.action_dims),
        "episode_lengths": list(dataset.episode_lengths),
        "names": list(dataset.names),
    }
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez(
            f,
            obs=_host(dataset.obs),
            action=_host(dataset.action),
            reward=_host(dataset.reward),
            task=_host(dataset.task),
            meta=np.asarray(json.dumps(meta)),
        )
    os.replace(tmp, path)


def load_dataset(path: str) -> MultiTaskDataset:
    """The dataset :func:`save_dataset` wrote to ``path``, checked.

    The arrays stay host NumPy arrays (``TDMPC2MultiTask`` places the
    dataset on the device once). Raises ``ValueError`` on another format or
    version, or when the data do not follow the schema.
    """
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        if (meta.get("format"), meta.get("version")) != (
            DATASET_FORMAT,
            DATASET_FORMAT_VERSION,
        ):
            raise ValueError(
                f"{path} is not a {DATASET_FORMAT} v{DATASET_FORMAT_VERSION} file"
                f" (format {meta.get('format')!r}, version {meta.get('version')!r})"
            )
        dataset = MultiTaskDataset(
            obs=data["obs"],
            action=data["action"],
            reward=data["reward"],
            task=data["task"],
            obs_dims=tuple(int(d) for d in meta["obs_dims"]),
            action_dims=tuple(int(d) for d in meta["action_dims"]),
            episode_lengths=tuple(int(t) for t in meta["episode_lengths"]),
            names=tuple(str(n) for n in meta["names"]),
        )
    dataset.check()
    return dataset


__all__ = [
    "DATASET_FORMAT",
    "DATASET_FORMAT_VERSION",
    "MultiTaskDataset",
    "TaskEpisodes",
    "concatenate_episodes",
    "export_episodes",
    "load_dataset",
    "pool_tasks",
    "save_dataset",
]
