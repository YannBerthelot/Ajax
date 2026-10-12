"""TD-MPC2 episode replay buffer: the b67b21c design, on device and fixed-shape.

The buffer behind the paper's curves (``nicklashansen/tdmpc2@b67b21c``,
``tdmpc2/common/buffer.py``; tdmpc2_spec 4.7-4.10, ``DESIGN.md`` §4.4) stores
whole episodes of ``T + 1`` rows, draws an episode uniformly with replacement
(``RandomSampler``, ``buffer.py:62``) and a sub-trajectory of ``H + 1`` rows at a
uniform offset inside it (``RandomCropTensorDict(H + 1)``, ``buffer.py:66``),
then drops the first row's action and reward (``DataPrepTransform``,
``buffer.py:26-28``). Its capacity is ``min(buffer_size, steps) // T`` episodes
(``buffer.py:40``); only finished episodes are added (``online_trainer.py:91``),
so a sub-trajectory never crosses an episode and never reads the running one.

Layout. With ``n`` lockstep envs and the row collector's static schedule
(:mod:`ajax.environments.row_collector`: tick ``i``, period ``P = T + 1``, held
tick ``i mod P == T``), tick ``i`` writes row ``i mod P`` of *round*
``i div P`` (``n`` episodes, one per env). The storage is a ring of ``R``
rounds::

    obs    [R * n, T + 1, obs_dim]     float32
    action [R * n, T + 1, A]           float32
    reward [R * n, T + 1]              float32

episode slot ``(round mod R) * n + env``. The rows of the round in progress
are written in place; a round becomes sampleable once its held tick (row
``T``) is written. ``R = ceil(capacity / (T n)) + 1``: ``R - 1`` committed
rounds hold at least ``capacity`` env steps and the extra *staging* round is
the one being written, never sampled (when the ring is full it holds the
oldest round, being overwritten). Which rounds are committed is static
arithmetic on the absolute tick (:meth:`EpisodeBuffer.committed_rounds`), the
same on every seed.

Rows are obs-aligned (``DESIGN.md`` §5.2): row ``k`` holds ``o_k``, the action
``a_k`` taken there (zero on the final row ``T``) and the reward ``r_{k-1}``
received on entering ``o_k`` (zero on row 0). The reference's rows hold
``(o_k, a_{k-1}, r_{k-1})`` with a dummy row 0 (``online_trainer.py:50-65``).
The slice at offset ``s`` is therefore obs rows ``s .. s + H``, action rows
``s .. s + H - 1`` and reward rows ``s + 1 .. s + H``: the same
``(o_s..o_{s+H}, a_s..a_{s+H-1}, r_s..r_{s+H-1})`` as ``DataPrepTransform``.

Capacity (registered as T26 in ``deviations.md``): b67b21c holds
``floor(min(buffer_size, steps) / T)`` episodes; Ajax holds
``ceil(min(buffer_size, total) / (T n))`` rounds of ``n`` episodes, ``total``
being the env steps the run has taken when the current ``train()`` call ends.
They agree whenever ``buffer_size`` is a multiple of ``T n`` (DMC: 1e6 / 500)
and whenever the run never fills the buffer. The clamp to the run length only
saves memory: a ring that holds every step of the run never evicts, so it
behaves as a ring of ``buffer_size``. A resumed run may need a larger ring than
the one it carries (a run trained in chunks); :meth:`EpisodeBuffer.adopt`
moves the carried rounds into it, so a chunked run replays exactly what an
uninterrupted one does.

Episodic mode (:class:`EpisodicBuffer`; deviation T29): episodes end at
data-dependent ticks. The latest code's replay (``e9f5932:tdmpc2/common/
buffer.py``, torchrl 0.8.1's ``SliceSampler(traj_key='episode',
strict_length=True)``) draws a finished episode uniformly among those of at
least ``H + 1`` rows, a uniform offset inside it, and adds ``terminated``
(rows ``s + 1 .. s + H``; a time limit is not terminal, so it bootstraps).
:class:`EpisodicBuffer` does the same on per-env rings of ``C`` rows written
in lockstep by the row collector's dynamic mode (tick ``i`` at slot ``i mod
C``). An overwritten episode keeps its stored tail, as in the reference. As
the reference's storage of ``min(buffer_size, steps)`` rows, the rings hold
``C = ceil(min(buffer_size, total) / n)`` rows per env, ``total`` being the
rows the run has written when the current ``train()`` call ends: a ring
that holds every row of the run never overwrites one, so it behaves as the
unclamped ring. A resumed run moves the carried rows into the rings of its
own call (:meth:`EpisodicBuffer.adopt`).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ajax.agents.TDMPC2.core import IntLike, TDMPC2Batch
from ajax.environments.row_collector import Row


def replay_capacity(buffer_size: int, total_timesteps: int) -> int:
    """Replay capacity in env steps of a run that has taken ``total_timesteps``
    env steps when its current ``train()`` call ends: ``min(buffer_size,
    total_timesteps)``.

    The reference clamps its buffer to the run length (``buffer.py:40``,
    tdmpc2_spec 4.7; ``DESIGN.md`` §5.6), which saves memory without changing
    what is replayed. A run of 0 steps (the checkpoint skeleton,
    ``ajax.checkpoint``) gets the smallest ring (capacity 1): the resumed run
    adopts the restored ring (:meth:`EpisodeBuffer.adopt`).
    """
    if buffer_size < 1:
        raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
    if total_timesteps < 0:
        raise ValueError(f"total_timesteps must be >= 0, got {total_timesteps}")
    return max(min(buffer_size, total_timesteps), 1)


@struct.dataclass
class EpisodeBufferState:
    """The ring's storage (module docstring); leading axis ``R * n_envs``."""

    obs: jax.Array
    action: jax.Array
    reward: jax.Array


@dataclasses.dataclass(frozen=True)
class EpisodeBuffer:
    """Static description of the episode ring and its pure operations.

    Attributes:
        n_envs: lockstep envs ``n``; one episode slot per env and round.
        episode_length: ``T``, agent steps per episode (``T + 1`` rows).
        n_rounds: ``R``, rounds in the ring, the staging round included
            (>= 2).
        obs_dim: flat observation size.
        action_dim: action size ``A``.
    """

    n_envs: int
    episode_length: int
    n_rounds: int
    obs_dim: int
    action_dim: int

    def __post_init__(self) -> None:
        if self.n_envs < 1 or self.episode_length < 1:
            raise ValueError(
                f"need n_envs >= 1 and episode_length >= 1, got {self.n_envs},"
                f" {self.episode_length}"
            )
        if self.n_rounds < 2:
            raise ValueError(
                "the ring needs at least one committed round and the staging"
                f" round (n_rounds >= 2), got {self.n_rounds}"
            )

    @classmethod
    def create(
        cls,
        *,
        capacity: int,
        n_envs: int,
        episode_length: int,
        obs_dim: int,
        action_dim: int,
    ) -> EpisodeBuffer:
        """The ring for ``capacity`` env steps: ``R = ceil(capacity / (T n)) + 1``."""
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1 env step, got {capacity}")
        n_rounds = math.ceil(capacity / (episode_length * n_envs)) + 1
        return cls(
            n_envs=n_envs,
            episode_length=episode_length,
            n_rounds=n_rounds,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

    # -- static layout ---------------------------------------------------
    @property
    def period(self) -> int:
        """Rows (ticks) per episode, ``T + 1``."""
        return self.episode_length + 1

    @property
    def n_slots(self) -> int:
        """Episode slots in the ring, ``R * n``."""
        return self.n_rounds * self.n_envs

    @property
    def nbytes(self) -> int:
        """Bytes of one seed's storage (float32 obs, action and reward)."""
        per_row = self.obs_dim + self.action_dim + 1
        return self.n_slots * self.period * per_row * 4

    def committed_rounds(self, tick: IntLike) -> Any:
        """Rounds sampleable once tick ``tick``'s row is written.

        Round ``r`` is complete once its held tick ``r P + T`` is written, so
        ``(tick + 1) div P`` rounds are complete; the ring keeps the newest
        ``R - 1`` of them (the staging slot is excluded). Works on Python
        ints and on (unbatched) traced ticks.
        """
        complete = (tick + 1) // self.period
        if isinstance(complete, int):
            return min(complete, self.n_rounds - 1)
        return jnp.minimum(complete, self.n_rounds - 1)

    def num_episodes(self, tick: IntLike) -> Any:
        """Committed episodes after tick ``tick``: ``committed_rounds * n``."""
        return self.committed_rounds(tick) * self.n_envs

    def committed_slots(self, tick: int) -> tuple[np.ndarray, int]:
        """Ring slots of the committed episodes after (host) tick ``tick``,
        oldest first, and the number of complete episodes no longer held.

        Rounds in the order they were collected, env order within a round
        (round ``r`` is in slot ``r mod R``). A ring that never wrapped holds
        every complete episode of the run (0 evicted); the rounds evicted by
        a wrapped ring are the oldest. The order of an export
        (:func:`ajax.agents.TDMPC2.dataset.export_episodes`).
        """
        complete = (tick + 1) // self.period
        held = min(complete, self.n_rounds - 1)
        rounds = np.arange(complete - held, complete) % self.n_rounds
        slots = (rounds[:, None] * self.n_envs + np.arange(self.n_envs)).reshape(-1)
        return slots, (complete - held) * self.n_envs

    def check_layout(
        self, state: EpisodeBufferState, *, any_rounds: bool = False
    ) -> None:
        """Raise unless ``state``'s trailing axes are this ring's.

        ``[..., R n, T + 1, obs_dim]`` (and the action / reward alike); leading
        (seed) axes are allowed. With ``any_rounds`` the number of rounds may
        differ (a carried ring, :meth:`adopt`): it must then be a whole number
        ``>= 2`` of rounds of ``n_envs`` slots. Static: a trace-time check.
        """
        slots = state.obs.shape[-3] if state.obs.ndim >= 3 else -1
        expected = {
            "obs": (slots, self.period, self.obs_dim),
            "action": (slots, self.period, self.action_dim),
            "reward": (slots, self.period),
        }
        for name, shape in expected.items():
            actual = getattr(state, name).shape
            if actual[len(actual) - len(shape) :] != shape:
                raise ValueError(
                    f"replay {name} has shape {actual}, expected [..., {shape}]"
                    f" for episodes of T + 1 = {self.period} rows"
                )
        if any_rounds:
            if slots % self.n_envs or slots // self.n_envs < 2:
                raise ValueError(
                    f"a carried ring of {slots} episode slots is not >= 2 rounds"
                    f" of n_envs = {self.n_envs}"
                )
        elif slots != self.n_slots:
            raise ValueError(
                f"the replay state has {slots} episode slots, but this ring has"
                f" {self.n_slots} ({self.n_rounds} rounds x {self.n_envs} envs):"
                " a carried ring must be adopted (EpisodeBuffer.adopt) first"
            )

    def adopt(self, state: EpisodeBufferState, tick: int) -> EpisodeBufferState:
        """A carried ring of any size, laid out as this one, for a run that
        continues at absolute tick ``tick`` (a resume).

        Round ``r`` lives in slot ``r mod R`` of a ring of ``R`` rounds, and
        which rounds a ring reads at a tick depends on the tick and ``R``
        only: at ``tick`` they are the ``min(c, R - 1)`` newest complete
        rounds and the round in progress ``c = tick div P``. Each moves from
        slot ``r mod R_carried`` to slot ``r mod R``. A carried ring that
        still holds all of them (it has not evicted one, which can only
        happen when it was built for a smaller ``buffer_size``) therefore
        gives the state an uninterrupted run with this ring would have.
        Other slots are zero-filled; they are written before they are read.

        ``tick`` is a host int (the agent's resume offset); leading (seed)
        axes are kept. Returns ``state`` itself when the sizes agree.

        Raises:
            ValueError: if the trailing shapes are not this ring's, or if the
                carried ring has evicted a round this ring would still read.
        """
        self.check_layout(state, any_rounds=True)
        carried = state.obs.shape[-3] // self.n_envs
        if carried == self.n_rounds:
            return state
        current = tick // self.period
        keep = min(current, self.n_rounds - 1)
        if keep > min(current, carried - 1):
            raise ValueError(
                f"the carried ring holds the {carried - 1} newest of {current}"
                f" complete rounds, but this ring of {self.n_rounds} rounds reads"
                f" the {keep} newest: it was built with a smaller buffer_size"
            )
        source = np.full(self.n_rounds, -1)  # carried round slot per round slot
        for r in range(current - keep, current + 1):
            source[r % self.n_rounds] = r % carried
        envs = np.arange(self.n_envs)
        valid = np.repeat(source >= 0, self.n_envs)
        index = np.where(valid, (source[:, None] * self.n_envs + envs).reshape(-1), 0)

        def move(store: jax.Array, row_axes: int) -> jax.Array:
            axis = store.ndim - 1 - row_axes
            moved = jnp.take(store, jnp.asarray(index), axis=axis)
            mask = valid.reshape((-1,) + (1,) * row_axes)
            return jnp.where(mask, moved, jnp.zeros((), store.dtype))

        return EpisodeBufferState(
            obs=move(state.obs, 2),
            action=move(state.action, 2),
            reward=move(state.reward, 1),
        )

    # -- operations ------------------------------------------------------
    def init(self) -> EpisodeBufferState:
        """Zero-filled storage."""
        rows = (self.n_slots, self.period)
        return EpisodeBufferState(
            obs=jnp.zeros((*rows, self.obs_dim), jnp.float32),
            action=jnp.zeros((*rows, self.action_dim), jnp.float32),
            reward=jnp.zeros(rows, jnp.float32),
        )

    def add(
        self,
        state: EpisodeBufferState,
        obs: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        tick: IntLike,
    ) -> EpisodeBufferState:
        """Write tick ``tick``'s rows (``[n_envs, ...]``) in place.

        Row ``tick mod P`` of the episode slots of round ``tick div P``: one
        ``dynamic_update_slice`` per field (the ``n`` slots of a round are
        contiguous).
        """
        self.check_layout(state)
        tick = jnp.asarray(tick, jnp.int32)
        at = ((tick // self.period) % self.n_rounds * self.n_envs, tick % self.period)
        return EpisodeBufferState(
            obs=_write_rows(state.obs, obs, at),
            action=_write_rows(state.action, action, at),
            reward=_write_rows(state.reward, reward, at),
        )

    def episode_slot(self, tick: IntLike, episode: jax.Array) -> jax.Array:
        """Ring slot of committed episode ``episode`` in ``[0, num_episodes(tick))``.

        Episodes are numbered newest round first, env order within a round:
        episode ``j`` is env ``j mod n`` of round ``c - 1 - j div n``, with
        ``c = (tick + 1) div P`` the rounds completed so far.
        """
        completed = (jnp.asarray(tick, jnp.int32) + 1) // self.period
        round_index = completed - 1 - episode // self.n_envs
        return (round_index % self.n_rounds) * self.n_envs + episode % self.n_envs

    def sample(
        self,
        state: EpisodeBufferState,
        key: jax.Array,
        tick: IntLike,
        batch_size: int,
        horizon: int,
    ) -> TDMPC2Batch:
        """``batch_size`` sub-trajectories of ``horizon + 1`` rows.

        Each draws a committed episode uniformly with replacement and an
        offset ``s`` uniformly in ``[0, T - H]`` (b67b21c ``RandomSampler`` +
        ``RandomCropTensorDict(H + 1)``, independent per sample), then
        :func:`slice_batch`. ``tick`` is the tick whose row was written last
        (unbatched); the schedule guarantees at least one committed episode
        (:attr:`~ajax.agents.TDMPC2.train_TDMPC2.Schedule.seed_step`).
        """
        self.check_layout(state)
        if horizon > self.episode_length:
            raise ValueError(
                f"horizon {horizon} needs episodes of at least horizon + 1 rows,"
                f" got T + 1 = {self.period}"
            )
        episode_key, start_key = jax.random.split(key)
        episode = jax.random.randint(
            episode_key, (batch_size,), 0, self.num_episodes(tick)
        )
        start = jax.random.randint(
            start_key, (batch_size,), 0, self.episode_length - horizon + 1
        )
        return slice_batch(state, self.episode_slot(tick, episode), start, horizon)


def _write_rows(store: jax.Array, value: jax.Array, at: tuple) -> jax.Array:
    """``value [n, ...]`` into ``store[at[0] : at[0] + n, at[1]]``: one
    ``dynamic_update_slice`` (the ``n`` rows of a tick are contiguous)."""
    value = value.reshape((-1,) + store.shape[2:]).astype(store.dtype)[:, None]
    zeros = (jnp.zeros((), jnp.int32),) * (store.ndim - 2)
    return jax.lax.dynamic_update_slice(store, value, tuple(at) + zeros)


def slice_batch(
    state: EpisodeBufferState | EpisodicBufferState,
    slot: jax.Array,
    start: jax.Array,
    horizon: int,
) -> TDMPC2Batch:
    """The time-major :class:`~ajax.agents.TDMPC2.core.TDMPC2Batch` of the
    sub-trajectories at ring slots ``slot [B]`` and row offsets ``start [B]``.

    Obs rows ``s .. s + H``, action rows ``s .. s + H - 1``, reward rows
    ``s + 1 .. s + H`` (module docstring): ``DataPrepTransform`` of b67b21c
    (``buffer.py:26-28``) on the obs-aligned layout. On the episodic rings
    (slot = env) the rows wrap around the ring, and ``terminated`` is read
    from the rows ``s + 1 .. s + H``.
    """
    rows = start[:, None] + jnp.arange(horizon + 1)[None]  # [B, H + 1]
    episodes = slot[:, None]

    def take(store: jax.Array, columns: jax.Array) -> jax.Array:
        return jnp.swapaxes(store[episodes, columns], 0, 1)  # time-major

    terminated = None
    if isinstance(state, EpisodicBufferState):
        rows = rows % state.obs.shape[-2]
        terminated = take(state.terminal, rows[:, 1:]).astype(jnp.float32)
    return TDMPC2Batch(
        obs=take(state.obs, rows),
        action=take(state.action, rows[:, :-1]),
        reward=take(state.reward, rows[:, 1:]),
        terminated=terminated,
    )


@struct.dataclass
class EpisodicBufferState:
    """Per-env rings of rows (:class:`EpisodicBuffer`), ``[n_envs, C, ...]``.

    Attributes:
        obs, action, reward: the row collector's obs-aligned row.
        terminal: the observation is a terminal state (``is_terminal``).
        last: the row ends its episode (``is_last``).
        step: the row's index in its episode (0 on the reset row).
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    terminal: jax.Array
    last: jax.Array
    step: jax.Array


@dataclasses.dataclass(frozen=True)
class EpisodicBuffer:
    """Variable-length episode replay of the episodic mode (module docstring).

    Attributes:
        n_envs: lockstep envs, one ring each.
        capacity: ``C``, rows per env.
        obs_dim: flat observation size.
        action_dim: action size ``A``.
    """

    n_envs: int
    capacity: int
    obs_dim: int
    action_dim: int

    def __post_init__(self) -> None:
        if self.n_envs < 1 or self.capacity < 1:
            raise ValueError(
                f"need n_envs >= 1 and capacity >= 1 row, got {self.n_envs},"
                f" {self.capacity}"
            )

    @classmethod
    def create(
        cls, *, buffer_size: int, n_envs: int, obs_dim: int, action_dim: int
    ) -> EpisodicBuffer:
        """The rings for ``buffer_size`` rows: ``ceil(buffer_size / n_envs)``
        rows per env (the agent passes the run's clamped capacity,
        :func:`replay_capacity`)."""
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        return cls(n_envs, math.ceil(buffer_size / n_envs), obs_dim, action_dim)

    @property
    def nbytes(self) -> int:
        """Bytes of one seed's storage."""
        per_row = (self.obs_dim + self.action_dim + 1) * 4 + 1 + 1 + 4
        return self.n_envs * self.capacity * per_row

    def init(self) -> EpisodicBufferState:
        """Zero-filled rings (no row ends an episode)."""
        rows = (self.n_envs, self.capacity)
        return EpisodicBufferState(
            obs=jnp.zeros((*rows, self.obs_dim), jnp.float32),
            action=jnp.zeros((*rows, self.action_dim), jnp.float32),
            reward=jnp.zeros(rows, jnp.float32),
            terminal=jnp.zeros(rows, bool),
            last=jnp.zeros(rows, bool),
            step=jnp.zeros(rows, jnp.int32),
        )

    def check_layout(
        self, state: EpisodicBufferState, *, any_rows: bool = False
    ) -> None:
        """Raise unless ``state`` is this buffer's (a trace-time check); with
        ``any_rows`` its rings may hold another number of rows (carried
        rings, :meth:`adopt`)."""
        expected = (self.n_envs, self.capacity, self.obs_dim)
        got = state.obs.shape[-3:]
        rows = any_rows or got[1:2] == (self.capacity,)
        if len(got) != 3 or got[::2] != expected[::2] or not rows:
            raise ValueError(
                f"replay obs has shape {state.obs.shape}, expected [..., {expected}]"
                " (n_envs, rows per env, obs_dim): carried rings of another"
                " length must be adopted (EpisodicBuffer.adopt) first"
            )

    def adopt(self, state: EpisodicBufferState, tick: int) -> EpisodicBufferState:
        """Carried rings of any length, laid out as these, for a run that
        continues at absolute tick ``tick`` (a resume).

        Tick ``i``'s rows live at slot ``i mod C``; from ``tick`` on these
        rings read the rows of the ``min(tick, C)`` latest ticks, each moved
        from slot ``i mod C_carried``. Carried rings that still hold them all
        (only rings built for a smaller ``buffer_size`` lose one) give the
        state an uninterrupted run with these rings would have. The other
        slots are zero (no row ends an episode there) until written.

        ``tick`` is a host int (the agent's resume offset); leading (seed)
        axes are kept. Returns ``state`` itself when the lengths agree.

        Raises:
            ValueError: if the carried rings are not this buffer's (envs,
                observation size), or if they lost a row these rings read.
        """
        self.check_layout(state, any_rows=True)
        carried = state.obs.shape[-2]
        if carried == self.capacity:
            return state
        keep = min(tick, self.capacity)
        if keep > min(tick, carried):
            raise ValueError(
                f"the carried rings hold the {carried} latest of {tick} rows per"
                f" env, but these rings of {self.capacity} read the {keep} latest:"
                " they were built with a smaller buffer_size"
            )
        ticks = np.arange(tick - keep, tick)
        index = np.zeros(self.capacity, int)
        index[ticks % self.capacity] = ticks % carried
        valid = np.zeros(self.capacity, bool)
        valid[ticks % self.capacity] = True
        axis = state.reward.ndim - 1  # [..., n_envs, C, ...]: the rows

        def move(store: jax.Array) -> jax.Array:
            moved = jnp.take(store, jnp.asarray(index), axis=axis)
            mask = valid.reshape((-1,) + (1,) * (store.ndim - 1 - axis))
            return jnp.where(mask, moved, jnp.zeros((), store.dtype))

        return jax.tree.map(move, state)

    def add(
        self, state: EpisodicBufferState, row: Row, step: jax.Array, tick: IntLike
    ) -> EpisodicBufferState:
        """Write tick ``tick``'s rows (``[n_envs, ...]``) at slot ``tick mod C``;
        ``step [n_envs]`` is each row's index in its episode."""
        self.check_layout(state)
        at = (jnp.zeros((), jnp.int32), jnp.asarray(tick, jnp.int32) % self.capacity)
        return EpisodicBufferState(
            obs=_write_rows(state.obs, row.obs, at),
            action=_write_rows(state.action, row.action, at),
            reward=_write_rows(state.reward, row.reward, at),
            terminal=_write_rows(state.terminal, row.is_terminal, at),
            last=_write_rows(state.last, row.is_last, at),
            step=_write_rows(state.step, step, at),
        )

    def episodes(
        self, state: EpisodicBufferState, tick: IntLike, horizon: int
    ) -> tuple[jax.Array, jax.Array]:
        """The sampleable episodes after tick ``tick``'s rows, by last row.

        Returns ``[n_envs, C]`` masks of the rows that end an episode with at
        least ``horizon + 1`` stored rows, and the tick of each episode's
        first stored row: its first row, or the oldest row in the ring when
        the ring has overwritten its head.
        """
        now = jnp.asarray(tick, jnp.int32)
        row_tick = now - (now - jnp.arange(self.capacity)) % self.capacity
        first = jnp.maximum(row_tick - state.step, now - self.capacity + 1)
        valid = state.last & (row_tick >= 0) & (row_tick - first >= horizon)
        return valid, first

    def num_episodes(
        self, state: EpisodicBufferState, tick: IntLike, horizon: int
    ) -> jax.Array:
        """How many episodes :meth:`sample` draws from."""
        return jnp.sum(self.episodes(state, tick, horizon)[0], dtype=jnp.int32)

    def sample(
        self,
        state: EpisodicBufferState,
        key: jax.Array,
        tick: IntLike,
        batch_size: int,
        horizon: int,
    ) -> TDMPC2Batch:
        """``batch_size`` sub-trajectories of ``horizon + 1`` rows with
        ``terminated``: each an episode drawn uniformly with replacement
        among :meth:`episodes`, then an offset uniformly among its stored
        rows. Draws garbage when there is no episode: the caller checks
        :meth:`num_episodes` first."""
        self.check_layout(state)
        valid, first = self.episodes(state, tick, horizon)
        counts = jnp.cumsum(valid.reshape(-1), dtype=jnp.int32)
        episode_key, start_key = jax.random.split(key)
        draw = jax.random.randint(
            episode_key, (batch_size,), 0, jnp.maximum(counts[-1], 1)
        )
        end = jnp.searchsorted(counts, draw, side="right")  # its last row
        env, slot = end // self.capacity, end % self.capacity
        now = jnp.asarray(tick, jnp.int32)
        start = first.reshape(-1)[end]
        rows = now - (now - slot) % self.capacity - start + 1
        start = start + jax.random.randint(
            start_key, (batch_size,), 0, jnp.maximum(rows - horizon, 1)
        )
        return slice_batch(state, env, start, horizon)


__all__ = [
    "EpisodeBuffer",
    "EpisodeBufferState",
    "EpisodicBuffer",
    "EpisodicBufferState",
    "replay_capacity",
    "slice_batch",
]
