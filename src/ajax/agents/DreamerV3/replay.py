"""DreamerV3's stream replay: per-env rings of rows, windows, online queue, write-back.

The paper-era replay (``danijar/dreamerv3@2411f7d:embodied/replay/replay.py``,
built by ``dreamerv3/main.py:136-162`` with ``length = batch_length = 65``,
``online = True`` and a uniform selector) as a fixed-shape, on-device
structure (``docs/world_models/DESIGN.md`` section 6.3; dreamerv3_spec 5.1-5.9
and Algorithm J):

* **Storage.** One ring of ``C`` rows per env, ``[n_envs, C, ...]``, written
  in lockstep: the rows of tick ``i`` go to slot ``i mod C`` of every env's
  ring. A row is the row collector's (:class:`ajax.environments.row_collector.
  Row`: ``obs``, ``reward``, ``is_first``, ``is_last``, ``is_terminal`` and the
  raw action, zeroed at ``is_last``) plus the posterior latent of that row as
  the policy computed it (``deter`` float32, ``stoch`` as class indices, uint8
  when ``classes <= 256``; dreamerv3_spec 5.7). The reference's stream per
  worker (``replay.py:97-144``) is the same sequence of rows; its chunks are
  storage detail. Its capacity counts items over all workers; here it counts
  rows per env, overwritten in lockstep (deviation D27; never binding in the
  paper protocols).
* **Items** are the windows of ``L = batch_length + 1`` consecutive rows of
  one env (stride 1, crossing episode boundaries freely: dreamerv3_spec 5.2)
  that are fully written and not yet overwritten -- they never cross the
  write head. After ``r`` rows per env, the stored rows are ``max(0, r - C)
  .. r - 1`` and the items start at ``max(0, r - C) .. r - L``; the reference
  inserts the item ending at every added row (``replay.py:130-138``).
* **Physical indices** are ``(start + k) mod C``: windows are read with
  ``take`` and written with modular scatters, never with ``dynamic_slice`` on
  the ring (which would clamp, not wrap).
* **Online queue** (``replay.py:140-144``, ``:178-182``; dreamerv3_spec 5.4).
  Worker ``w`` pushes the item it inserts when its earlier row count is a
  multiple of ``L``: rows ``n = L, 2L, ...`` insert the items starting at
  ``1, 1 + L, ...``. With lockstep envs added in env order the queue is
  item ``q`` = env ``q mod n_envs``, start ``1 + L (q div n_envs)``, so a
  counter of popped items is the whole queue state. A training batch pops
  the pending items first (at most ``B``) and draws the rest uniformly over
  all items with replacement (``replay.py:263-272``). A pending item the ring
  has already overwritten is skipped, as the reference's ``KeyError`` retry
  skips an evicted one (``replay.py:178-191``).
* **Annotation** (``replay.py:302-317``): ``is_first[:, 0] = True`` and
  ``is_last |= next is_first`` (the last column excluded).
* **Replay context** (Algorithm J; ``29eb964:dreamerv3/agent.py:172-183``,
  the upstream fix of 2411f7d's first previous action): the carry is the
  latent stored with row 0 of the window, the trained rows are ``1..L-1``
  and their previous actions are ``action[0 : L-1]``
  (:class:`~ajax.agents.DreamerV3.world_model.ReplayContextBatch` holds the
  whole window and the context latent; the world-model loss slices it).
* **Write-back** (``replay.py:146-167``, ``29eb964:dreamerv3/agent.py:197-
  202``; dreamerv3_spec 5.9): the training step's posterior latents of rows
  ``1..L-1`` overwrite the stored ones, row by row in batch order, so where
  two sampled windows overlap the later row wins, as in the reference's
  sequential loop. The context row 0 is never written.

The ring's length ``C`` and every count derived from the tick are static or
unbatched; only the stored rows and the popped-item counter are per seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents.DreamerV3.world_model import PosteriorEntries, ReplayContextBatch
from ajax.environments.row_collector import Row

#: Largest number of classes whose indices fit the uint8 storage.
_UINT8_CLASSES = 256


def stoch_dtype(classes: int) -> jnp.dtype:
    """Storage dtype of the stochastic latent's class indices."""
    return jnp.dtype(jnp.uint8) if classes <= _UINT8_CLASSES else jnp.dtype(jnp.int32)


@struct.dataclass
class ReplayState:
    """The per-env rings and the online-queue counter (one seed).

    Attributes:
        obs: ``[n_envs, C, O]`` float32.
        action: ``[n_envs, C, A]`` float32 (continuous) or ``[n_envs, C]``
            int32 (discrete): the agent's raw action, as the collector
            emitted it.
        reward, is_first, is_last, is_terminal: ``[n_envs, C]``.
        deter: ``[n_envs, C, D]`` float32, the posterior ``h`` of each row.
        stoch: ``[n_envs, C, S]`` class indices of its posterior ``z``.
        popped: ``[]`` int32, online-queue items consumed so far.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    is_first: jax.Array
    is_last: jax.Array
    is_terminal: jax.Array
    deter: jax.Array
    stoch: jax.Array
    popped: jax.Array


class BatchIndex(NamedTuple):
    """Where the ``B`` windows of a batch come from.

    ``env [B]`` and ``start [B]`` (the absolute row index of the window's
    first row in that env's stream) as int32; ``online [B]`` marks the rows
    popped from the online queue.
    """

    env: jax.Array
    start: jax.Array
    online: jax.Array


class Window(NamedTuple):
    """``B`` annotated windows of ``L`` rows and the latents of their row 0.

    The row fields are ``[B, L, ...]`` (``action`` raw, as stored);
    ``context_deter [B, D]`` and ``context_stoch [B, S]`` (int32) are the
    latents stored with row 0, the replay context. The other rows' stored
    latents are not read: the training step recomputes them
    (``29eb964:dreamerv3/agent.py:175-176`` keeps ``[:, :K]`` only).
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    is_first: jax.Array
    is_last: jax.Array
    is_terminal: jax.Array
    context_deter: jax.Array
    context_stoch: jax.Array


@dataclass(frozen=True)
class StreamReplay:
    """The replay's static geometry (hashable).

    Attributes:
        n_envs: number of env streams.
        capacity: ``C``, rows kept per env (resolved by the agent:
            ``min(ceil(replay_capacity / n_envs), rows per env of the run)``,
            ``DESIGN.md`` section 5.6).
        batch_size: ``B``, windows per batch.
        batch_length: ``T``, trained rows per window; a window has ``L = T +
            1`` rows, the first being context (2411f7d ``batch_length: 65``
            with ``replay_context: 1``).
    """

    n_envs: int
    capacity: int
    batch_size: int
    batch_length: int

    def __post_init__(self) -> None:
        for name in ("n_envs", "capacity", "batch_size", "batch_length"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.capacity < self.length:
            raise ValueError(
                f"The ring holds {self.capacity} rows per env, fewer than one"
                f" window of length {self.length} (batch_length + 1)."
            )

    @property
    def length(self) -> int:
        """``L = batch_length + 1``: rows per window, context included."""
        return self.batch_length + 1

    # ------------------------------------------------------------- storage

    def init(
        self,
        obs_dim: int,
        action_dim: int,
        discrete: bool,
        deter: int,
        stoch: int,
        classes: int,
    ) -> ReplayState:
        """Empty rings: zeros everywhere, nothing popped.

        ``action_dim`` is the action's dimension; a discrete action is
        stored as its int32 index.
        """
        n, c = self.n_envs, self.capacity
        action = (
            jnp.zeros((n, c), jnp.int32)
            if discrete
            else jnp.zeros((n, c, action_dim), jnp.float32)
        )
        return ReplayState(
            obs=jnp.zeros((n, c, obs_dim), jnp.float32),
            action=action,
            reward=jnp.zeros((n, c), jnp.float32),
            is_first=jnp.zeros((n, c), bool),
            is_last=jnp.zeros((n, c), bool),
            is_terminal=jnp.zeros((n, c), bool),
            deter=jnp.zeros((n, c, deter), jnp.float32),
            stoch=jnp.zeros((n, c, stoch), stoch_dtype(classes)),
            popped=jnp.zeros((), jnp.int32),
        )

    def add(self, state: ReplayState, row: Row, tick: jax.Array) -> ReplayState:
        """Write tick ``tick``'s row of every env at slot ``tick mod C``.

        ``row.extras`` is the posterior latent the policy filtered the row's
        observation into, ``(deter [n_envs, D], stoch [n_envs, S])``
        (class indices). ``tick`` is the absolute, unbatched tick index: the
        row is the ``tick``-th of every env's stream (0-indexed).
        """
        slot = jnp.asarray(tick, jnp.int32) % self.capacity
        deter, stoch = row.extras

        def write(ring: jax.Array, value: jax.Array) -> jax.Array:
            return ring.at[:, slot].set(jnp.asarray(value, ring.dtype))

        return state.replace(
            obs=write(state.obs, row.obs.reshape(self.n_envs, -1)),
            action=write(state.action, row.action),
            reward=write(state.reward, row.reward),
            is_first=write(state.is_first, row.is_first),
            is_last=write(state.is_last, row.is_last),
            is_terminal=write(state.is_terminal, row.is_terminal),
            deter=write(state.deter, deter),
            stoch=write(state.stoch, stoch),
        )

    # -------------------------------------------------------------- counts

    def oldest(self, rows: jax.Array) -> jax.Array:
        """First row index still stored after ``rows`` rows per env."""
        return jnp.maximum(rows - self.capacity, 0)

    def items_per_env(self, rows: jax.Array) -> jax.Array:
        """Windows per env after ``rows`` rows: ``max(0, min(rows, C) - L +
        1)`` (``len(replay) = n_envs`` times this)."""
        return jnp.maximum(jnp.minimum(rows, self.capacity) - self.length + 1, 0)

    def online_pushed(self, rows: jax.Array) -> jax.Array:
        """Online-queue items pushed after ``rows`` rows per env.

        Each env pushes at its rows ``n = L, 2L, ...`` (0-indexed), so after
        ``rows`` rows it has pushed ``floor((rows - 1) / L)`` items
        (``replay.py:140-144``: ``lengths[worker] % length == 0``).
        """
        per_env = jnp.maximum(rows - 1, 0) // self.length
        return self.n_envs * per_env

    def online_item(self, q: jax.Array) -> tuple[jax.Array, jax.Array]:
        """``(env, start)`` of online-queue item ``q``."""
        return q % self.n_envs, 1 + self.length * (q // self.n_envs)

    def first_live_online_item(self, rows: jax.Array) -> jax.Array:
        """Smallest queue position whose window is still stored.

        Item ``q`` starts at ``1 + L (q div n_envs)``; it is live when that
        start is ``>= oldest(rows)``.
        """
        oldest = self.oldest(rows)
        per_env = jnp.maximum(oldest + self.length - 2, 0) // self.length
        return self.n_envs * per_env

    # ------------------------------------------------------------ sampling

    def sample(
        self, state: ReplayState, rows: jax.Array, key: jax.Array
    ) -> tuple[ReplayState, BatchIndex]:
        """Choose the ``B`` windows of one batch; pops the online queue.

        ``rows`` is the (unbatched) number of rows per env written so far;
        the replay must hold at least one item. Rows ``b < n_online`` take
        the pending online items in queue order, ``n_online = min(B,
        pending)``; the others are uniform over the ``n_envs *
        items_per_env(rows)`` items, with replacement (2411f7d
        ``Replay._sample``, ``replay.py:169-191``, called once per batch row
        by ``dataset``, ``:263-272``; ``selectors.Uniform``).
        """
        b = jnp.arange(self.batch_size, dtype=jnp.int32)
        rows = jnp.asarray(rows, jnp.int32)
        head = jnp.maximum(state.popped, self.first_live_online_item(rows))
        pending = jnp.maximum(self.online_pushed(rows) - head, 0)
        n_online = jnp.minimum(pending, self.batch_size)
        online = b < n_online
        online_env, online_start = self.online_item(head + b)

        items = jnp.maximum(self.items_per_env(rows), 1)
        u = jax.random.randint(key, (self.batch_size,), 0, self.n_envs * items)
        uniform_env = u // items
        uniform_start = self.oldest(rows) + u % items

        index = BatchIndex(
            env=jnp.where(online, online_env, uniform_env).astype(jnp.int32),
            start=jnp.where(online, online_start, uniform_start).astype(jnp.int32),
            online=online,
        )
        return state.replace(popped=head + n_online), index

    def physical(self, start: jax.Array, offsets: jax.Array) -> jax.Array:
        """Ring slots ``(start + offsets) mod C``, broadcast ``[B, len(offsets)]``."""
        return (start[:, None] + offsets[None, :]) % self.capacity

    def gather(self, state: ReplayState, index: BatchIndex) -> Window:
        """The annotated windows of ``index`` (``replay.py:280-317``).

        Rows are read with ``take`` on the ``[n_envs * C, ...]`` flattened
        rings at ``env * C + (start + k) mod C``; the stored latents only at
        ``k = 0``. Then ``is_first[:, 0] = True`` and ``is_last |=`` the next
        row's ``is_first`` (last column excluded).
        """
        c = self.capacity
        slots = self.physical(index.start, jnp.arange(self.length, dtype=jnp.int32))
        flat = index.env[:, None] * c + slots

        def take(ring: jax.Array, where: jax.Array) -> jax.Array:
            return jnp.take(ring.reshape(-1, *ring.shape[2:]), where, axis=0)

        is_first = take(state.is_first, flat).at[:, 0].set(True)
        next_is_first = jnp.concatenate(
            [is_first[:, 1:], jnp.zeros_like(is_first[:, :1])], 1
        )
        context = index.env * c + index.start % c
        return Window(
            obs=take(state.obs, flat),
            action=take(state.action, flat),
            reward=take(state.reward, flat),
            is_first=is_first,
            is_last=take(state.is_last, flat) | next_is_first,
            is_terminal=take(state.is_terminal, flat),
            context_deter=take(state.deter, context),
            context_stoch=take(state.stoch, context).astype(jnp.int32),
        )

    # ---------------------------------------------------------- write-back

    def write_back(
        self, state: ReplayState, index: BatchIndex, entries: PosteriorEntries
    ) -> ReplayState:
        """Store a training step's posterior latents of rows ``1..L-1``.

        ``entries.deter [B, T, D]`` and ``entries.stoch [B, T, S]`` (class
        indices) were computed on the windows of ``index``; batch row ``b``
        writes its ``T`` latents at slots ``(start_b + 1 + k) mod C`` of env
        ``env_b``, never its context row. The rows are written one after the
        other in batch order (a ``fori_loop``), so where windows overlap the
        later batch row wins, deterministically: the reference's sequential
        ``for i, stepid in enumerate(stepid)`` (``replay.py:158-167``).
        """
        offsets = jnp.arange(1, self.length, dtype=jnp.int32)
        slots = self.physical(index.start, offsets)
        stoch = entries.stoch.astype(state.stoch.dtype)
        deter = entries.deter.astype(state.deter.dtype)

        def write_row(b, rings):
            ring_deter, ring_stoch = rings
            env, where = index.env[b], slots[b]
            return (
                ring_deter.at[env, where].set(deter[b]),
                ring_stoch.at[env, where].set(stoch[b]),
            )

        ring_deter, ring_stoch = jax.lax.fori_loop(
            0, self.batch_size, write_row, (state.deter, state.stoch)
        )
        return state.replace(deter=ring_deter, stoch=ring_stoch)


def context_batch(window: Window, num_actions: Optional[int]) -> ReplayContextBatch:
    """The learner's batch (Algorithm J) from annotated windows.

    Discrete actions (``num_actions`` given) are one-hot encoded
    (``jaxutils.onehot_dict``, ``29eb964:dreamerv3/agent.py:233``);
    continuous ones pass through raw (the dynamics bound them).
    """
    action = (
        jax.nn.one_hot(window.action, num_actions, dtype=jnp.float32)
        if num_actions is not None
        else jnp.asarray(window.action, jnp.float32)
    )
    return ReplayContextBatch(
        obs=window.obs,
        action=action,
        reward=window.reward,
        is_first=window.is_first,
        is_last=window.is_last,
        is_terminal=window.is_terminal,
        context_deter=window.context_deter,
        context_stoch=window.context_stoch,
    )


#: The per-row fields of :class:`ReplayState`, ``[..., n_envs, C, ...]``.
RING_FIELDS = (
    "obs",
    "action",
    "reward",
    "is_first",
    "is_last",
    "is_terminal",
    "deter",
    "stoch",
)


def grow_rings(state: ReplayState, rows: int, capacity: int) -> ReplayState:
    """``state``'s rings lengthened to ``capacity`` rows per env.

    Valid while no row has been overwritten, ``rows <= C`` (``rows`` written
    per env so far, ``C`` the current ring length): row ``a`` then sits at
    slot ``a`` of either ring, so the longer ring is the shorter one followed
    by empty slots -- exactly the ring a run with ``capacity`` rows per env
    holds after the same rows. The online-queue counter is unchanged.
    Leading batch axes (the seed ``vmap``) are kept: the ring axis follows
    the env axis, after ``popped``'s axes. Host-side, before a resumed run.
    """
    axis = jnp.ndim(state.popped) + 1
    current = int(jnp.shape(state.obs)[axis])
    if capacity < current:
        raise ValueError(f"Cannot shrink a ring of {current} rows to {capacity}.")
    if rows > current:
        raise ValueError(
            f"A ring of {current} rows per env that has seen {rows} rows has"
            f" overwritten its rows 0..{rows - current - 1}; it cannot grow to"
            f" {capacity} rows."
        )

    def pad(ring: jax.Array) -> jax.Array:
        widths = [(0, 0)] * jnp.ndim(ring)
        widths[axis] = (0, capacity - current)
        return jnp.pad(ring, widths)

    return state.replace(**{name: pad(getattr(state, name)) for name in RING_FIELDS})


def bytes_per_row(
    obs_dim: int, action_dim: int, discrete: bool, deter: int, stoch: int, classes: int
) -> int:
    """Bytes one stored row takes (all envs' rings: times ``n_envs * C``)."""
    action = 4 if discrete else 4 * action_dim
    flags = 3  # is_first, is_last, is_terminal (bool)
    stoch_bytes = stoch * stoch_dtype(classes).itemsize
    return 4 * obs_dim + action + 4 + flags + 4 * deter + stoch_bytes


__all__ = [
    "RING_FIELDS",
    "BatchIndex",
    "ReplayState",
    "StreamReplay",
    "Window",
    "bytes_per_row",
    "context_batch",
    "grow_rings",
    "stoch_dtype",
]
