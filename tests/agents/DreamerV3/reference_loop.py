"""Python ports of the paper-era DreamerV3 driver loop, replay and ``Ratio``.

Literal transcriptions of ``danijar/dreamerv3@2411f7d`` (MIT licence),
restricted to the control flow the Ajax agent reproduces -- which rows exist,
when updates run, which items the online queue hands out:

* :class:`Ratio` -- ``embodied/core/when.py:26-42``, verbatim;
* :class:`ReferenceReplay` -- ``embodied/replay/replay.py``: ``add``
  (``:97-144``: the per-worker stream, one item inserted per added row once
  the stream holds ``length`` rows, the online-queue push at
  ``lengths[worker] % length == 0``), ``__len__`` (``:72-73``) and the
  online part of ``_sample`` (``:178-187``). Chunks, step ids, capacity
  eviction and the uniform selector's RNG are left out: rows are named by
  ``(worker, index)``;
* :func:`reference_train_loop` -- ``embodied/run/train.py:26-28``, ``:67-91``
  with ``embodied/core/driver.py:55-81``: per vector step, the callbacks of
  each env's transition in env order (``step += 1``, ``replay.add``,
  ``train_step``), ``train_step`` returning while ``len(replay) <
  batch_size`` and otherwise running ``Ratio(step)`` updates, each sampling
  ``batch_size`` rows. ``updates_after_tick=True`` moves every tick's
  updates after its adds (Ajax's deviation D3) without changing their
  number (the gate and ``Ratio`` are still evaluated per transition; only
  the sampling moves).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional


class Ratio:
    """``2411f7d:embodied/core/when.py:26-42``."""

    def __init__(self, ratio):
        assert ratio >= 0, ratio
        self._ratio = ratio
        self._prev = None

    def __call__(self, step):
        step = int(step)
        if self._ratio == 0:
            return 0
        if self._prev is None:
            self._prev = step
            return 1
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


class ReferenceReplay:
    """The item and online-queue bookkeeping of 2411f7d ``Replay``."""

    def __init__(self, length: int, online: bool = True):
        self.length = length
        self.online = online
        self.items: dict = {}
        self.itemid = 0
        self.streams: dict = defaultdict(deque)
        self.counts: dict = defaultdict(int)  # the chunk index, per worker
        if online:
            self.lengths: dict = defaultdict(int)
            self.queue: deque = deque()

    def __len__(self) -> int:
        return len(self.items)

    def add(self, worker: int) -> None:
        index = self.counts[worker]
        self.counts[worker] += 1
        stream = self.streams[worker]
        stream.append((worker, index))
        if len(stream) >= self.length:
            key = stream.popleft()
            self.items[self.itemid] = key
            self.itemid += 1
            if self.online and self.lengths[worker] % self.length == 0:
                self.queue.append(key)
        if self.online:
            self.lengths[worker] += 1

    def sample_online(self) -> Optional[tuple]:
        """The online part of ``_sample``: the oldest queued item, or None
        (the reference then draws a uniform item)."""
        if self.online and self.queue:
            return self.queue.popleft()
        return None


def reference_train_loop(
    n_envs: int,
    batch_size: int,
    batch_length: int,
    train_ratio: float,
    ticks: int,
    updates_after_tick: bool = False,
) -> tuple[list[int], list[list[Optional[tuple]]]]:
    """Per-tick update counts and, per update, the batch's online items.

    ``batch_length`` counts the trained rows; the reference's
    ``batch_length`` config is that plus the replay context (65), and its
    ``batch_steps = batch_size * (65 - 1)``.
    """
    length = batch_length + 1
    replay = ReferenceReplay(length)
    should_train = Ratio(train_ratio / (batch_size * (length - 1)))
    step = 0
    counts: list[int] = []
    batches: list[list[Optional[tuple]]] = []

    def sample_batches(repeats: int) -> None:
        for _ in range(repeats):
            batches.append([replay.sample_online() for _ in range(batch_size)])

    for _ in range(ticks):
        count = 0
        for worker in range(n_envs):
            step += 1
            replay.add(worker)
            if len(replay) < batch_size:
                continue
            repeats = should_train(step)
            count += repeats
            if not updates_after_tick:
                sample_batches(repeats)
        if updates_after_tick:
            sample_batches(count)
        counts.append(count)
    return counts, batches
