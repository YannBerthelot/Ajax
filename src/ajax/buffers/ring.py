"""Replay rings sized to the run.

A ring holds, per env, the rows its run writes, at most its capacity's: a
short run reserves no memory it never fills. Until it wraps, a ring keeps
row a at slot a whatever its length, so it holds what a ring of the full
capacity would, and a resumed run lengthens the ring of an earlier, shorter
call to the uninterrupted run's. The world models' replays (DreamerV3,
TD-MPC2) follow these rules.
"""

from __future__ import annotations


def count_ring_rows(rows_per_env: int, capacity_rows: int, min_rows: int) -> int:
    """The ring length C for a run of rows_per_env rows per env in all: the
    capacity's rows per env or the run's, whichever is fewer, never below
    min_rows, the ring the first batch needs (a run too short to reach the
    first update keeps that minimal ring)."""
    return max(min(capacity_rows, rows_per_env), min_rows)


def size_ring(held: int, seen: int, wanted: int, capacity_name: str) -> int:
    """The ring length a resumed run continues with, from a state whose ring
    holds held rows per env after seen were written.

    wanted is the ring the uninterrupted run of all the rows keeps
    (count_ring_rows). A ring at least that long is kept. A shorter one was
    sized by an earlier, shorter call and, while it has overwritten nothing
    (seen <= held), grows to that length exactly; one that has is refused,
    naming the agent's capacity_name. Host-side, before a resumed run.
    """
    if wanted <= held:
        return held
    if seen > held:
        raise ValueError(
            f"The state's replay keeps the last {held} of its {seen}"
            f" rows per env, but this agent's {capacity_name} keeps"
            f" {wanted} for the resumed run: resume with the"
            f" {capacity_name} that produced the state."
        )
    return wanted
