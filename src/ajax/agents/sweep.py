"""A sweep: hyperparameter configurations trained with every seed, all in
one compiled program (``ActorCritic.train``'s ``sweep``).

The seeds already train as one program, vmapped over their keys. A sweep
puts its configurations on the same axis: each member, one configuration
with one seed, builds its agent with the configuration's values while the
program is traced (``ActorCritic.member``), so they are numbers the
program reads, one per member, instead of constants compiled into it.
Members run configuration-major: member ``k * n_seeds + s`` is
configuration ``k`` with seed ``s``, and draws the keys a run of that
configuration with seed ``s`` draws.

Only an agent's sweepable fields (``ActorCritic.sweepable``) vary: numbers
its training reads as numbers. A setting that sets a shape, a loop's length
or which code is built needs a program of its own: one train call per value.
"""

from __future__ import annotations

import itertools
import numbers
from collections.abc import Mapping, Sequence
from typing import Optional, Union

import jax
import jax.numpy as jnp

#: A sweep: a dict of value lists, whose grid is trained, or a list of
#: configurations.
Sweep = Union[Mapping[str, Sequence[float]], Sequence[Mapping[str, float]]]


def configurations(sweep: Optional[Sweep]) -> list[dict[str, float]]:
    """The sweep's configurations in order: a dict's grid (its last field
    varying fastest, as ``itertools.product``) or the list given; none
    without a sweep."""
    if sweep is None:
        return []
    if isinstance(sweep, Mapping):
        names = list(sweep)
        grid = itertools.product(*(sweep[name] for name in names))
        found = [dict(zip(names, values, strict=True)) for values in grid]
    else:
        found = [dict(configuration) for configuration in sweep]
    if not found or not found[0]:
        raise ValueError("A sweep needs a configuration setting a field.")
    for configuration in found:
        if set(configuration) != set(found[0]):
            raise ValueError(
                "Every configuration of a sweep sets the same fields: got"
                f" {sorted(found[0])} and {sorted(configuration)}."
            )
        for name, value in configuration.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"A sweep's values are numbers: {name}={value!r}.")
    return found


def check(agent_name: str, sweepable: frozenset[str], names: Sequence[str]) -> None:
    """Refuse the fields outside the agent's sweepable ones, naming them."""
    refused = sorted(set(names).difference(sweepable))
    if refused:
        allowed = ", ".join(sorted(sweepable)) or "none"
        raise ValueError(
            f"{agent_name} cannot sweep {', '.join(refused)} in one program: a"
            " sweep varies the numbers its training reads as numbers (here"
            f" {allowed}). A setting that changes shapes or the program's"
            " structure takes one train call per value."
        )


def member_row(configuration: Mapping[str, float], seed: int) -> dict[str, float]:
    """A member's seed and values, as ``ActorCritic.members`` lists them."""
    row: dict[str, float] = {"seed": int(seed)}
    row.update(configuration)
    return row


def member_values(
    configurations_by_member: Sequence[Mapping[str, float]],
) -> dict[str, jax.Array]:
    """Each swept field's values, one per member: float32, the precision of
    the arrays they multiply, as a separate run's Python number takes it."""
    names = list(configurations_by_member[0]) if configurations_by_member else []
    return {
        name: jnp.asarray(
            [configuration[name] for configuration in configurations_by_member],
            dtype=jnp.float32,
        )
        for name in names
    }
