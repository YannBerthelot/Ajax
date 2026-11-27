"""Training a run's members (its seeds, or a sweep's configurations over the
seeds) on several devices: each device trains a slice of the members, as one
device would train them all (``ActorCritic.train(devices=...)``)."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


def checked_devices(
    devices: Optional[Sequence[Any]], in_python: bool, logging_config: Any
) -> Optional[list]:
    """``train``'s ``devices`` as a list (``None`` stays ``None``), refusing
    an empty one, an environment that steps in Python and an
    ``evaluation_device``."""
    if devices is None:
        return None
    devices = list(devices)
    if not devices:
        raise ValueError("devices is empty: pass None, or at least one")
    if in_python:
        raise ValueError(
            "A Gymnasium environment steps in this process, one seed after"
            " another: it takes no devices."
        )
    if getattr(logging_config, "evaluation_device", None) is not None:
        raise ValueError(
            "Members split over devices evaluate where they train: devices"
            " takes no evaluation_device."
        )
    return devices


def member_slices(n_members: int, n_devices: int) -> list[slice]:
    """Consecutive slices of ``n_members`` rows, one per device, whose sizes
    differ by one at most, the larger first; fewer members than devices use
    as many devices as there are members."""
    if n_members < 1 or n_devices < 1:
        raise ValueError(
            f"{n_members} members cannot be split over {n_devices} devices:"
            " both must be at least 1"
        )
    used = min(n_members, n_devices)
    size, larger = divmod(n_members, used)
    slices, start = [], 0
    for position in range(used):
        stop = start + size + (1 if position < larger else 0)
        slices.append(slice(start, stop))
        start = stop
    return slices


def train_on_devices(
    train: Callable, batched: tuple, devices: Optional[Sequence[Any]]
) -> Any:
    """``train(*batched)``: on the default device without ``devices``, else with the rows of ``batched`` (pytrees whose leaves
    have the members on their first axis) split over ``devices``
    (:func:`member_slices`), the members' results back in their order on
    ``devices[0]``.

    Each slice is put on its device, where JAX then runs ``train``, in a
    thread of its own: a training program's host-side steps (its logging
    callbacks, its evaluations between chunks) would otherwise make the
    devices take turns.
    """
    if devices is None:
        return train(*batched)
    n_members = int(jax.tree.leaves(batched)[0].shape[0])
    slices = member_slices(n_members, len(devices))

    def train_slice(rows: slice, device: Any) -> Any:
        inputs = jax.tree.map(lambda leaf: leaf[rows], batched)
        return jax.block_until_ready(train(*jax.device_put(inputs, device)))

    with ThreadPoolExecutor(max_workers=len(slices)) as pool:
        futures = [
            pool.submit(train_slice, rows, device)
            for rows, device in zip(slices, devices, strict=False)
        ]
        results = [future.result() for future in futures]
    # Leaf by leaf, into the first result's structure: each call builds its
    # own networks and optimisers, which a state carries as static fields
    # that do not compare equal between calls.
    gathered = [
        jax.tree.leaves(jax.device_put(result, devices[0])) for result in results
    ]
    members = [jnp.concatenate(leaves) for leaves in zip(*gathered, strict=True)]
    return jax.tree.unflatten(jax.tree.structure(results[0]), members)
