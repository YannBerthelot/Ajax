"""Checkpoint save/restore for Ajax agents.

Serializes the pytree *data* of a ``BaseAgentState`` (arrays, counters,
and other leaves) while discarding non-data fields such as optax
``GradientTransformation`` closures and flax ``apply_fn`` callables.
Those closures are rebuilt at load time by re-initializing a skeleton
state from the same agent config; the loaded data is then grafted onto
the skeleton via ``flax.serialization.from_state_dict``.

Minimal API:
    save_checkpoint(state, path)
    state = restore_into(skeleton_state, path)
    checkpoint_exists(path)

Typical resume flow (see ``run_final_eval.py``):
    skeleton = agent.train(seed=seeds, n_timesteps=0)       # init only
    if checkpoint_exists(path):
        skeleton = restore_into(skeleton, path)
    final = agent.train(..., initial_state=skeleton, ...)
    save_checkpoint(final, path)
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import jax
import numpy as np
from flax import serialization


def _to_host_numpy(state: Any) -> Any:
    return jax.tree.map(
        lambda x: np.asarray(x) if hasattr(x, "shape") else x,
        state,
    )


def save_checkpoint(state: Any, path: str) -> None:
    """Save the pytree data of ``state`` to ``path``. Atomic write."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    host_state = _to_host_numpy(state)
    state_dict = serialization.to_state_dict(host_state)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def restore_into(skeleton: Any, path: str) -> Any:
    """Load ``path`` and graft its data onto ``skeleton``.

    ``skeleton`` is a freshly-initialized agent state with the same
    structure as the saved one. The optimizer transformations and apply
    functions come from ``skeleton``; only the pytree data (params,
    optimizer mu/nu/count, buffer arrays, rngs, counters) is replaced.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint file at {path}")
    with open(path, "rb") as f:
        state_dict = pickle.load(f)
    restored = serialization.from_state_dict(skeleton, state_dict)
    # Push data back onto device
    return jax.tree.map(
        lambda x: jax.device_put(x) if isinstance(x, np.ndarray) else x,
        restored,
    )


def checkpoint_exists(path: str) -> bool:
    return os.path.isfile(os.path.abspath(path))
