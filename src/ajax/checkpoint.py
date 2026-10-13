"""Checkpoint save/restore for Ajax agents.

Serializes the pytree *data* of a ``BaseAgentState`` (arrays, counters,
and other leaves) while discarding non-data fields such as optax
``GradientTransformation`` closures and flax ``apply_fn`` callables.
Those closures are rebuilt at load time by re-initializing a skeleton
state from the same agent config; the loaded data is then grafted onto
the skeleton via ``flax.serialization.from_state_dict``.

Typical resume flow (``resume`` in tests/probing/runs.py runs it):
    state = None                                            # a fresh run
    if checkpoint_exists(path):
        state = restore_into(skeleton_of(agent, seeds), path)
    final, _ = agent.train(..., initial_state=state, ...)
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


def skeleton_of(agent: Any, seed: Any, **kwargs: Any) -> Any:
    """The state :func:`restore_into` grafts a checkpoint of ``agent`` onto:
    a fresh run's (``agent.train(seed=seed, n_timesteps=0, **kwargs)``),
    traced with ``jax.eval_shape``. The checkpoint replaces every leaf, so
    only the structure is needed: no program is compiled or run."""
    state, _ = jax.eval_shape(lambda: agent.train(seed=seed, n_timesteps=0, **kwargs))
    return state


def restore_into(skeleton: Any, path: str) -> Any:
    """Load ``path`` and graft its data onto ``skeleton``.

    ``skeleton`` is a freshly-initialized agent state, or its shapes
    (:func:`skeleton_of`), with the same structure as the saved one. The
    optimizer transformations and apply functions come from ``skeleton``;
    only the pytree data (params, optimizer mu/nu/count, buffer arrays,
    rngs, counters) is replaced.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint file at {path}")
    with open(path, "rb") as f:
        state_dict = pickle.load(f)
    state_dict = _with_replay_next_obs(skeleton, state_dict)
    state_dict = _with_new_empty_fields(
        serialization.to_state_dict(skeleton), state_dict
    )
    state_dict = _with_raw_obs_info(skeleton, state_dict)
    restored = serialization.from_state_dict(skeleton, state_dict)
    return jax.tree.map(
        lambda x: jax.device_put(x) if isinstance(x, np.ndarray) else x,
        restored,
    )


#: Next-observation fields a replay buffer gained, by the field each is
#: rebuilt from in a checkpoint saved before it.
_REPLAY_SUCCESSORS = {"next_obs": "obs", "raw_next_obs": "raw_obs"}


def _with_replay_next_obs(skeleton: Any, state_dict: Any) -> Any:
    """Give a replay buffer saved before it stored ``next_obs`` (or
    ``raw_next_obs``) that field, rebuilt as each row's successor: the
    observation its replay read then (the reset one, not the final one, at
    a time limit).

    Deliberate expand-then-contract shim (CLAUDE.md, refactoring rule 10),
    so a run resumes across the change; delete it once no checkpoint from
    before it remains. The buffer state is pickled whole (flashbax's state
    is no flax-serialisable node), hence the attribute access.
    """
    buffer = getattr(getattr(skeleton, "collector_state", None), "buffer_state", None)
    wanted = [k for k in _REPLAY_SUCCESSORS if k in getattr(buffer, "experience", {})]
    if not wanted:  # no replay buffer (no collector, or an on-policy one)
        return state_dict
    saved = state_dict["collector_state"]["buffer_state"]
    missing = [k for k in wanted if k not in saved.experience]
    if not missing:
        return state_dict
    # Rows run along the second-to-last axis, the ring wrapping round.
    rebuilt = {
        k: np.roll(saved.experience[_REPLAY_SUCCESSORS[k]], -1, axis=-2)
        for k in missing
    }
    saved = saved.replace(experience={**saved.experience, **rebuilt})
    collector = {**state_dict["collector_state"], "buffer_state": saved}
    return {**state_dict, "collector_state": collector}


def _with_new_empty_fields(skeleton: Any, saved: Any) -> Any:
    """Give a checkpoint saved before a field was added the fields its
    skeleton holds empty (None), at any depth: ``Transition.raw_next_obs``
    in every rollout the collector and the last-rollout placeholder hold.
    An empty field has no data to restore; a field with data stays
    required, so a mismatched checkpoint still fails.

    Expand-then-contract shim (CLAUDE.md, refactoring rule 10), as
    ``_with_replay_next_obs``.
    """
    if not isinstance(skeleton, dict) or not isinstance(saved, dict):
        return saved
    out = dict(saved)
    for key, value in skeleton.items():
        if key in out:
            out[key] = _with_new_empty_fields(value, out[key])
        elif value is None:
            out[key] = None
    return out


def _with_raw_obs_info(skeleton: Any, state_dict: Any) -> Any:
    """Give a brax env state saved before the observation normaliser kept
    its raw observation in the info (``RAW_OBS_KEY``) that entry: the saved
    observation un-normalised by the statistics saved with it.

    Expand-then-contract shim (CLAUDE.md, refactoring rule 10), as
    ``_with_replay_next_obs``.
    """
    from ajax.utils import online_denormalize
    from ajax.wrappers import RAW_OBS_KEY

    env_state = getattr(getattr(skeleton, "collector_state", None), "_env_state", None)
    if RAW_OBS_KEY not in getattr(env_state, "info", {}):
        return state_dict
    saved = state_dict["collector_state"]["_env_state"]
    if RAW_OBS_KEY in saved["info"]:
        return state_dict
    stats = saved["info"]["normalization_info"]["obs"]
    raw = online_denormalize(
        saved["obs"], stats["count"], stats["mean"], stats["mean_2"]
    )
    info = {**saved["info"], RAW_OBS_KEY: np.asarray(raw, saved["obs"].dtype)}
    collector = {**state_dict["collector_state"], "_env_state": {**saved, "info": info}}
    return {**state_dict, "collector_state": collector}


def checkpoint_exists(path: str) -> bool:
    return os.path.isfile(os.path.abspath(path))
