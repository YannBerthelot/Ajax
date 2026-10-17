"""Checkpoint save/restore for Ajax agents.

Serializes the pytree *data* of a ``BaseAgentState`` (arrays, counters,
and other leaves) while discarding non-data fields such as optax
``GradientTransformation`` closures and flax ``apply_fn`` callables.
Those closures are rebuilt at load time by re-initializing a skeleton
state from the same agent config; the loaded data is then grafted onto
the skeleton via ``flax.serialization.from_state_dict``.

Typical resume flow (``resume`` in tests/probing/runs.py runs it):
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
    skeleton_dict = serialization.to_state_dict(skeleton)
    _check_memory_final_norm(skeleton_dict, state_dict)
    _check_log_std_head(skeleton_dict, state_dict)
    state_dict = _with_replay_next_obs(skeleton, state_dict)
    state_dict = _with_new_empty_fields(skeleton_dict, state_dict)
    state_dict = _with_raw_obs_info(skeleton, state_dict)
    restored = serialization.from_state_dict(skeleton, state_dict)
    _check_leaf_shapes(skeleton, restored)
    return jax.tree.map(
        lambda x: jax.device_put(x) if isinstance(x, np.ndarray) else x,
        restored,
    )


def _check_leaf_shapes(skeleton: Any, restored: Any) -> None:
    """Refuse a checkpoint whose arrays do not fit the skeleton's, saved by
    an agent of another size (networks, ``n_envs``, buffer, seeds): flax
    grafts any array, and the run would fail later, far from the cause, or
    broadcast silently. Leaves only one side holds (replay fields an older
    checkpoint kept, which the buffer then carries) are not compared."""

    def shapes(tree: Any) -> dict[str, tuple]:
        flat = jax.tree_util.tree_flatten_with_path(tree)[0]
        return {jax.tree_util.keystr(path): np.shape(leaf) for path, leaf in flat}

    want, got = shapes(skeleton), shapes(restored)
    bad = [
        f"{path} is {got[path]} in the checkpoint, {shape} in this agent"
        for path, shape in want.items()
        if path in got and got[path] != shape
    ]
    if bad:
        raise ValueError(
            "The checkpoint does not fit this agent's state: "
            + "; ".join(bad[:5])
            + (f"; and {len(bad) - 5} more" if len(bad) > 5 else "")
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


def _check_memory_final_norm(skeleton: Any, saved: Any) -> None:
    """Refuse, saying how to restore it, a checkpoint whose transformer or
    mamba memory core lacks the final LayerNorm (``ln_f``) of the network
    restoring it, or has one that network lacks: flax reports a missing key
    without its cause and drops an extra one silently, and the norm with it.
    A checkpoint saved before the cores ended with ``ln_f`` restores into
    the network it was trained with (``final_norm=False``).

    Expand-then-contract (CLAUDE.md, refactoring rule 10): delete this
    check with ``MemoryConfig.final_norm`` once no such checkpoint remains.
    """
    if not isinstance(skeleton, dict) or not isinstance(saved, dict):
        return
    if "ln_f" in skeleton and "ln_f" not in saved:
        raise ValueError(
            "This checkpoint was saved before the transformer and mamba memory"
            " cores ended with a final LayerNorm (ln_f). Restore it into an"
            " agent built with memory=MemoryConfig(..., final_norm=False), its"
            " other settings unchanged. An APG Controller's parameters also map"
            " exactly onto the new network, its LayerNorm becoming the core's"
            " ln_f: ajax.agents.APG.networks.controller_params_with_final_norm."
            " It maps the parameters only, so a resumed training state must"
            " restart its optimizer."
        )
    if "ln_f" in saved and "ln_f" not in skeleton:
        raise ValueError(
            "This checkpoint's transformer or mamba memory core ends with a final"
            " LayerNorm (ln_f), which the agent restoring it, built with"
            " memory=MemoryConfig(..., final_norm=False), lacks. Restore it into"
            " an agent built with final_norm=True, the default."
        )
    for key, value in skeleton.items():
        if key in saved:
            _check_memory_final_norm(value, saved[key])


def _check_log_std_head(skeleton: Any, saved: Any) -> None:
    """Refuse, saying how to restore it, a checkpoint whose actor has the
    other log-std head than the network restoring it: a Dense layer over the
    features (``log_std``) or one parameter per action dimension
    (``log_std_param``). flax reports the missing head without its cause and
    drops the extra one silently. PPO's checkpoints saved before it learned
    the parameter, as its reference code does, hold the Dense head.

    Expand-then-contract (CLAUDE.md, refactoring rule 10): delete this
    check once no such checkpoint remains.
    """
    if not isinstance(skeleton, dict) or not isinstance(saved, dict):
        return
    if "log_std_param" in skeleton and "log_std_param" not in saved:
        if "log_std" in saved:
            raise ValueError(
                "This checkpoint's actor computes its log-std from the"
                " observation (a Dense head, log_std), PPO's default before it"
                " learned one log-std per action dimension. Restore it into an"
                " agent built with log_std_state_independent=False and"
                " log_std_init=-1.0, the network it was saved from, its other"
                " settings unchanged."
            )
    if "log_std_param" in saved and "log_std_param" not in skeleton:
        if "log_std" in skeleton:
            raise ValueError(
                "This checkpoint's actor learns one log-std per action dimension"
                " (log_std_param), which the agent restoring it, built with"
                " log_std_state_independent=False, lacks. Restore it into an"
                " agent built with log_std_state_independent=True, PPO's"
                " default."
            )
    for key, value in skeleton.items():
        if key in saved:
            _check_log_std_head(value, saved[key])


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
