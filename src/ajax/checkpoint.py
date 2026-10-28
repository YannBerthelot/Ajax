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
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from ajax.random_keys import derive_key
from ajax.state import CollectorState


def _to_host_numpy(state: Any) -> Any:
    return jax.tree.map(
        lambda x: np.asarray(x) if hasattr(x, "shape") else x,
        state,
    )


#: The key under which a checkpoint of a pytree states its format, and the
#: format written now. 1: a replay keeps the environment's own observations
#: and rewards. 2: collection draws on its own stream of keys. 3: an env
#: whose resets are keyed by episode keeps its episode counters. A
#: checkpoint without the key is older (0).
FORMAT_KEY = "ajax_checkpoint_format"
FORMAT = 3


def save_checkpoint(state: Any, path: str) -> None:
    """Save the pytree data of ``state`` to ``path``. Atomic write."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    host_state = _to_host_numpy(state)
    state_dict = serialization.to_state_dict(host_state)
    if isinstance(state_dict, dict):
        state_dict = {**state_dict, FORMAT_KEY: FORMAT}
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
    saved_format = 0
    if isinstance(state_dict, dict):
        state_dict = dict(state_dict)
        saved_format = state_dict.pop(FORMAT_KEY, 0)
        _check_raw_replay(skeleton, state_dict, saved_format)
    skeleton_dict = serialization.to_state_dict(skeleton)
    _check_episode_keys(skeleton_dict, state_dict, saved_format)
    _check_memory_final_norm(skeleton_dict, state_dict)
    _check_log_std_head(skeleton_dict, state_dict)
    state_dict = _without_removed_fields(skeleton, state_dict)
    state_dict = _with_added_fields(skeleton, state_dict)
    state_dict = _with_replay_next_obs(skeleton, state_dict)
    state_dict = _without_segment_raw_obs(skeleton, state_dict)
    state_dict = _with_new_empty_fields(skeleton_dict, state_dict)
    if _took_a_joint_clip(skeleton):
        state_dict = _without_own_gradient_clip(skeleton_dict, state_dict)
    state_dict = _with_agent_obs_stats(skeleton, state_dict)
    restored = serialization.from_state_dict(skeleton, state_dict)
    _check_leaf_shapes(skeleton, restored)
    if saved_format < 2:
        restored = _with_collection_stream(restored)
    return jax.tree.map(
        lambda x: jax.device_put(x) if isinstance(x, np.ndarray) else x,
        restored,
    )


def _with_collection_stream(state: Any) -> Any:
    """A state saved before collection had its own stream of keys (format
    2) kept as its collector's key the carry of its last collection, an
    ancestor of the updates' key: resumed as it is, collection would hand
    out keys the updates already drew. Its stream restarts on a key derived
    from the updates' ("collector"). The world models' row collectors had a
    stream of their own already."""
    if type(state) in (tuple, list) and state:  # (state, logs) as saved
        return type(state)([_with_collection_stream(state[0]), *state[1:]])
    collector = getattr(state, "collector_state", None)
    if type(collector) is not CollectorState:
        return state
    keys = jnp.asarray(state.rng)
    seeds = keys.reshape(-1, keys.shape[-1])  # one key per seed
    derived = jax.vmap(lambda key: derive_key(key, "collector"))(seeds)
    collector = collector.replace(rng=derived.reshape(keys.shape))
    return state.replace(collector_state=collector)


def _check_leaf_shapes(skeleton: Any, restored: Any) -> None:
    """Refuse a checkpoint whose network parameters or optimizer states do
    not fit the skeleton's, saved by an agent with other networks: flax
    grafts any array, and the run would fail later, far from the cause, or
    broadcast silently. Only those leaves, which the agent's configuration
    fixes, are compared: a replay legitimately outgrows its skeleton (the
    world models size theirs to the run, a resumed run's skeleton covers no
    step yet)."""

    def shapes(tree: Any) -> dict[str, tuple]:
        flat = jax.tree_util.tree_flatten_with_path(tree)[0]
        return {jax.tree_util.keystr(path): np.shape(leaf) for path, leaf in flat}

    want, got = shapes(skeleton), shapes(restored)
    bad = [
        f"{path} is {got[path]} in the checkpoint, {shape} in this agent"
        for path, shape in want.items()
        if ("params" in path or "opt_state" in path)
        and path in got
        and got[path] != shape
    ]
    if bad:
        raise ValueError(
            "The checkpoint does not fit this agent's state: "
            + "; ".join(bad[:5])
            + (f"; and {len(bad) - 5} more" if len(bad) > 5 else "")
            # A state class's ClassVar: how to restore its former layouts.
            + getattr(skeleton, "restore_hint", "")
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


def _without_segment_raw_obs(skeleton: Any, state_dict: Any) -> Any:
    """Restore a UDRL segment buffer saved while it held the env's own
    observations twice, in ``obs`` and, for a run with an expert, in
    ``raw_obs``: the field is dropped, and ``obs`` holds them. A buffer of
    observations the env normalised is refused: each row kept the statistics
    of its collection, which are not saved, and UDRL now normalises the rows
    it reads itself.

    Contract shim (CLAUDE.md, refactoring rule 10) for checkpoints written
    between ``raw_obs``'s addition and its removal, both on 2026-10-17, and
    for UDRL's env normalisation before. Delete it on 2026-11-17.
    """
    from ajax.agents.UDRL.buffer import SegmentBuffer

    saved = state_dict.get("buffer") if isinstance(state_dict, dict) else None
    segments = getattr(skeleton, "buffer", None)
    if not isinstance(segments, SegmentBuffer) or not isinstance(saved, dict):
        return state_dict
    if _saved_obs_normalised(state_dict):
        raise ValueError(
            "This UDRL checkpoint's segment buffer holds observations the"
            " environment normalised, each with the statistics of its"
            " collection, which are not saved; UDRL now keeps them raw and"
            " normalises them itself. Resume the run with the Ajax version that"
            " wrote it, or start it afresh."
        )
    saved = {k: v for k, v in saved.items() if k != "raw_obs"}
    return {**state_dict, "buffer": saved}


def _saved_obs_normalised(saved_agent: dict) -> bool:
    """Whether a saved agent state's environment normalised observations:
    its state holds their statistics."""
    env = (saved_agent.get("collector_state") or {}).get("_env_state") or {}
    stats = env.get("normalization_info") or env.get("info", {}).get(
        "normalization_info"
    )
    return bool(stats) and stats.get("obs") is not None


def _with_new_empty_fields(skeleton: Any, saved: Any) -> Any:
    """Give a checkpoint saved before a field was added the fields its
    skeleton holds empty, at any depth: None, as ``Transition.raw_next_obs``
    in every rollout the collector and the last-rollout placeholder hold,
    or a mapping holding nothing, as the entry in ext_state of a stateless
    extension appended to a stack the checkpoint was saved without (SAC's
    expert prefill, which joins its stack as ExpertPrefill). An empty field
    has no data to restore; a field with data stays required, so a
    mismatched checkpoint still fails.

    Expand-then-contract shim (CLAUDE.md, refactoring rule 10), as
    ``_with_replay_next_obs``.
    """
    if not isinstance(skeleton, dict) or not isinstance(saved, dict):
        return saved
    out = dict(saved)
    for key, value in skeleton.items():
        if key in out:
            out[key] = _with_new_empty_fields(value, out[key])
        elif value is None or (isinstance(value, dict) and not value):
            out[key] = value
    return out


def _took_a_joint_clip(skeleton: Any) -> bool:
    """Whether skeleton's class declares joint_gradient_clip (a
    ClassVar[bool]): PPO's and APO's networks each clipped their own
    gradient until one clip over both took its place
    (update_steps.apply_gradients_clipped_together).
    Another agent's optimiser saved behind a clip is refused, as flax
    refuses any other optimiser layout."""
    return getattr(skeleton, "joint_gradient_clip", False)


def _without_removed_fields(skeleton: Any, saved: Any) -> Any:
    """Drop the saved fields a state class deleted and lists in its
    ``removed_fields`` (a ``ClassVar[frozenset[str]]``), at any depth, so a
    checkpoint saved with them restores. Flax refuses a field its class
    lacks; a field the class does not list still fails, so a checkpoint of
    another agent or layout stays a mismatch.

    Expand-then-contract shim (CLAUDE.md, refactoring rule 10): a class
    drops a name from ``removed_fields`` once no checkpoint holding that
    field remains.
    """
    if not isinstance(saved, dict):
        return saved
    removed: frozenset[str] = getattr(skeleton, "removed_fields", frozenset())
    children = dict(_children(skeleton))
    return {
        key: _without_removed_fields(children[key], value) if key in children else value
        for key, value in saved.items()
        if key not in removed
    }


def _with_added_fields(skeleton: Any, saved: Any) -> Any:
    """Give a checkpoint saved before a state class gained the fields it
    lists in ``added_fields`` (a ``ClassVar[frozenset[str]]``) those fields,
    at any depth, in the skeleton's structure, as NaN (-1 in integer
    leaves, as ajax.log.unevaluated): the run that saved it never computed
    them. A field the skeleton holds as None stays for
    _with_new_empty_fields. A missing field the class does not list still
    fails, so a checkpoint of another agent or layout stays a mismatch.

    Expand-then-contract shim (CLAUDE.md, refactoring rule 10): a class
    drops a name from ``added_fields`` once no checkpoint without that
    field remains.
    """
    if not isinstance(saved, dict):
        return saved
    children = dict(_children(skeleton))
    out = {
        key: _with_added_fields(children[key], value) if key in children else value
        for key, value in saved.items()
    }
    added: frozenset[str] = getattr(skeleton, "added_fields", frozenset())
    for name in added.difference(out.keys()):
        if children.get(name) is not None:
            never_computed = jax.tree.map(_never_computed, children[name])
            out[name] = serialization.to_state_dict(never_computed)
    return out


def _never_computed(leaf: Any) -> np.ndarray:
    """NaN in leaf's shape and type; -1 in an integer leaf."""
    leaf = np.asarray(leaf)
    return np.full_like(leaf, -1 if np.issubdtype(leaf.dtype, np.integer) else np.nan)


def _children(node: Any) -> Iterator[tuple[str, Any]]:
    """``node``'s children by their keys in its state dict: jax's key paths
    name a class's or NamedTuple's field, a dict's key and a list's or
    tuple's index as flax's state dict does."""
    children, _ = jax.tree_util.tree_flatten_with_path(
        node, is_leaf=lambda child: child is not node
    )
    for path, child in children:
        if path:  # empty when ``node`` is itself a leaf
            yield jax.tree_util.keystr(path, simple=True), child


def _without_own_gradient_clip(skeleton: Any, saved: Any) -> Any:
    """Give an optimiser state saved when its optimiser clipped the
    gradient first (get_adam_tx's max_grad_norm: a chain whose clip holds an
    empty state) the layout of the unclipped optimiser restoring it. The
    clip holds no data, so every moment and step count is kept. PPO's and
    APO's networks clipped their own gradients until each took its
    reference code's one clip over both
    (update_steps.apply_gradients_clipped_together); only their states take
    it (their classes declare joint_gradient_clip).

    Expand-then-contract shim (CLAUDE.md, refactoring rule 10), as the one
    that gives an old replay buffer its next observations.
    """
    if not isinstance(skeleton, dict) or not isinstance(saved, dict):
        return saved
    out = dict(saved)
    for key, value in skeleton.items():
        if key not in out:
            continue
        old = out[key]
        if key == "opt_state" and _behind_a_clip(value, old):
            out[key] = old["1"]
        else:
            out[key] = _without_own_gradient_clip(value, old)
    return out


def _behind_a_clip(new: Any, old: Any) -> bool:
    """Whether ``old`` is ``new``'s layout chained after a clip's empty
    state."""
    if not isinstance(old, dict) or old.keys() != {"0", "1"}:
        return False
    empty = isinstance(old["0"], dict) and not old["0"]
    return empty and _same_keys(new, old["1"])


def _same_keys(a: Any, b: Any) -> bool:
    """Whether two state dicts nest the same keys."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return not isinstance(a, dict) and not isinstance(b, dict)
    return a.keys() == b.keys() and all(_same_keys(a[k], b[k]) for k in a)


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


def _with_agent_obs_stats(skeleton: Any, state_dict: Any) -> Any:
    """Give an on-policy agent's checkpoint saved while its observation
    statistics lived in its env's normaliser (before 2026-10-23) those
    statistics in its collector, where they live now: a gymnax env's sets,
    one per env, merged into one (the parallel variance of Chan, Golub and
    LeVeque, 1979, cited from memory), a brax batch's rows, all the same,
    read once. The env state keeps its reward statistics, and a brax one
    its raw observation (its info's, else un-normalised from the saved
    statistics). The saved last observation stays as the env normalised
    it: one step acts on it.

    Expand-then-contract shim (CLAUDE.md, refactoring rule 10), as
    ``_with_replay_next_obs``.
    """
    from ajax.utils import online_denormalize
    from ajax.wrappers import RAW_OBS_KEY

    collector = getattr(skeleton, "collector_state", None)
    stats = getattr(collector, "obs_norm_info", None)
    saved = state_dict.get("collector_state") if isinstance(state_dict, dict) else None
    if stats is None or not isinstance(saved, dict) or saved.get("obs_norm_info"):
        return state_dict
    env = dict(saved["_env_state"])
    brax = isinstance(env.get("info"), dict)
    holder = dict(env["info"]) if brax else env
    norm = holder.get("normalization_info")
    if not isinstance(norm, dict) or norm.get("obs") is None:
        return state_dict
    count, mean, mean_2 = (
        np.asarray(norm["obs"][k]) for k in ("count", "mean", "mean_2")
    )
    if brax:
        raw = holder.pop(RAW_OBS_KEY, None)
        if raw is None:
            raw = online_denormalize(env["obs"], count, mean, mean_2)
        env["obs"] = np.asarray(raw, np.asarray(env["obs"]).dtype)
        count, mean, mean_2 = count[..., :1, :], mean[..., :1, :], mean_2[..., :1, :]
    else:
        seeds = np.ndim(stats.count) - 2  # the leading axes of a seed batch
        c = count.reshape(count.shape[: seeds + 1])  # (*seeds, n_envs)
        m = mean.reshape(*c.shape, -1)
        m2 = mean_2.reshape(*c.shape, -1)
        total = c.sum(-1, keepdims=True)[..., None]  # (*seeds, 1, 1)
        merged = (c[..., None] * m).sum(-2, keepdims=True) / total
        mean_2 = (m2 + c[..., None] * (m - merged) ** 2).sum(-2, keepdims=True)
        count, mean = total, merged
    assert collector is not None  # it holds the statistics
    env_state = collector._env_state
    kept = env_state.info if brax else vars(env_state)
    if "normalization_info" in kept:
        holder["normalization_info"] = {**norm, "obs": None}
    else:
        del holder["normalization_info"]
    if brax:
        env["info"] = holder
    arrays = {"var": mean_2 / count, "count": count, "mean": mean, "mean_2": mean_2}
    moved: dict[str, Any] = {k: v.astype(np.float32) for k, v in arrays.items()}
    moved["returns"] = None
    collector_dict = {**saved, "_env_state": env, "obs_norm_info": moved}
    return {**state_dict, "collector_state": collector_dict}


def _check_raw_replay(skeleton: Any, saved: dict, saved_format: int) -> None:
    """Refuse a checkpoint of a replay agent written before its replay kept
    the environment's own observations and rewards (format 0), when its rows
    were normalised: the rewards when the environment normalises them, the
    observations when the saved environment state holds their statistics.
    Each row was normalised with the statistics of its collection, which
    are gone, so no migration recovers them, and this agent would normalise
    them again.

    Expand-then-contract (CLAUDE.md, refactoring rule 10): delete this check
    once no such checkpoint remains.
    """
    from ajax.environments.interaction import reward_statistics

    if saved_format >= 1:
        return
    agent, saved_agent = skeleton, saved
    if isinstance(skeleton, (tuple, list)) and skeleton:
        agent, saved_agent = skeleton[0], saved.get("0", {})
    collector = getattr(agent, "collector_state", None)
    if collector is None or collector.buffer_state is None:
        return
    obs_normalised = _saved_obs_normalised(saved_agent)
    if obs_normalised or reward_statistics(collector.env_state) is not None:
        raise ValueError(
            "This checkpoint was saved before a replay agent kept the"
            " environment's own observations and rewards in its replay: its"
            " rows were normalised with the statistics of their collection,"
            " which are not saved, and this agent would normalise them again."
            " Resume the run with the Ajax version that wrote it, or start it"
            " afresh."
        )


def _check_episode_keys(skeleton: Any, saved: Any, saved_format: int) -> None:
    """Refuse a checkpoint whose envs kept no episode counters for an agent
    whose envs key their resets by episode (ajax.wrappers.episode_start_key):
    resumed, every env's count would restart and replay the starts of its
    first episodes. Refuse the reverse too, which flax would report as an
    unknown field."""
    from ajax.wrappers import EPISODE_ROOT_KEY

    keyed = _holds_key(skeleton, EPISODE_ROOT_KEY)
    saved_keyed = _holds_key(saved, EPISODE_ROOT_KEY)
    if keyed and not saved_keyed:
        when = (
            f"before environment resets were keyed by episode (format {saved_format})"
            if saved_format < 3
            else "with episode keys off"
        )
        raise ValueError(
            f"This checkpoint was saved {when}: it holds no episode counters, so"
            " a resumed run would restart every environment's episode count and"
            " replay the starts of its first episodes. Resume it with episode"
            " keys off, as it was trained: build_env_from_id(env_id, ...,"
            " episode_keys=False), or"
            " ajax.environments.create.with_episode_keys(env, on=False) for an"
            " environment you built. Or start the run afresh."
        )
    if saved_keyed and not keyed:
        raise ValueError(
            "This checkpoint's environments keyed their resets by episode (it"
            " holds their episode counters), which the agent restoring it,"
            " built with episode_keys=False, lacks. Restore it into an agent"
            " whose environment keeps episode keys on, the default."
        )


def _holds_key(state_dict: Any, name: str) -> bool:
    """Whether a state dict holds name as a key, at any depth."""
    if not isinstance(state_dict, dict):
        return False
    return name in state_dict or any(_holds_key(v, name) for v in state_dict.values())


def checkpoint_exists(path: str) -> bool:
    return os.path.isfile(os.path.abspath(path))
