"""Transition datasets: logged or recorded transitions, kept on disk and
added to a replay.

A :class:`TransitionDataset` holds whole episodes as flat rows in time
order, each episode contiguous (the replay agents read single transitions;
TD-MPC2's offline datasets, :mod:`ajax.agents.TDMPC2.dataset`, hold
``[N, L, ...]`` episodes for its sequence sampler):

================  ==========  ===============================================
field             shape       content
================  ==========  ===============================================
``obs``           [N, O] f32  the env's own observation (not normalised)
``action``        [N, A] f32  the action taken, in agent units ([-1, 1])
``reward``        [N] f32     the env's own reward
``terminated``    [N] bool    a true terminal (no bootstrap)
``truncated``     [N] bool    a time limit (bootstrap on ``next_obs``)
``next_obs``      [N, O] f32  the next observation, the final one at an end
``is_expert``     [N] bool    whether the expert chose ``action``
``a_expert``      [N, A] f32  the expert's action at ``obs`` (NaN: unknown)
``return_to_go``  [N] f32     sum_k gamma^k r_{t+k} to the episode's end
``episode``       [N] i32     the episode's index
================  ==========  ===============================================

``return_to_go`` and ``episode`` follow from the others. ``return_to_go``
is Cal-QL's Monte-Carlo return (Nakamoto et al. 2023), cut at a time limit.

:func:`record` rolls an expert out, :func:`export` reads a run's replay,
:func:`fill_replay` adds a dataset to one. :func:`save` and :func:`load`
follow TD-MPC2's file conventions: an uncompressed ``.npz`` of plain arrays
and a JSON ``meta`` string, no pickle.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ajax.environments.interaction import collect_experience_from_expert_policy
from ajax.state import EnvironmentConfig

#: Identifies :func:`save`'s file format (bump the version on any change of
#: the stored fields).
FORMAT, VERSION = "ajax.transition_dataset", 1
FIELDS = (
    *("obs", "action", "reward", "terminated", "truncated", "next_obs"),
    *("is_expert", "a_expert", "return_to_go", "episode"),
)
_FLAGS = ("terminated", "truncated", "is_expert")


@dataclasses.dataclass(frozen=True, eq=False)
class TransitionDataset:
    """Whole episodes of transitions (the module docstring's schema).

    ``is_expert`` defaults to False and ``a_expert`` to NaN (logged data);
    ``gamma`` discounts ``return_to_go``; ``env`` names the environment.
    """

    # Array-likes, kept as numpy arrays of the schema's dtypes.
    obs: Any
    action: Any
    reward: Any
    terminated: Any
    truncated: Any
    next_obs: Any
    is_expert: Any = None
    a_expert: Any = None
    gamma: float = 0.99
    env: str = "env"
    return_to_go: np.ndarray = dataclasses.field(init=False)
    episode: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        n, set_ = len(self.reward), object.__setattr__  # frozen
        defaults = {
            "is_expert": np.zeros(n),
            "a_expert": np.full(np.shape(self.action), np.nan),
        }
        for f in FIELDS[:8]:
            x = defaults[f] if getattr(self, f) is None else getattr(self, f)
            set_(self, f, np.asarray(x, bool if f in _FLAGS else np.float32))
            if len(getattr(self, f)) != n:
                raise ValueError(f"{f} has {len(getattr(self, f))} rows, reward {n}")
        o, a = self.obs.shape, self.action.shape
        if self.next_obs.shape != o or self.a_expert.shape != a:
            raise ValueError(
                f"next_obs, a_expert must be shaped as obs {o}, action {a}"
            )
        ends = self.terminated | self.truncated
        if n == 0 or not ends[-1]:
            raise ValueError("a dataset holds whole episodes: its last row ends one")
        set_(self, "return_to_go", returns_to_go(self.reward, ends, self.gamma))
        set_(self, "episode", np.cumsum(np.r_[0, ends[:-1]]).astype(np.int32))

    @property
    def n_episodes(self) -> int:
        return int(self.episode[-1]) + 1


def returns_to_go(reward: np.ndarray, ends: np.ndarray, gamma: float) -> np.ndarray:
    """``sum_k gamma^k reward[t + k]`` to the end of ``t``'s episode."""
    out, g = np.zeros_like(reward), 0.0
    for t in range(len(reward) - 1, -1, -1):
        g = reward[t] + gamma * g * (not ends[t])
        out[t] = g
    return out


def _whole_episodes(
    streams: dict, gamma: float, env: str, cut: bool
) -> TransitionDataset:
    """The dataset of per-env streams ``[n_envs, T, ...]`` in time order,
    env after env: each env's rows to its last episode end, from its first
    row or, if ``cut`` (its first episode may lack its start), from the row
    after its first episode end."""
    keep = []
    for e in streams["terminated"] | streams["truncated"]:
        (at,) = np.nonzero(e)
        keep.append(slice(at[0] + 1 if cut else 0, at[-1] + 1) if len(at) else slice(0))
    rows = {
        f: np.concatenate([r[s] for r, s in zip(x, keep, strict=True)])
        for f, x in streams.items()
    }
    if not len(rows["reward"]):
        raise ValueError("no whole episode to keep")
    return TransitionDataset(**rows, gamma=gamma, env=env)


def record(
    expert_policy: Any,
    env_args: EnvironmentConfig,
    key: jax.Array,
    n_steps: int,
    gamma: float = 0.99,
    env: str = "env",
) -> TransitionDataset:
    """The whole episodes of ``n_steps`` steps per env of ``expert_policy``
    (Ajax's expert collector, so its state is kept for a stateful one): the
    actions in agent units, as a replay keeps them. Each env's last
    episode, cut by the end of the recording, is left out."""
    t = collect_experience_from_expert_policy(
        expert_policy,
        rng=key,
        mode=env_args.mode,
        env_args=env_args,
        n_timesteps=n_steps,
    )
    rows = {"obs": t.raw_obs, "action": t.action, "reward": t.reward[..., 0]}
    rows |= {
        "terminated": t.terminated[..., 0] > 0,
        "truncated": t.truncated[..., 0] > 0,
    }
    rows |= {"next_obs": t.raw_next_obs, "a_expert": t.a_expert}
    streams = {f: np.swapaxes(np.asarray(x), 0, 1) for f, x in rows.items()}
    streams["is_expert"] = np.ones_like(streams["reward"], bool)
    return _whole_episodes(streams, gamma, env, cut=False)


def export(
    state: Any, gamma: float = 0.99, env: str = "env", seed_index: Optional[int] = None
) -> TransitionDataset:
    """The whole episodes in a replay agent's replay (``state`` as its
    ``train`` returned it; ``seed_index`` picks a seed), oldest first, env
    after env. A ring that wrapped has lost its oldest episode's start:
    that episode is left out. The observations are the replay's
    ``raw_obs`` and ``raw_next_obs`` where it keeps them (else, after a
    termination, ``next_obs`` is the reset observation it stores)."""
    replay = state.collector_state.buffer_state
    exp = {k: np.asarray(v) for k, v in replay.experience.items()}
    index, full = np.asarray(replay.current_index), np.asarray(replay.is_full)
    if (index.ndim == 1) != (seed_index is not None):
        raise ValueError("pass seed_index exactly when the state has a seed axis")
    if seed_index is not None:
        exp = {k: v[seed_index] for k, v in exp.items()}
        index, full = index[seed_index], full[seed_index]
    rows, wrapped = int(index), bool(full)
    order = (
        np.roll(np.arange(exp["obs"].shape[1]), -rows) if wrapped else np.arange(rows)
    )
    flag = lambda k: exp[k][..., 0] > 0  # noqa: E731
    no_expert = np.zeros(exp["reward"].shape[:-1], bool)
    streams = {
        "obs": exp.get("raw_obs", exp["obs"]),
        "action": exp["action"],
        "reward": exp["reward"][..., 0],
        "terminated": flag("terminated"),
        "truncated": flag("truncated"),
        "next_obs": exp.get("raw_next_obs", exp["next_obs"]),
        "is_expert": flag("is_expert") if "is_expert" in exp else no_expert,
        "a_expert": exp.get("a_expert", np.full_like(exp["action"], np.nan)),
    }
    return _whole_episodes(
        {f: x[:, order] for f, x in streams.items()}, gamma, env, wrapped
    )


def _replay_rows(data: TransitionDataset) -> dict[str, np.ndarray]:
    """``data``'s rows in the fields a replay keeps (:func:`init_buffer`)."""
    col = lambda x: np.asarray(x, np.float32)[:, None]  # noqa: E731
    # After a termination collection stores the reset observation
    # (interaction.bootstrap_obs), which ASAC's differential target
    # continues from: here the next episode's first.
    after_end = np.roll(data.obs, -1, axis=0)
    next_obs = np.where(data.terminated[:, None], after_end, data.next_obs)
    return {
        "obs": data.obs,
        "action": data.action,
        "reward": col(data.reward),
        "terminated": col(data.terminated),
        "truncated": col(data.truncated),
        "next_obs": next_obs,
        "raw_obs": data.obs,
        "is_expert": col(data.is_expert),
        "a_expert": data.a_expert,
        "raw_next_obs": data.next_obs,
        "return_to_go": col(data.return_to_go),
    }


def fill_replay(buffer: Any, buffer_state: Any, data: TransitionDataset) -> Any:
    """``buffer_state`` with ``data``'s rows added, one row per env per
    add as collection adds its own: each env's replay row gets a contiguous
    run of ``data``, so a sequence replay reads its episodes whole. A replay
    already holding rows drops its oldest when full, as collection does.

    A flat replay samples a row with its successor, so it never samples its
    newest row: each env's last row of ``data`` trains only once a later
    row follows it (collection's next step).

    Raises ``ValueError`` when the replay keeps fields ``data`` does not
    hold or of another width, when ``data``'s rows do not divide among its
    envs (flashbax advances every env's row together), or when they
    outnumber its capacity."""
    schema, rows = buffer_state.experience, _replay_rows(data)
    n_envs, capacity = jax.tree.leaves(schema)[0].shape[:2]
    missing = sorted(set(schema) - set(rows))
    if missing:
        raise ValueError(f"the replay keeps {missing}, which a dataset does not hold")
    for k, x in schema.items():
        if rows[k].shape[1:] != x.shape[2:]:
            raise ValueError(
                f"replay {k} rows {x.shape[2:]}, dataset {rows[k].shape[1:]}"
            )
    n = len(data.reward)
    if n % n_envs:
        raise ValueError(f"{n} dataset rows do not divide among {n_envs} envs")
    if n > n_envs * capacity:
        raise ValueError(f"{n} dataset rows, a replay of {n_envs * capacity}")

    def stream(k: str) -> jax.Array:  # [n // n_envs, n_envs, ...]
        x = rows[k].reshape(n_envs, n // n_envs, *rows[k].shape[1:])
        return jnp.asarray(x.swapaxes(0, 1), schema[k].dtype)

    def add(state: Any, row: dict) -> tuple[Any, None]:
        return buffer.add(state, row), None

    return jax.lax.scan(add, buffer_state, {k: stream(k) for k in schema})[0]


def save(data: TransitionDataset, path: str) -> None:
    """Write ``data`` to ``path``, through a temporary file renamed, so an
    interrupted save leaves no partial file there."""
    meta = {"format": FORMAT, "version": VERSION, "gamma": data.gamma, "env": data.env}
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"{path}.tmp", "wb") as f:
        arrays = {k: getattr(data, k) for k in FIELDS}
        np.savez(f, meta=np.asarray(json.dumps(meta)), **arrays)
    os.replace(f"{path}.tmp", path)


def load(path: str) -> TransitionDataset:
    """The dataset :func:`save` wrote to ``path``; raises ``ValueError`` on
    another format or version. ``return_to_go`` and ``episode`` are
    recomputed from the rows."""
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        if (meta.get("format"), meta.get("version")) != (FORMAT, VERSION):
            raise ValueError(f"{path} is not a {FORMAT} v{VERSION} file")
        rows = {k: d[k] for k in FIELDS[:8]}
    return TransitionDataset(**rows, gamma=float(meta["gamma"]), env=str(meta["env"]))


__all__ = ["TransitionDataset", "export", "fill_replay", "load", "record", "save"]
