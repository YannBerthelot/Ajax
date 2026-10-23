"""Ajax's side of the TQC check against bayesgroup/tqc_pytorch (test_tqc_original.py): one update from
each of the authors' recorded states (benchmarks/reference/conformance_tqc.py), written back in the
recorder's names.

The state maps one to one, kernels transposed: the actor's trunk and its last layer's mean and log-std
rows; the 5 critics and their targets along the ensemble axis; every Adam's moments and step;
log_alpha. The recorded batch is served by the update's sampler, the two noises (the next actions',
the actor's) taped by role; the update compiles once per env and mode.
"""

from __future__ import annotations

import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ajax
import ajax.agents.TQC.train_TQC as train_TQC
from tests.conformance import kit

WIDTH = 64  # the recorder's (benchmarks/reference/conformance_tqc.py)
#: main.py's and trainer.py's settings, in Ajax's names.
SETTINGS: dict[str, Any] = {
    "actor_architecture": (str(WIDTH), "relu") * 2,
    "critic_architecture": (str(WIDTH), "relu") * 3,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "num_critics": 5,
    "n_quantiles": 25,
    "top_quantiles_to_drop_per_net": 2,
}
FAULTS: dict[str, dict[str, Any]] = {"gamma": {"gamma": 0.991}}
LAYERS = {"layers_0": "fc0", "layers_2": "fc1", "layers_4": "fc2"}
HEADS = {"mean": "mean", "log_std": "log_std", "model": "last_fc"}
LEAF = re.compile(
    r"\.(actor_state|critic_state|alpha)\.(params|target_params|step|opt_state\[0\]\.(mu|nu|count))"
    r"(?:\['params'\]\['ensemble'\]|\['params'\])?(.*)"
)


def reference_names(path: Any) -> tuple[list[str], str] | None:
    """An Ajax leaf's names in the record (five for an ensemble leaf) and its kind: "kernel",
    "bias", "scalar" or "count"; None for leaves the authors' code has no counterpart of."""
    m = LEAF.fullmatch(jax.tree_util.keystr(path))
    if m is None:
        return None
    owner, kind, moment, rest = m.groups()
    if owner == "actor_state" and kind == "target_params":
        return None  # their actor has no target
    if kind == "step":  # a TrainState's step: its Adam's count
        moment = "count"
    tail = {"mu": ".exp_avg", "nu": ".exp_avg_sq", "count": ".step", None: ""}[moment]
    if owner == "alpha":
        name = ("opt.log_alpha" + tail) if moment else "log_alpha"
        return [name], "count" if moment == "count" else "scalar"
    if owner == "actor_state":
        nets = ["actor"]
    else:
        prefix = "target" if kind == "target_params" else "critic"
        nets = [f"{prefix}.qf{i}" for i in range(SETTINGS["num_critics"])]
    if moment:
        nets = ["opt." + n for n in nets]
    if moment == "count":
        return [f"{nets[0]}.fc0.weight{tail}"], "count"
    keys = re.findall(r"\['([^']+)'\]", rest)
    module = LAYERS.get(keys[-2]) or HEADS[keys[-2]]
    leaf = "weight" if keys[-1] == "kernel" else "bias"
    return [f"{n}.{module}.{leaf}{tail}" for n in nets], keys[-1]


def to_ajax(template: Any, arrays: dict, dtype: Any) -> Any:
    def put(path: Any, x: Any) -> Any:
        found = reference_names(path)
        if found is None:
            return jnp.asarray(x, dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x
        names, kind = found
        parts = [np.asarray(arrays[f"in.{n}"], np.float64) for n in names]
        if kind == "count":
            return jnp.asarray(int(parts[0].reshape(-1)[0]), x.dtype)
        parts = [p.T if kind == "kernel" else p for p in parts]
        value = (np.stack(parts) if len(parts) > 1 else parts[0]).reshape(x.shape)
        return jnp.asarray(value, dtype)

    return jax.tree_util.tree_map_with_path(put, template)


def to_reference(state: Any) -> dict:
    """``state`` in the record's names."""
    out = {}
    for path, x in jax.tree_util.tree_leaves_with_path(state):
        found = reference_names(path)
        if found is None:
            continue
        names, kind = found
        value = np.asarray(x, np.float64)
        if kind == "count":
            out[names[0]] = value
            continue
        values = list(value) if len(names) > 1 else [value]
        for n, v in zip(names, values, strict=True):
            out[n] = (
                v.T if kind == "kernel" else v.reshape(-1) if kind == "scalar" else v
            )
    return out


def run(records: Any, precision: str, fault: str | None = None) -> dict[str, dict]:
    """Ajax's state after one update from each record, in the record's names, by record name."""
    dtype = np.float64 if precision == "f64" else np.float32
    given: dict = {}
    sample = train_TQC.sample_replay

    def injected(agent_state: Any, buffer: Any, key: Any, *args: Any, **kw: Any) -> Any:
        b, carries = sample(agent_state, buffer, key, *args, **kw)
        values = {
            "obs": given["obs"],
            "action": given["action"],
            "next_obs": given["next_obs"],
            "reward": given["reward"],
            "terminated": 1.0 - given["not_done"],
            "raw_obs": given["obs"],
            "raw_next_obs": given["next_obs"],
        }
        new = {
            f: jnp.asarray(v, getattr(b, f).dtype).reshape(getattr(b, f).shape)
            for f, v in values.items()
            if getattr(b, f, None) is not None
        }
        if getattr(b, "truncated", None) is not None:
            new["truncated"] = jnp.zeros_like(b.truncated)
        return b.replace(**new), carries

    envs = sorted({p.name.rsplit("_", 1)[0] for p in records.glob("*.npz")})
    out: dict = {}
    train_TQC.sample_replay = injected
    try:
        for env_id in envs:
            out |= _env_updates(records, env_id, dtype, fault, given)
    finally:
        train_TQC.sample_replay = sample
    return out


def _env_updates(
    records: Any, env_id: str, dtype: Any, fault: str | None, given: dict
) -> dict:
    agent = ajax.TQC(env_id, **(SETTINGS | FAULTS.get(fault or "", {})))
    step = kit.capture_step(agent, 2_000)
    template, update = step.fresh(jax.random.PRNGKey(0)), step.update
    assert update is not None, "TQC passes its update to its train loop"
    arrays = {p.stem: dict(np.load(p)) for p in sorted(records.glob(f"{env_id}_*.npz"))}
    fields = ("obs", "action", "next_obs", "reward", "not_done")

    def from_batch(state: Any, batch: dict) -> Any:
        given.clear()
        given.update(batch)
        return update(state, None)[0]

    def batch(a: dict, dt: Any) -> dict:
        return {f: np.asarray(a[f"batch.{f}"], dt) for f in fields}

    # The tape's roles and keys, from one recording in float32: every record starts from the
    # template's key (the records carry no Ajax key), and keys do not depend on the precision.
    first = next(iter(arrays.values()))
    with kit.recording() as rec:
        jax.block_until_ready(
            jax.jit(from_batch)(
                to_ajax(template, first, np.float32), batch(first, np.float32)
            )
        )
    draws = kit.by_role(rec.log)
    roles = _roles(draws)
    table: dict = {}

    def with_draws(state: Any, batch: dict, tape: dict) -> Any:
        table.update(tape)
        return from_batch(state, batch)

    out = {}
    with jax.enable_x64(dtype == np.float64), kit.replaying(table) as replay:
        replayed = jax.jit(with_draws)
        for n, a in arrays.items():
            tape = {
                role: (
                    np.stack([k for k, _ in draws[role]]),
                    np.asarray(a["noise"][i], dtype).reshape(1, *role[2]),
                )
                for i, role in enumerate(roles)
            }
            out[n] = to_reference(
                replayed(to_ajax(template, a, dtype), batch(a, dtype), tape)
            )
    assert not replay.misses
    return out


def _roles(draws: dict) -> list:
    """The update's two normal draws in the authors' order: the next actions', then the actor's."""
    normal = {r[1]: r for r in draws if r[0] == "normal"}
    [nxt] = [r for site, r in normal.items() if site.endswith(":sample_next_actions")]
    [actor] = [
        r for site, r in normal.items() if site.endswith(":soft_policy_objective")
    ]
    return [nxt, actor]
