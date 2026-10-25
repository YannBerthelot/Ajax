"""Ajax's side of the REDQ check against watchernyu/REDQ (test_redq_original.py): one step (20 critic
updates, then the actor's and the temperature's) from each of REDQ's recorded states
(benchmarks/reference/conformance_redq.py), written back in REDQ's names.

The state maps one to one, kernels transposed: the policy's trunk, mean and log_std heads; the 10 Q
networks and their targets along the ensemble axis; every Adam's moments and step; log_alpha. The 20
recorded batches are served by the key each critic update samples with, the subsets and noises taped
by role; the step compiles once per env and mode. Ajax's REDQ reads the temperature REDQ's
``__init__`` leaves for its first step and REDQ's tanh log-det, so nothing is substituted.
"""

from __future__ import annotations

import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

import ajax
import ajax.agents.REDQ.REDQ as redq_module
from tests.conformance import kit

#: REDQSACAgent's defaults (target entropy 'auto'), in Ajax's names.
SETTINGS: dict[str, Any] = {
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "alpha_learning_rate": 3e-4,
    "gamma": 0.99,
    "batch_size": 256,
    "tau": 0.005,
    "target_entropy_per_dim": -1.0,
    "num_critic_updates": 20,
    "num_critics": 10,
    "subset_size": 2,
    "learning_starts": 1_000,
}
FAULTS: dict[str, dict[str, Any]] = {"gamma": {"gamma": 0.991}}
LAYERS = {"layers_0": "hidden_layers.0", "layers_2": "hidden_layers.1"}
HEADS = {
    "mean": "last_fc_layer",
    "log_std": "last_fc_log_std",
    "model": "last_fc_layer",
}
LEAF = re.compile(
    r"\.(actor_state|critic_state|alpha)\.(params|target_params|step|opt_state\[0\]\.(mu|nu|count))"
    r"(?:\['params'\]\['ensemble'\]|\['params'\])?(.*)"
)


def reference_names(path: Any) -> tuple[list[str], str] | None:
    """An Ajax leaf's names in REDQ's record (ten for an ensemble leaf) and its kind: "kernel",
    "bias", "scalar" or "count"; None for leaves REDQ has no counterpart of."""
    m = LEAF.fullmatch(jax.tree_util.keystr(path))
    if m is None:
        return None
    owner, kind, moment, rest = m.groups()
    if owner == "actor_state" and kind == "target_params":
        return None  # REDQ's policy has no target
    if kind == "step":  # a TrainState's step: its Adam's count
        moment = "count"
    tail = {"mu": ".exp_avg", "nu": ".exp_avg_sq", "count": ".step", None: ""}[moment]
    if owner == "alpha":
        name = ("opt.alpha" + tail) if moment else "log_alpha"
        return [name], "count" if moment == "count" else "scalar"
    nets = (
        ["policy"]
        if owner == "actor_state"
        else [f"q{i}{'_target' if kind == 'target_params' else ''}" for i in range(10)]
    )
    if moment:
        nets = ["opt." + n for n in nets]
    if moment == "count":
        first = "hidden_layers.0.weight"
        return [f"{n}.{first}{tail}" for n in nets[:1]], "count"
    keys = re.findall(r"\['([^']+)'\]", rest)
    layer = LAYERS.get(keys[-2]) if len(keys) > 1 else None
    module = layer or HEADS[keys[-2] if len(keys) > 1 else keys[0]]
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
    """``state`` in REDQ's names."""
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


def run(
    records: Any,
    precision: str,
    fault: str | None = None,
) -> dict[str, dict]:
    """Ajax's state after one step from each record, in REDQ's names, by record name."""
    dtype = np.float64 if precision == "f64" else np.float32
    # The 20 batches and their keys, set while the update is traced: arguments of the compiled
    # update, so each env compiles once.
    given: dict = {}
    sample = redq_module.sample_replay

    def injected(
        agent_state: Any, buffer: Any, key: Any, *args: Any, **kwargs: Any
    ) -> Any:
        b, carries = sample(agent_state, buffer, key, *args, **kwargs)
        if (
            "log" in given
        ):  # recording: the key, in program order (a pass-through callback)
            key = io_callback(
                given["log"], jax.ShapeDtypeStruct(key.shape, key.dtype), key
            )
        i = jnp.argmax(jnp.all(given["keys"] == kit._key_data(key), axis=-1))
        values = {
            "obs": given["obs"][i],
            "action": given["action"][i] / given["max_action"],
            "next_obs": given["next_obs"][i],
            "reward": given["reward"][i],
            "terminated": given["done"][i],
            "raw_obs": given["obs"][i],
            "raw_next_obs": given["next_obs"][i],
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
    redq_module.sample_replay = injected
    try:
        for env_id in envs:
            out.update(_env_updates(records, env_id, dtype, fault, given))
    finally:
        redq_module.sample_replay = sample
    return out


def _env_updates(
    records: Any,
    env_id: str,
    dtype: Any,
    fault: str | None,
    given: dict,
) -> dict:
    settings = dict(SETTINGS)
    settings.update(FAULTS.get(fault or "", {}))
    agent = ajax.REDQ(env_id, **settings)
    step = kit.capture_step(agent, 2_000)
    template, update = step.fresh(jax.random.PRNGKey(0)), step.update
    assert update is not None, "REDQ passes its update to its train loop"
    arrays = {p.stem: dict(np.load(p)) for p in sorted(records.glob(f"{env_id}_*.npz"))}
    max_action = float(next(iter(arrays.values()))["max_action"])
    fields = ("obs", "action", "next_obs", "reward", "done")

    def batches(a: dict, dt: Any) -> dict:
        return {f: np.asarray(a[f"batch.{f}"], dt) for f in fields}

    def from_batches(state: Any, keys: Any, batch: dict) -> Any:
        log = given.get("log")
        given.clear()
        given.update(batch, keys=keys, max_action=max_action)
        if log is not None:
            given["log"] = log
        return update(state, None)[0]

    # The 20 sampling keys and the tape's roles, record by record, in float32 (keys do not depend on
    # the precision; the tape logs on a callback thread, outside jax.enable_x64).
    keys, draws = {}, {}
    for n, a in arrays.items():
        logged: list = []

        def log(k: np.ndarray, logged: list = logged) -> np.ndarray:
            logged.append(np.asarray(k))
            return np.asarray(k)

        state = to_ajax(template, a, np.float32)
        no_keys = np.zeros((20, 2), np.uint32)
        with kit.recording() as rec:
            given["log"] = log
            jax.block_until_ready(
                jax.jit(from_batches)(state, no_keys, batches(a, np.float32))
            )
        given.pop("log", None)
        keys[n], draws[n] = np.stack(logged), kit.by_role(rec.log)
    roles = _roles(next(iter(draws.values())))
    table: dict = {}

    def with_draws(state: Any, keys: Any, batch: dict, tape: dict) -> Any:
        table.update(tape)
        return from_batches(state, keys, batch)

    out = {}
    with jax.enable_x64(dtype == np.float64), kit.replaying(table) as replay:
        replayed = jax.jit(with_draws)
        for n, a in arrays.items():
            tape = _tape(roles, draws[n], a, dtype)
            new = replayed(
                to_ajax(template, a, dtype),
                keys[n],
                batches(a, dtype),
                tape,
            )
            out[n] = to_reference(new)
    assert not replay.misses
    return out


def _roles(draws: dict) -> dict:
    """The tape's roles of the update's draws: the next actions' noise, the actor's, the subsets."""
    normal = [r for r in draws if r[0] == "normal"]
    nxt = [r for r in normal if len(draws[r]) == 20]
    actor = [r for r in normal if len(draws[r]) == 1]
    [subset] = [r for r in draws if r[0] == "choice"]
    assert len(nxt) == 1 and len(actor) == 1, sorted(draws)
    return {"next": nxt[0], "actor": actor[0], "subset": subset}


def _tape(roles: dict, draws: dict, a: dict, dtype: Any) -> dict:
    noise = np.asarray(a["noise"], np.float32)
    values = {
        "next": noise[:20],
        "actor": noise[20:],
        "subset": np.asarray(a["subset"], np.int32),
    }
    return {
        r: (
            np.stack([k for k, _ in draws[r]]),
            values[name].reshape(len(draws[r]), *r[2]),
        )
        for name, r in roles.items()
    }
