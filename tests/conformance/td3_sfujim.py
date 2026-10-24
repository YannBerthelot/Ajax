"""Ajax's side of the TD3 check against sfujim/TD3 (test_td3_sfujim.py): one update from each of
sfujim's recorded states (benchmarks/reference/conformance_sfujim_td3.py), written back in sfujim's
names. The state maps one to one, kernels transposed: both sum their critics' losses, both count the
update before the policy delay's test, and Ajax's critics read actions in the env's units, as sfujim's
do. The recorded batch is injected where Ajax samples its replay, and the recorded target noise taped.
Float64 runs under ``jax.enable_x64``; each env's update compiles once per mode, the batch and the
noise its arguments.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ajax
import ajax.agents.TD3.TD3 as td3_module
from tests.conformance import kit

#: sfujim's main.py settings (its noise levels are max_action times these: Ajax's [-1, 1] units).
SETTINGS: dict[str, Any] = {
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "exploration_noise": 0.1,
    "learning_starts": 1000,
}
FAULTS: dict[str, dict[str, Any]] = {"target_noise": {"target_policy_noise": 0.22}}
ACTOR = {"layers_0": "l1", "layers_2": "l2", "head": "l3"}
CRITIC = {"layers_0": ("l1", "l4"), "layers_2": ("l2", "l5"), "model": ("l3", "l6")}
LEAF = re.compile(
    r"\.(actor|critic)_state\.(params|target_params|opt_state\[0\]\.(mu|nu))\[.*'(layers_0|layers_2|head|model)'\]\['(kernel|bias)'\]"
)


def sfujim_names(path: Any) -> tuple[list[str], bool, bool] | None:
    """An Ajax leaf's sfujim names (two for the critic ensemble), and whether it is the ensemble and a
    kernel; None for leaves sfujim has no counterpart of."""
    m = LEAF.fullmatch(jax.tree_util.keystr(path).replace("['params']", ""))
    if m is None:
        return None
    side, kind, moment, layer, leaf = m.groups()
    if side == "actor" and kind == "target_params":
        owner = "actor_target"
    else:
        owner = {"params": side, "target_params": f"{side}_target"}.get(
            kind, f"opt.{side}"
        )
    tail = {None: "", "mu": ".exp_avg", "nu": ".exp_avg_sq"}[moment]
    modules = [ACTOR[layer]] if side == "actor" else list(CRITIC[layer])
    names = [
        f"{owner}.{n}.{'weight' if leaf == 'kernel' else 'bias'}{tail}" for n in modules
    ]
    return names, side == "critic", leaf == "kernel"


def to_ajax(template: Any, arrays: dict, total_it: int, dtype: Any) -> Any:
    def put(path: Any, x: Any) -> Any:
        found = sfujim_names(path)
        name = jax.tree_util.keystr(path)
        if found is not None:
            names, ensemble, kernel = found
            parts = [np.asarray(arrays[f"in.{n}"]) for n in names]
            parts = [p.T if kernel else p for p in parts]
            return jnp.asarray(
                (np.stack(parts) if ensemble else parts[0]).reshape(x.shape), dtype
            )
        if re.fullmatch(r"\.(actor|critic)_state\.(opt_state\[0\]\.count|step)", name):
            side = "actor" if "actor" in name else "critic"
            steps = [
                v
                for k, v in arrays.items()
                if k.startswith(f"in.opt.{side}.") and k.endswith(".step")
            ]
            return jnp.asarray(int(np.asarray(steps[0]).reshape(-1)[0]), x.dtype)
        if name == ".n_updates":
            return jnp.asarray(total_it, x.dtype)
        return jnp.asarray(x, dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x

    return jax.tree_util.tree_map_with_path(put, template)


def to_sfujim(state: Any) -> dict:
    out = {}
    for path, x in jax.tree_util.tree_leaves_with_path(state):
        found = sfujim_names(path)
        if found is None:
            continue
        names, ensemble, kernel = found
        value = np.asarray(x, np.float64)
        for n, v in zip(names, list(value) if ensemble else [value], strict=False):
            out[n] = v.T if kernel else v
    return out


def updater(step: kit.Step) -> Any:
    update = step.update
    assert update is not None, "TD3 passes its update to its train loop"
    return lambda state: update(state, None)[0]


def run(
    records: pathlib.Path,
    precision: str,
    fault: str | None = None,
    extensions: tuple = (),
) -> dict[str, dict]:
    """Ajax's state after one update from each record, in sfujim's names, by record name; TD3 with
    ``extensions`` (TD3+BC's term against sfujim/TD3_BC, whose state maps the same way)."""
    dtype = np.float64 if precision == "f64" else np.float32
    # The batch the update samples, set while it is traced: the compiled update takes it as an
    # argument, so each env compiles once, not once per record.
    given: dict = {}
    sample = td3_module.sample_replay

    def injected(
        agent_state: Any, buffer: Any, key: Any, *args: Any, **kwargs: Any
    ) -> Any:
        b, carries = sample(agent_state, buffer, key, *args, **kwargs)
        values = {
            "obs": given["state"],
            "action": given["action"] / given["max_action"],  # sfujim's: env units
            "next_obs": given["next_state"],
            "reward": given["reward"],
            "terminated": 1.0 - given["not_done"],
            "raw_obs": given["state"],
            "raw_next_obs": given["next_state"],
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
    td3_module.sample_replay = injected
    try:
        for env_id in envs:
            out.update(_env_updates(records, env_id, dtype, fault, given, extensions))
    finally:
        td3_module.sample_replay = sample
    return out


def _env_updates(
    records: pathlib.Path,
    env_id: str,
    dtype: Any,
    fault: str | None,
    given: dict,
    extensions: tuple,
) -> dict[str, dict]:
    settings = dict(SETTINGS)
    settings.update(FAULTS.get(fault or "", {}))
    step = kit.capture_step(ajax.TD3(env_id, **settings, extensions=extensions), 2_000)
    template, update = step.fresh(jax.random.PRNGKey(0)), updater(step)
    paths = sorted(records.glob(f"{env_id}_*.npz"))
    arrays = {p.stem: dict(np.load(p)) for p in paths}
    batches = {
        n: {k[len("batch.") :]: v for k, v in a.items() if k.startswith("batch.")}
        for n, a in arrays.items()
    }
    max_action = float(next(iter(arrays.values()))["max_action"])

    def from_batch(state: Any, batch: dict) -> Any:
        given.clear()
        given.update(batch, max_action=max_action)
        return update(state)

    def state_of(a: dict, dt: Any) -> Any:
        return to_ajax(template, a, int(a["in.total_it"]), dt)

    # The target noise's key, record by record. Keys do not depend on the precision, and the tape
    # logs on a callback thread, outside jax.enable_x64: recorded in float32.
    with kit.recording() as rec:
        recorded, keys = jax.jit(from_batch), {}
        for n, a in arrays.items():
            start = len(rec.log)
            jax.block_until_ready(recorded(state_of(a, np.float32), batches[n]))
            draws = kit.by_role(rec.log[start:])
            [role] = [
                r for r in draws if r[0] == "normal" and "compute_td3_td_target" in r[1]
            ]
            keys[n] = draws[role][0][0]
    table: dict = {}

    def with_noise(state: Any, batch: dict, key: Any, noise: Any) -> Any:
        table[(*role[:3], np.dtype(dtype).name)] = (key[None], noise[None])
        return from_batch(state, batch)

    out = {}
    with jax.enable_x64(dtype == np.float64), kit.replaying(table) as replay:
        replayed = jax.jit(with_noise)
        for n, a in arrays.items():
            noise = np.asarray(a["noise"], dtype).reshape(role[2])
            new = replayed(state_of(a, dtype), batches[n], keys[n], noise)
            out[n] = to_sfujim(new)
    assert not replay.misses
    return out
