"""Ajax's side of the TD3 check against sfujim/TD3 (test_td3_sfujim.py): one update from each of
sfujim's recorded states (benchmarks/reference/conformance_sfujim_td3.py), written back in sfujim's
names. The state maps one to one, kernels transposed: both sum their critics' losses, both count the
update before the policy delay's test, and Ajax's critics read actions in the env's units, as sfujim's
do. The recorded batch is injected where Ajax samples its replay, and the recorded target noise taped.

usage: python -m tests.conformance.td3_sfujim RECORDS OUT f32|f64 [FAULT]

A process of its own: float64 needs JAX_ENABLE_X64 before JAX starts, and the replay seam patches a
module of the agent.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__" and sys.argv[3] == "f64":
    os.environ["JAX_ENABLE_X64"] = "1"

import pathlib
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ajax
import ajax.agents.TD3.train_TD3 as train_TD3
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
    records: pathlib.Path, out: pathlib.Path, precision: str, fault: str | None = None
) -> None:
    dtype = np.float64 if precision == "f64" else np.float32
    batch: dict = {}
    sample = train_TD3.sample_replay

    def injected(
        agent_state: Any, buffer: Any, key: Any, *args: Any, **kwargs: Any
    ) -> Any:
        b, carries = sample(agent_state, buffer, key, *args, **kwargs)
        values = {
            "obs": batch["state"],
            "action": batch["action"] / batch["max_action"],
            "next_obs": batch["next_state"],
            "reward": batch["reward"],
            "terminated": 1.0 - batch["not_done"],
            "raw_obs": batch["state"],
            "raw_next_obs": batch["next_state"],
        }
        new = {
            f: jnp.asarray(np.asarray(v, dtype).reshape(getattr(b, f).shape))
            for f, v in values.items()
            if getattr(b, f, None) is not None
        }
        if getattr(b, "truncated", None) is not None:
            new["truncated"] = jnp.zeros_like(b.truncated)
        return b.replace(**new), carries

    train_TD3.sample_replay = injected
    out.mkdir(parents=True, exist_ok=True)
    for env_id in sorted({p.name.rsplit("_", 1)[0] for p in records.glob("*.npz")}):
        step = kit.capture_step(
            ajax.TD3(env_id, **(SETTINGS | FAULTS.get(fault or "", {}))), 2_000
        )
        template = step.fresh(jax.random.PRNGKey(0))
        update = updater(step)
        for path in sorted(records.glob(f"{env_id}_*.npz")):
            arrays = dict(np.load(path))
            batch.clear()
            batch.update(
                {
                    k[len("batch.") :]: v
                    for k, v in arrays.items()
                    if k.startswith("batch.")
                }
            )
            batch["max_action"] = float(
                arrays["max_action"]
            )  # sfujim's actions are in env units
            state = to_ajax(template, arrays, int(arrays["in.total_it"]), dtype)
            with kit.recording() as rec:
                jax.block_until_ready(jax.jit(update)(state))
            draws = kit.by_role(rec.log)
            [noise] = [
                r for r in draws if r[0] == "normal" and "compute_td3_td_target" in r[1]
            ]
            table = {
                noise: (
                    np.stack([draws[noise][0][0]]),
                    np.asarray(arrays["noise"], dtype).reshape((1, *noise[2])),
                )
            }
            with kit.replaying(table) as replay:
                new = jax.jit(update)(state)
                jax.block_until_ready(new)
            assert not replay.misses
            np.savez(out / path.name, **to_sfujim(new))


if __name__ == "__main__":
    run(
        pathlib.Path(sys.argv[1]),
        pathlib.Path(sys.argv[2]),
        sys.argv[3],
        sys.argv[4] if len(sys.argv) > 4 else None,
    )
