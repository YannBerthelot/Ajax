"""Ajax's side of the ReBRAC check against tinkoff-ai/ReBRAC (test_rebrac_original.py): one update
from each of ReBRAC's recorded states (benchmarks/reference/conformance_rebrac.py), written back in
ReBRAC's names. Ajax is the docs' recipe, TD3 with TD3+BC's actor term and TargetBC's penalty, at
ReBRAC's sizes and rates. Both are flax, so the state maps one to one: the actor's three hidden layers
and head, and each critic's three hidden layers, their LayerNorms and head, the two critics stacked on
a leading axis on both sides. The update count is set so that TD3's policy delay steps the actor where
ReBRAC's does (at even updates counted from 0). The recorded batch is injected where Ajax samples its
replay, its next actions as the rows' ``next_action``, and the recorded target noise taped. Float64
runs under ``jax.enable_x64``; each task's update compiles once per mode.
"""

from __future__ import annotations

import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ajax
import ajax.agents.TD3.TD3 as td3_module
from ajax.extensions.expert import ImitationLoss
from ajax.extensions.offline import TargetBC
from tests.conformance import kit
from tests.conformance.td3_sfujim import updater

#: ReBRAC's Config defaults (rebrac.py:33-50), as TD3's settings.
SETTINGS: dict[str, Any] = {
    "actor_architecture": ("256", "relu") * 3,
    "critic_architecture": ("256", "relu", "layernorm") * 3,
    "actor_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "batch_size": 1024,
    "tau": 5e-3,
    "gamma": 0.99,
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "learning_starts": 1_100,
}
#: The gymnax task Ajax's agent is built on, per recorded task (MountainCarEnds only ends episodes).
TASK = {"MountainCarEnds": "MountainCarContinuous-v0"}
ACTOR = {"hidden.layers_0": 0, "hidden.layers_2": 1, "hidden.layers_4": 2, "head": 3}
CRITIC = {"layers_0": "Dense_0", "layers_2": "LayerNorm_0", "layers_3": "Dense_1"}
CRITIC.update(
    {"layers_5": "LayerNorm_1", "layers_6": "Dense_2", "layers_8": "LayerNorm_2"}
)
CRITIC.update({"model": "Dense_3"})


def extensions(beta: float = 1.0) -> tuple:
    """actor_bc_coef 1 (the actions' dimension is 1: TD3+BC's mean is ReBRAC's sum) and
    critic_bc_coef ``beta``."""
    return (ImitationLoss(None, target="replay", alpha=1.0), TargetBC(beta=beta))


def rebrac_name(path: Any) -> str | None:
    """An Ajax leaf's name in ReBRAC's records; None for leaves ReBRAC has no counterpart of."""
    keys = [
        str(getattr(p, "key", getattr(p, "name", getattr(p, "idx", p)))) for p in path
    ]
    if not keys[0].endswith("_state") or keys[0][:-6] not in ("actor", "critic"):
        return None
    side, kind, rest = keys[0][:-6], keys[1], keys[2:]
    if kind == "step" and not rest:
        return f"{side}.step"
    if kind == "opt_state" and rest == ["0", "count"]:
        return f"{side}.opt.0.count"
    if kind in ("params", "target_params"):
        owner, rest = f"{'params' if kind == 'params' else 'target'}.params", rest[1:]
    elif kind == "opt_state" and rest[:2] in (["0", "mu"], ["0", "nu"]):
        owner, rest = f"opt.0.{rest[1]}.params", rest[3:]
    else:
        return None
    if side == "actor":
        return f"actor.{owner}.Dense_{ACTOR['.'.join(rest[:-1])]}.{rest[-1]}"
    return f"critic.{owner}.VmapCritic_0.{CRITIC[rest[-2]]}.{rest[-1]}"


def to_ajax(template: Any, arrays: dict, update: int, dtype: Any) -> Any:
    def put(path: Any, x: Any) -> Any:
        name = rebrac_name(path)
        if name is not None:
            value = np.asarray(arrays[f"in.{name}"]).reshape(x.shape)
            floating = jnp.issubdtype(x.dtype, jnp.floating)
            return jnp.asarray(value, dtype if floating else x.dtype)
        if jax.tree_util.keystr(path) == ".n_updates":
            return jnp.asarray(
                update + 1, x.dtype
            )  # TD3 counts the update before its test
        return jnp.asarray(x, dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x

    return jax.tree_util.tree_map_with_path(put, template)


def to_rebrac(state: Any) -> dict:
    out = {}
    for path, x in jax.tree_util.tree_leaves_with_path(state):
        name = rebrac_name(path)
        if name is not None:
            out[name] = np.asarray(
                x, np.float64 if jnp.issubdtype(x.dtype, jnp.floating) else x.dtype
            )
    return out


def run(records: pathlib.Path, precision: str, beta: float = 1.0) -> dict[str, dict]:
    """Ajax's state after one update from each record, in ReBRAC's names, by record name."""
    dtype = np.float64 if precision == "f64" else np.float32
    given: dict = {}
    sample = td3_module.sample_replay

    def injected(
        agent_state: Any, buffer: Any, key: Any, *args: Any, **kwargs: Any
    ) -> Any:
        b, carries = sample(agent_state, buffer, key, *args, **kwargs)
        values = {
            "obs": given["states"],
            "action": given["actions"],  # the box is [-1, 1]: ReBRAC's units are Ajax's
            "next_obs": given["next_states"],
            "reward": given["rewards"],
            "terminated": given["dones"],
            "raw_obs": given["states"],
            "raw_next_obs": given["next_states"],
            "next_action": given["next_actions"],
        }
        new = {
            f: jnp.asarray(v, getattr(b, f).dtype).reshape(getattr(b, f).shape)
            for f, v in values.items()
            if getattr(b, f, None) is not None
        }
        if getattr(b, "truncated", None) is not None:
            new["truncated"] = jnp.zeros_like(b.truncated)
        return b.replace(**new), carries

    tasks = sorted({p.name.rsplit("_", 1)[0] for p in records.glob("*.npz")})
    out: dict = {}
    td3_module.sample_replay = injected
    try:
        for task in tasks:
            out.update(_task_updates(records, task, dtype, beta, given))
    finally:
        td3_module.sample_replay = sample
    return out


def _task_updates(
    records: pathlib.Path, task: str, dtype: Any, beta: float, given: dict
) -> dict[str, dict]:
    agent = ajax.TD3(TASK.get(task, task), **SETTINGS, extensions=extensions(beta))
    step = kit.capture_step(agent, 2_000)
    template, update = step.fresh(jax.random.PRNGKey(0)), updater(step)
    arrays = {p.stem: dict(np.load(p)) for p in sorted(records.glob(f"{task}_*.npz"))}
    batches = {
        n: {k[len("batch.") :]: v for k, v in a.items() if k.startswith("batch.")}
        for n, a in arrays.items()
    }

    def from_batch(state: Any, batch: dict) -> Any:
        given.clear()
        given.update(batch)
        return update(state)

    def state_of(a: dict, dt: Any) -> Any:
        return to_ajax(template, a, int(a["update"]), dt)

    # The target noise's key, record by record: keys do not depend on the precision.
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
            out[n] = to_rebrac(replayed(state_of(a, dtype), batches[n], keys[n], noise))
    assert not replay.misses
    return out
