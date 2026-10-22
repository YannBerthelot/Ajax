"""Ajax's side of the CrossQ check against adityab/CrossQ (test_crossq_original.py): one update from
each of the authors' recorded states (benchmarks/reference/conformance_crossq.py), written back in
their names.

The state maps one to one, layer by layer in order: the actor's and the two critics' Dense and Batch
Renorm layers (Batch Renorm's running means, variances and pass counts; the authors' constant r_max
and d_max are module attributes in Ajax), the heads, every Adam's moments and count, the networks'
step counts and the temperature. The recorded batch is served to the update's replay sample, the
recorded normals by role: the next actions' and, on the policy delay's turn, the actor's. The update
compiles once per env and precision. Two differences of form change no number beyond the tolerance:
the authors' critic loss is half the sum over the two critics of each one's mean (Ajax's: the mean
over both), and their log-probability is of their own sample (Ajax's: sample_and_log_prob).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ajax
import ajax.agents.CrossQ.train_CrossQ as train_CrossQ
from tests.conformance import kit

#: The recorder's model in Ajax's names: train.py -algo crossq, 256 wide.
SETTINGS: dict[str, Any] = {
    "actor_width": 256,
    "critic_width": 256,
    "learning_starts": 1_000,
}
FAULTS: dict[str, dict[str, Any]] = {"gamma": {"gamma": 0.991}}
#: Ajax's layers, in order, and their names in the authors' modules.
LAYERS = {
    "layers_0": "BatchRenorm_0",
    "layers_1": "Dense_0",
    "layers_3": "BatchRenorm_1",
    "layers_4": "Dense_1",
    "layers_6": "BatchRenorm_2",
    "mean": "Dense_2",
    "log_std": "Dense_3",
    "model": "Dense_2",
}
SIDES = {"actor_state": "actor", "critic_state": "qf", "alpha": "ent"}


def reference_name(path: Any) -> str | None:
    """An Ajax leaf's name in the authors' record; None for leaves they have no counterpart of (the
    unused target copies, the collector, the keys, the temperature's step count)."""
    keys = [
        str(getattr(p, "key", getattr(p, "name", getattr(p, "idx", p)))) for p in path
    ]
    side = SIDES.get(keys[0])
    if side is None or len(keys) < 2:
        return None
    field, rest = keys[1], keys[2:]
    if field == "step":
        return None if side == "ent" else f"{side}.step"
    if field == "opt_state":  # ['0', 'mu' | 'nu' | 'count', ...]
        prefix, rest = f"{side}.opt.0.{rest[1]}", rest[2:]
        if not rest:
            return prefix
    elif field in ("params", "batch_stats"):
        prefix = f"{side}.{field}"
    else:
        return None
    rest = rest[1:] if rest[:1] == ["params"] else rest  # Ajax's variables collection
    if side == "ent":
        return f"{prefix}.log_ent_coef"
    *modules, leaf = rest  # a leaf's own name ("mean") is not a layer's
    names = ["VmapCritic_0" if m == "ensemble" else LAYERS.get(m, m) for m in modules]
    return ".".join(
        [prefix, *(n for n in names if n not in ("encoder", "network")), leaf]
    )


def to_ajax(template: Any, arrays: dict, dtype: Any) -> Any:
    def put(path: Any, x: Any) -> Any:
        if jax.tree_util.keystr(path) == ".n_updates":
            return jnp.asarray(arrays["n_updates"], x.dtype)
        name = reference_name(path)
        floating = jnp.issubdtype(x.dtype, jnp.floating)
        if name is None:  # an integer strongly typed: float64 runs keep its width
            return jnp.asarray(x, dtype) if floating else jnp.asarray(np.asarray(x))
        value = np.asarray(arrays[f"in.{name}"]).reshape(x.shape)
        return jnp.asarray(value, dtype if floating else x.dtype)

    return jax.tree_util.tree_map_with_path(put, template)


def to_reference(state: Any) -> dict:
    """``state`` in the authors' names."""
    out = {}
    for path, x in jax.tree_util.tree_leaves_with_path(state):
        name = reference_name(path)
        if name is not None:
            floating = jnp.issubdtype(x.dtype, jnp.floating)
            out[name] = np.asarray(x, np.float64 if floating else np.int64)
    return out


def run(records: Any, precision: str, fault: str | None = None) -> dict[str, dict]:
    """Ajax's state after one update from each record, in the authors' names, by record name."""
    dtype = np.float64 if precision == "f64" else np.float32
    given: dict = {}
    sample = train_CrossQ.sample_replay

    def injected(agent_state: Any, buffer: Any, key: Any, *args: Any) -> Any:
        b, carries = sample(agent_state, buffer, key, *args)
        values = {
            "obs": given["observations"],
            "action": given["actions"],
            "next_obs": given["next_observations"],
            "reward": given["rewards"],
            "terminated": given["dones"],
        }
        new = {
            f: jnp.asarray(v, getattr(b, f).dtype).reshape(getattr(b, f).shape)
            for f, v in values.items()
        }
        return b.replace(**new, truncated=jnp.zeros_like(b.truncated)), carries

    envs = sorted({p.name.rsplit("_", 1)[0] for p in records.glob("*.npz")})
    out: dict = {}
    train_CrossQ.sample_replay = injected
    try:
        for env_id in envs:
            out |= _env_updates(records, env_id, dtype, fault, given)
    finally:
        train_CrossQ.sample_replay = sample
    return out


def _env_updates(
    records: Any, env_id: str, dtype: Any, fault: str | None, given: dict
) -> dict:
    agent = ajax.CrossQ(env_id, **(SETTINGS | FAULTS.get(fault or "", {})))
    step = kit.capture_step(agent, 2_000)
    template, update = step.fresh(jax.random.PRNGKey(0)), step.update
    assert update is not None, "CrossQ passes its update to its train loop"
    arrays = {p.stem: dict(np.load(p)) for p in sorted(records.glob(f"{env_id}_*.npz"))}
    fields = ("observations", "actions", "next_observations", "rewards", "dones")

    def batch(a: dict, dt: Any) -> dict:
        return {f: np.asarray(a[f"batch.{f}"], dt) for f in fields}

    def from_batch(state: Any, b: dict) -> Any:
        given.clear()
        given.update(b)
        return update(state, None)[0]

    # Each record's draws by role, in float32: the keys Ajax draws them on.
    draws = {}
    for n, a in arrays.items():
        with kit.recording() as rec:
            new = jax.jit(from_batch)(
                to_ajax(template, a, np.float32), batch(a, np.float32)
            )
            jax.block_until_ready(new)
        draws[n] = kit.by_role(rec.log)
    table: dict = {}

    def with_draws(state: Any, b: dict, tape: dict) -> Any:
        table.update(tape)
        return from_batch(state, b)

    out = {}
    with jax.enable_x64(dtype == np.float64), kit.replaying(table) as replay:
        replayed = jax.jit(with_draws)
        for n, a in arrays.items():
            new = replayed(
                to_ajax(template, a, dtype), batch(a, dtype), _tape(draws[n], a)
            )
            out[n] = to_reference(new)
    assert not replay.misses
    return out


def _tape(draws: dict, a: dict) -> dict:
    """The record's normals by Ajax's roles: the next actions' first, then the actor's."""
    noise = np.asarray(a["noise"], np.float32)
    roles = {
        "next": [
            r for r in draws if r[0] == "normal" and "sample_next_actions" in r[1]
        ],
        "actor": [
            r for r in draws if r[0] == "normal" and "soft_policy_objective" in r[1]
        ],
    }
    assert len(roles["next"]) == 1 and len(roles["actor"]) <= 1, sorted(draws)
    assert len(noise) == 1 + len(roles["actor"]), (noise.shape, sorted(draws))
    tape = {}
    for i, role in enumerate([*roles["next"], *roles["actor"]]):
        keys = np.stack([k for k, _ in draws[role]])
        tape[role] = (keys, noise[i].reshape(len(keys), *role[2]))
    return tape
