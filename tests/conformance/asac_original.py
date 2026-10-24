"""Ajax's side of the ASAC check against BobMak/asac (test_asac_original.py): one update from each of
ASAC's recorded states (benchmarks/reference/conformance_asac.py), written back in ASAC's names.

The state maps one to one, kernels transposed: SB3's actor (latent_pi, mu, log_std) onto Ajax's trunk
and heads, the two Qsa critics (fc1-fc3) and their targets onto the ensemble, both Adams (Ajax's sit
after its gradient clip in an optax chain), theta and the termination penalty. The recorded batch is
injected where Ajax samples its replay; the actor's and the next actions' noises are taped, the only
draws that yield values (asserted), so no draw differs between the two precisions. ``PendulumEnds``
(the recorder's two-torque, terminating Pendulum) is built for Ajax as a probe env of the same shapes:
the update never steps it.
"""

from __future__ import annotations

import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ajax
import ajax.agents.ASAC.train_ASAC as train_ASAC
from tests.conformance import kit
from tests.probing import envs as probe_envs

#: InvertedPendulum-v5's asac.yaml, in Ajax's names (beta 5: a fixed alpha of 0.2).
SETTINGS: dict[str, Any] = {
    "actor_learning_rate": 1e-4,
    "critic_learning_rate": 5e-4,
    "batch_size": 256,
    "tau": 0.005,
    "tau_theta": 0.005,
    "alpha_init": 0.2,
    "fixed_alpha": True,
    "max_grad_norm": 10.0,
    "p_0": 20.0,
    "learning_starts": 1_000,
}
FAULTS: dict[str, dict[str, Any]] = {"tau": {"tau": 0.0055}}
ACTOR = {
    "layers_0": "latent_pi.0",
    "layers_2": "latent_pi.2",
    "mean": "mu",
    "log_std": "log_std",
}
CRITIC = {"layers_0": "fc1", "layers_2": "fc2", "model": "fc3"}
#: Ajax's stand-in for the recorder's PendulumEnds, of the same shapes (obs 3, a torque in [-2, 2]^2).
ENVS = {
    "PendulumEnds": probe_envs.Spec(
        "pendulum-ends",
        lambda st, a, key: (st, 0.0, False),
        lambda st: jnp.zeros(3),
        obs_dim=3,
        box=(-2.0, 2.0, 2),
    )
}
LEAF = re.compile(
    r"\.(actor_state|critic_state)\.(params|target_params|step|opt_state\[1\]\[0\]\.(mu|nu|count))"
    r"(?:\['params'\]\['ensemble'\]|\['params'\])?(.*)"
)
SCALARS = {".theta": "theta", ".episode_termination_penalty": "penalty"}


def reference_names(path: Any) -> tuple[list[str], str] | None:
    """An Ajax leaf's names in ASAC's record (two for an ensemble leaf) and its kind: "kernel",
    "bias", "scalar" or "count"; None for leaves ASAC's record has no counterpart of."""
    k = jax.tree_util.keystr(path)
    if k in SCALARS:
        return [SCALARS[k]], "scalar"
    m = LEAF.fullmatch(k)
    if m is None:
        return None
    owner, kind, moment, rest = m.groups()
    if kind == "step":
        moment = "count"
    if owner == "actor_state" and kind == "target_params":
        return None  # SB3's actor has no target
    nets = (
        ["actor"]
        if owner == "actor_state"
        else [f"q{i}{'_target' if kind == 'target_params' else ''}" for i in range(2)]
    )
    if moment == "count":
        first = "latent_pi.0.weight" if owner == "actor_state" else "fc1.weight"
        return [f"opt.{nets[0]}.{first}.step"], "count"
    keys = re.findall(r"\['([^']+)'\]", rest)
    module = (ACTOR if owner == "actor_state" else CRITIC)[keys[-2]]
    leaf = "weight" if keys[-1] == "kernel" else "bias"
    tail = {"mu": ".exp_avg", "nu": ".exp_avg_sq", None: ""}[moment]
    prefix = "opt." if moment else ""
    return [f"{prefix}{n}.{module}.{leaf}{tail}" for n in nets], keys[-1]


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
    out = {}
    for path, x in jax.tree_util.tree_leaves_with_path(state):
        found = reference_names(path)
        if found is None:
            continue
        names, kind = found
        value = np.asarray(x, np.float64)
        values = list(value) if len(names) > 1 else [value]
        for n, v in zip(names, values, strict=True):
            out[n] = v.T if kind == "kernel" else v
    return out


def run(records: Any, precision: str, fault: str | None = None) -> dict[str, dict]:
    """Ajax's state after one update from each record, in ASAC's names, by record name."""
    dtype = np.float64 if precision == "f64" else np.float32
    given: dict = {}  # the update's batch, set while it is traced (an argument of the compiled update)
    sample = train_ASAC.sample_replay

    def injected(
        agent_state: Any, buffer: Any, key: Any, *args: Any, **kwargs: Any
    ) -> Any:
        b, carries = sample(agent_state, buffer, key, *args, **kwargs)
        values = {
            "obs": given["obs"],
            "action": given["action"],  # SB3 stores [-1, 1] actions, as Ajax
            "next_obs": given["next_obs"],
            "reward": given["reward"],
            "terminated": given["done"],
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
    train_ASAC.sample_replay = injected
    try:
        for env_id in envs:
            out |= _env_updates(records, env_id, dtype, fault, given)
    finally:
        train_ASAC.sample_replay = sample
    return out


def _env_updates(
    records: Any, env_id: str, dtype: Any, fault: str | None, given: dict
) -> dict:
    settings = SETTINGS | FAULTS.get(fault or "", {})
    if env_id in ENVS:
        env, params = ENVS[env_id].make()
        agent = ajax.ASAC(env, env_params=params, **settings)
    else:
        agent = ajax.ASAC(env_id, **settings)
    step = kit.capture_step(agent, 2_000)
    template, update = step.fresh(jax.random.PRNGKey(0)), step.update
    assert update is not None, "ASAC passes its update to its train loop"
    arrays = {p.stem: dict(np.load(p)) for p in sorted(records.glob(f"{env_id}_*.npz"))}
    fields = ("obs", "action", "next_obs", "reward", "done")

    def from_batch(state: Any, batch: dict) -> Any:
        given.clear()
        given.update(batch)
        return update(state, None)[0]

    # The noises' keys, record by record, in float32 (keys do not depend on the precision; the tape
    # logs on a callback thread, outside jax.enable_x64).
    keys: dict = {}
    for n, a in arrays.items():
        batch = {f: np.asarray(a[f"batch.{f}"], np.float32) for f in fields}
        with kit.recording() as rec:
            state = to_ajax(template, a, np.float32)
            jax.block_until_ready(jax.jit(from_batch)(state, batch))
        draws = kit.by_role(rec.log)
        keys[n] = {r: draws[r][0][0] for r in draws if r[0] == "normal"}
    roles = next(iter(keys.values()))
    actor = next(r for r in roles if r[1].endswith("soft_policy_objective"))
    nxt = next(r for r in roles if r[1].endswith("sample_next_actions"))
    # Every noise that yields values is taped (shape-only traces log nothing), so no draw differs
    # between the two precisions.
    assert set(roles) == {actor, nxt}, sorted(roles)
    table: dict = {}

    def with_noise(state: Any, batch: dict, tape: dict) -> Any:
        table.update(tape)
        return from_batch(state, batch)

    out = {}
    with jax.enable_x64(dtype == np.float64), kit.replaying(table) as replay:
        replayed = jax.jit(with_noise)
        for n, a in arrays.items():
            noise = np.asarray(
                a["noise"], np.float32
            )  # the actor's sample, then the next actions'
            tape = {
                actor: (keys[n][actor][None], noise[0].reshape(1, *actor[2])),
                nxt: (keys[n][nxt][None], noise[1].reshape(1, *nxt[2])),
            }
            batch = {f: np.asarray(a[f"batch.{f}"], dtype) for f in fields}
            out[n] = to_reference(replayed(to_ajax(template, a, dtype), batch, tape))
    assert not replay.misses
    return out
