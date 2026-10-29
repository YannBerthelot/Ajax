"""Ajax's side of the RVI-SAC check against yhisaki/average-reward-drl (test_rvisac_original.py): one
update from each of their recorded states (benchmarks/reference/conformance_rvisac.py), written back in
their names.

The state maps one to one. Their actor's one output layer splits into Ajax's mean head (its first
dim A rows) and log-std head (the others), kernels transposed; their MultiLinear twin critics are
Ajax's ensemble as they are (both store (in, out) kernels), their biases' middle axis dropped; their
reset critic is the reset cost's, kernels transposed. The five Adams, the temperature, the reset cost
and the offsets fq and fq_reset map to theirs. The recorded batch is injected where Ajax samples its
replay; the actor's and the next actions' noises are taped, the only draws that yield values
(asserted), so no draw differs between the two precisions.
"""

from __future__ import annotations

import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ajax.agents.RVISAC.RVISAC as rvisac_module
from ajax.agents.centering import RVIBatchMean
from ajax.agents.reset_cost import ResetCostSettings
from tests.conformance import kit
from tests.probing import envs as probe_envs

ACTION = 2  # the recorder's dim A
SMALL = ("16", "relu", "16", "relu")
#: rvi_sac.yaml's settings at the recorder's widths, in Ajax's names.
SETTINGS: dict[str, Any] = {
    "actor_architecture": SMALL,
    "critic_architecture": SMALL,
    "reset_cost": ResetCostSettings(architecture=("8", "relu", "8", "relu")),
    "batch_size": 64,
    "learning_starts": 1_000,
}
FAULTS: dict[str, dict[str, Any]] = {
    "kappa": {"centering": RVIBatchMean(rate=0.0055)},
    "eps": {
        "reset_cost": ResetCostSettings(0.0011, architecture=("8", "relu", "8", "relu"))
    },
}
#: The recorder's task shapes: observations of 3, actions in [-1, 1]^2; the update never steps it.
ENV = probe_envs.Spec(
    "rvisac-shapes",
    lambda st, a, key: (st, 0.0, False),
    lambda st: jnp.zeros(3),
    obs_dim=3,
    box=(-1.0, 1.0, ACTION),
)
ACTOR = {"layers_0": "0", "layers_2": "2", "mean": "4", "log_std": "4"}
CRITIC = {"layers_0": "1", "layers_2": "3", "model": "5"}
RESET = {"Dense_0": "1", "Dense_1": "3", "Dense_2": "5"}
MOMENTS = {"mu": "exp_avg", "nu": "exp_avg_sq"}
SCALARS = {
    ".offset": "fq",
    ".reset_cost.rate": "fq_reset",
    ".alpha.params['log_alpha']": "temperature.value",
    ".reset_cost.cost.params['cost']": "reset_cost.value",
}
COUNTS = {
    "actor_state": "actor.0.weight",
    "critic_state": "critic.1.weight",
    "reset_cost.critic": "critic_reset.1.weight",
    "alpha": "temperature.value",
    "reset_cost.cost": "reset_cost.value",
}


def reference_name(path: Any) -> tuple[str, str] | None:
    """An Ajax leaf's name in the record and how it maps: "kernel" (transposed), "ensemble" (as
    it is), "mean" / "log_std" (a half of their output layer), "count" or "scalar"; None for a leaf
    their record has no counterpart of."""
    k = jax.tree_util.keystr(path)
    if k in SCALARS:
        return SCALARS[k], "scalar"
    for owner, first in COUNTS.items():
        if k in (f".{owner}.opt_state[0].count", f".{owner}.step"):
            return f"opt.{first}.step", "count"
    m = re.fullmatch(r"\.alpha\.opt_state\[0\]\.(mu|nu)\['log_alpha'\]", k)
    if m:
        return f"opt.temperature.value.{MOMENTS[m.group(1)]}", "scalar"
    m = re.fullmatch(r"\.reset_cost\.cost\.opt_state\[0\]\.(mu|nu)\['cost'\]", k)
    if m:
        return f"opt.reset_cost.value.{MOMENTS[m.group(1)]}", "scalar"
    m = re.fullmatch(
        r"\.(actor_state|critic_state|reset_cost\.critic|reset_cost)"
        r"\.(params|target_params|opt_state\[0\]\.(mu|nu))(.*)",
        k,
    )
    if m is None:
        return None
    owner, kind, moment, rest = m.groups()
    keys = re.findall(r"\['([^']+)'\]", rest)
    if owner == "actor_state" and kind == "target_params":
        return None  # their actor has no target
    leaf = "weight" if keys[-1] == "kernel" else "bias"
    if owner == "actor_state":
        net, module, how = "actor", ACTOR[keys[-2]], keys[-2]
        how = how if how in ("mean", "log_std") else "kernel"
    elif owner == "critic_state":
        net, module, how = "critic", CRITIC[keys[-2]], "ensemble"
        net = "critic_target" if kind == "target_params" else net
    else:
        net, module, how = "critic_reset", RESET[keys[-2]], "kernel"
        net = "critic_reset_target" if kind == "target_params" else net
    if moment:
        return f"opt.{net}.{module}.{leaf}.{MOMENTS[moment]}", how
    return f"{net}.{module}.{leaf}", how


def from_reference(value: np.ndarray, how: str, leaf_is_kernel: bool) -> np.ndarray:
    """Their array as Ajax's leaf holds it."""
    if how in ("mean", "log_std"):
        half = value[:ACTION] if how == "mean" else value[ACTION:]
        return half.T if leaf_is_kernel else half
    if how == "kernel" and leaf_is_kernel:
        return value.T
    return value


def to_ajax(template: Any, arrays: dict, dtype: Any) -> Any:
    def put(path: Any, x: Any) -> Any:
        found = reference_name(path)
        if found is None:
            return jnp.asarray(x, dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x
        name, how = found
        value = np.asarray(arrays[f"in.{name}"], np.float64)
        if how == "count":
            return jnp.asarray(int(value.reshape(-1)[0]), x.dtype)
        is_kernel = jax.tree_util.keystr(path).endswith("['kernel']")
        return jnp.asarray(
            from_reference(value, how, is_kernel).reshape(x.shape), dtype
        )

    return jax.tree_util.tree_map_with_path(put, template)


def to_reference(state: Any) -> dict:
    """Ajax's state in their names; their output layer reassembled from the two heads."""
    out: dict = {}
    halves: dict = {}
    for path, x in jax.tree_util.tree_leaves_with_path(state):
        found = reference_name(path)
        if found is None or found[1] == "count":
            continue
        name, how = found
        value = np.asarray(x, np.float64)
        is_kernel = jax.tree_util.keystr(path).endswith("['kernel']")
        if how in ("mean", "log_std"):
            halves.setdefault(name, {})[how] = value.T if is_kernel else value
        elif how == "kernel" and is_kernel:
            out[name] = value.T
        elif how == "ensemble" and not is_kernel:
            out[name] = value[:, None, :]  # MultiLinear's (n, 1, out) bias
        else:
            out[name] = value
    for name, parts in halves.items():
        out[name] = np.concatenate([parts["mean"], parts["log_std"]])
    return out


def run(records: Any, precision: str, fault: str | None = None) -> dict[str, dict]:
    """Ajax's state after one update from each record, in their names, by record name."""
    dtype = np.float64 if precision == "f64" else np.float32
    given: dict = {}  # the update's batch, set while it is traced (an argument of the compiled update)
    sample = rvisac_module.sample_replay

    def injected(
        agent_state: Any, buffer: Any, key: Any, *args: Any, **kwargs: Any
    ) -> Any:
        b, carries = sample(agent_state, buffer, key, *args, **kwargs)
        values = {
            "obs": given["state"],
            "action": given["action"],
            "next_obs": given["next_state"],
            "reward": given["reward"],
            "terminated": given["terminated"],
            "truncated": given["truncated"],
        }
        new = {
            f: jnp.asarray(v, getattr(b, f).dtype).reshape(getattr(b, f).shape)
            for f, v in values.items()
        }
        return b.replace(**new), carries

    settings = dict(SETTINGS)
    settings.update(FAULTS.get(fault or "", {}))
    env, params = ENV.make()
    agent = rvisac_module.RVISAC(env, env_params=params, **settings)
    rvisac_module.sample_replay = injected
    try:
        return _updates(records, agent, dtype, given)
    finally:
        rvisac_module.sample_replay = sample


def _updates(records: Any, agent: Any, dtype: Any, given: dict) -> dict:
    step = kit.capture_step(agent, 2_000)
    template, update = step.fresh(jax.random.PRNGKey(0)), step.update
    assert update is not None, "RVI-SAC passes its update to its train loop"
    arrays = {p.stem: dict(np.load(p)) for p in sorted(records.glob("rvisac_*.npz"))}
    fields = ("state", "action", "next_state", "reward", "terminated", "truncated")

    def from_batch(state: Any, batch: dict) -> Any:
        given.clear()
        given.update(batch)
        return update(state, None)[0]

    # The noises' keys, record by record, in float32 (keys do not depend on the precision).
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
    assert set(roles) == {actor, nxt}, sorted(roles)
    table: dict = {}

    def with_noise(state: Any, batch: dict, tape: dict) -> Any:
        table.update(tape)
        return from_batch(state, batch)

    out = {}
    with jax.enable_x64(dtype == np.float64), kit.replaying(table) as replay:
        replayed = jax.jit(with_noise)
        for n, a in arrays.items():
            tape = {
                actor: (keys[n][actor][None], tape_value(a["draw.action"], actor)),
                nxt: (keys[n][nxt][None], tape_value(a["draw.next_action"], nxt)),
            }
            batch = {f: np.asarray(a[f"batch.{f}"], dtype) for f in fields}
            out[n] = to_reference(replayed(to_ajax(template, a, dtype), batch, tape))
    assert not replay.misses
    return out


def tape_value(draw: np.ndarray, role: tuple) -> np.ndarray:
    """Their standard-normal draw as the tape gives it for one draw of role (its sample axis)."""
    return np.asarray(draw, np.float32).reshape(1, *role[2])


def deterministic_actions(records: Any, precision: str) -> dict[str, np.ndarray]:
    """Ajax's evaluation action on each record's states that has some, from its recorded actor."""
    dtype = np.float64 if precision == "f64" else np.float32
    env, params = ENV.make()
    agent = rvisac_module.RVISAC(env, env_params=params, **SETTINGS)
    template = kit.capture_step(agent, 2_000).fresh(jax.random.PRNGKey(0))
    out = {}
    with jax.enable_x64(dtype == np.float64):
        for path in sorted(records.glob("rvisac_*.npz")):
            a = dict(np.load(path))
            if "act.state" not in a:
                continue
            actor = to_ajax(template, a, dtype).actor_state
            policy = actor.apply_fn(actor.params, jnp.asarray(a["act.state"], dtype))
            out[path.stem] = np.asarray(policy.mean(), np.float64)
    return out
