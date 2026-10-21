"""Ajax's side of the AVG check against gauthamvasan/avg (test_avg_original.py): from each of AVG's
recorded states (benchmarks/reference/conformance_avg.py), the TD-error statistics' fold and the update
on the recorded transition, written back in AVG's names.

The state maps one to one, kernels transposed: the actor's trunk, mean and log-std heads; the critic
(Ajax's one-critic ensemble); both Adams; the TD-error scaler's reward, discount and squared-return
statistics (n, m, s as count, mean, mean_2) and the return G. AVG's MultivariateNormal reads exp(log_std)
as a variance (``LOG_STD_SCALE``, its log-std head maps at that scale onto Ajax's, which reads it as a
standard deviation). The update's next-action noise is taped; the update compiles once per env and mode.
"""

from __future__ import annotations

import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ajax
import ajax.agents.AVG.train_AVG as train_AVG
from ajax.environments.interaction import get_pi
from ajax.state import Transition
from tests.conformance import kit

#: avg.py main()'s defaults, in Ajax's names.
SETTINGS: dict[str, Any] = {
    "actor_learning_rate": 0.0063,
    "critic_learning_rate": 0.0087,
    "gamma": 0.99,
    "alpha_init": 0.07,
}
FAULTS: dict[str, dict[str, Any]] = {"gamma": {"gamma": 0.991}}
LAYERS = {"layers_0": "phi.0", "layers_2": "phi.2"}
HEADS = {"mean": "mu", "log_std": "log_std", "model": "q"}
#: Ajax's log-std head is AVG's times this: AVG's exp(log_std) is a variance, Ajax's a deviation.
LOG_STD_SCALE = 0.5
LEAF = re.compile(
    r"\.(actor_state|critic_state)\.(params|step|opt_state\[0\]\.(mu|nu|count))"
    r"(?:\['params'\]\['ensemble'\]|\['params'\])(.*)"
)
SCALER = {"reward": "reward_rms", "gamma": "gamma_rms", "G_return": "return_sq_rms"}
MOMENTS = {"count": "n", "mean": "m", "mean_2": "s"}


def reference_names(path: Any) -> tuple[str, str, float] | None:
    """An Ajax leaf's name in AVG's record, its kind ("kernel", "bias", "count", "stat") and the
    factor of Ajax's value over AVG's; None for leaves AVG has no counterpart of."""
    k = jax.tree_util.keystr(path)
    stat = re.fullmatch(r"\.(reward|gamma|G_return)\.(count|mean|mean_2|value)", k)
    if stat:
        owner, field = stat.groups()
        if field == "value":
            return ("G", "stat", 1.0) if owner == "G_return" else None
        return f"scaler.{SCALER[owner]}.{MOMENTS[field]}", "stat", 1.0
    count = re.fullmatch(
        r"\.(actor_state|critic_state)\.(step|opt_state\[0\]\.count)", k
    )
    if count:  # a TrainState's step and its Adam's count
        net = "actor" if count.group(1) == "actor_state" else "Q"
        return f"opt.{net}.phi.0.weight.step", "count", 1.0
    m = LEAF.fullmatch(k)
    if m is None:
        return None
    owner, _, moment, rest = m.groups()
    net = "actor" if owner == "actor_state" else "Q"
    keys = re.findall(r"\['([^']+)'\]", rest)
    module = LAYERS.get(keys[-2]) or HEADS[keys[-2]]
    leaf = "weight" if keys[-1] == "kernel" else "bias"
    tail = {"mu": ".exp_avg", "nu": ".exp_avg_sq", None: ""}[moment]
    name = f"{'opt.' if moment else ''}{net}.{module}.{leaf}{tail}"
    factor = 1.0
    if (
        net == "actor" and module == "log_std"
    ):  # W_ajax = s W_avg, so g_ajax = g_avg / s
        factor = {
            None: LOG_STD_SCALE,
            "mu": 1 / LOG_STD_SCALE,
            "nu": 1 / LOG_STD_SCALE**2,
        }[moment]
    return name, keys[-1], factor


def to_ajax(template: Any, arrays: dict, dtype: Any) -> Any:
    def put(path: Any, x: Any) -> Any:
        found = reference_names(path)
        if found is None:
            return jnp.asarray(x, dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x
        name, kind, factor = found
        value = np.asarray(arrays[f"in.{name}"], np.float64)
        if kind == "count":
            return jnp.asarray(int(value.reshape(-1)[0]), x.dtype)
        value = (value.T if kind == "kernel" else value) * factor
        floating = jnp.issubdtype(x.dtype, jnp.floating)
        return jnp.asarray(value.reshape(x.shape), dtype if floating else x.dtype)

    return jax.tree_util.tree_map_with_path(put, template)


def to_reference(state: Any) -> dict:
    out = {}
    for path, x in jax.tree_util.tree_leaves_with_path(state):
        found = reference_names(path)
        if found is None:
            continue
        name, kind, factor = found
        value = np.asarray(x, np.float64)
        if kind == "kernel":
            value = value.reshape(value.shape[-2:]).T
        elif kind in ("bias",):
            value = value.reshape(-1)
        elif kind in ("count", "stat"):
            value = value.reshape(())
        out[name] = value / factor
    return out


def run(records: Any, precision: str, fault: str | None = None) -> dict[str, dict]:
    """Ajax's state after the fold and the update from each record, in AVG's names, by record name."""
    dtype = np.float64 if precision == "f64" else np.float32
    envs = sorted({p.name.rsplit("_", 1)[0] for p in records.glob("*.npz")})
    out: dict = {}
    for env_id in envs:
        out |= _env_updates(records, env_id, dtype, fault)
    return out


def _env_updates(records: Any, env_id: str, dtype: Any, fault: str | None) -> dict:
    agent = ajax.AVG(env_id, **(SETTINGS | FAULTS.get(fault or "", {})))
    step = kit.capture_step(agent, 2_000)
    template, update = step.fresh(jax.random.PRNGKey(0)), step.update
    assert update is not None, "AVG passes its update to its train loop"
    config = agent.agent_config
    arrays = {p.stem: dict(np.load(p)) for p in sorted(records.glob(f"{env_id}_*.npz"))}

    def transition(state: Any, a: dict, dt: Any) -> Transition:
        row = lambda v: jnp.asarray(np.asarray(v, dt).reshape(1, -1))  # noqa: E731
        raw = row(a["action_pre"])
        actor = state.actor_state
        log_prob = get_pi(actor, actor.params, row(a["obs"]))[0].log_prob_from_raw(raw)
        return Transition(
            obs=row(a["obs"]),
            action=row(a["action"]),
            reward=row(a["reward"]),
            terminated=row(a["terminated"]),
            truncated=jnp.zeros((1, 1), dt),
            next_obs=row(a["next_obs"]),
            raw_obs=row(a["obs"]),
            log_prob=log_prob,
            raw_action=raw,
        )

    def one_step(state: Any, t: Transition) -> Any:
        return update(train_AVG.update_AVG_values(state, t, config), t)[0]

    # The next action's noise key, record by record, in float32 (keys do not depend on the precision;
    # the tape logs on a callback thread, outside jax.enable_x64).
    keys, roles = {}, {}
    for n, a in arrays.items():
        state = to_ajax(template, a, np.float32)
        with kit.recording() as rec:
            jax.block_until_ready(
                jax.jit(one_step)(state, transition(state, a, np.float32))
            )
        draws = kit.by_role(rec.log)
        [role] = [r for r in draws if r[0] == "normal"]
        keys[n], roles[n] = draws[role][0][0], role
    table: dict = {}

    def with_noise(state: Any, t: Transition, key: Any, noise: Any) -> Any:
        table[roles[next(iter(roles))]] = (key[None], noise[None])
        return one_step(state, t)

    out = {}
    with jax.enable_x64(dtype == np.float64), kit.replaying(table) as replay:
        replayed = jax.jit(with_noise)
        for n, a in arrays.items():
            state = to_ajax(template, a, dtype)
            role = roles[n]
            noise = np.asarray(a["noise.next"], np.float32).reshape(role[2])
            out[n] = to_reference(
                replayed(state, transition(state, a, dtype), keys[n], noise)
            )
    assert not replay.misses
    return out
