"""Ajax's side of the APO check against xtma/apo (test_apo_original.py): one update from each of
APPO's recorded states (benchmarks/reference/conformance_apo.py), written back in rlpyt's names.

The state maps one to one, kernels transposed: MujocoFfModel's mean MLP (its tanh the actor's
``tanh_mean``) onto Ajax's actor trunk and mean head, its value MLP onto the one-critic ensemble, its
log_std onto the state-independent log-std, its one Adam onto Ajax's two (per coordinate, stepped on
one clipped gradient), eta and value_bias onto average_reward and b. ``APO.update`` runs on
the recorded rollout: a done that is not a time limit is a termination, a time limit a truncation,
next_obs each env's own observation before a reset, the stored log-probabilities the sampler's
Gaussian's. Each epoch's minibatch permutation is the record's, from rlpyt's rows (b T + t) to Ajax's
(t B + b); the update compiles once per env and mode.
"""

from __future__ import annotations

import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import ajax
from ajax.state import Transition
from tests.conformance import kit

T, B = 200, 10
#: mujoco_appo.py's algo config, the run grid's discount 1.0 and gae_lambda, lr_eta and
#: rm_vbias_coeff at Ajax's defaults, in Ajax's names.
SETTINGS: dict[str, Any] = {
    "n_envs": B,
    "n_steps": T,
    "num_minibatches": 20,
    "n_epochs": 10,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "max_grad_norm": 10.0,
    "ent_coef": 0.01,
    "vf_coef": 1.0,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "alpha": 0.1,
    "nu": 0.1,
    "normalize_advantage": False,
}
FAULTS: dict[str, dict[str, Any]] = {"gae_lambda": {"gae_lambda": 0.96}}
ACTOR = {"layers_0": "mu.0.model.0", "layers_2": "mu.0.model.2", "mean": "mu.0.model.4"}
CRITIC = {"layers_0": "v.model.0", "layers_2": "v.model.2", "model": "v.model.4"}
LEAF = re.compile(
    r"\.(actor_state|critic_state)\.(params|step|opt_state\[0\]\.(mu|nu|count))"
    r"(?:\['params'\]\['ensemble'\]|\['params'\])?(.*)"
)
SCALARS = {".average_reward": "eta", ".b": "value_bias"}


def reference_names(path: Any) -> tuple[str, str] | None:
    """An Ajax leaf's name in APPO's record and its kind ("kernel", "bias", "log_std", "scalar",
    "count"); None for leaves the record has no counterpart of."""
    k = jax.tree_util.keystr(path)
    if k in SCALARS:
        return SCALARS[k], "scalar"
    m = LEAF.fullmatch(k)
    if m is None:
        return None
    owner, kind, moment, rest = m.groups()
    if kind == "step" or moment == "count":
        return "opt.log_std.step", "count"
    keys = re.findall(r"\['([^']+)'\]", rest)
    tail = {"mu": ".exp_avg", "nu": ".exp_avg_sq", None: ""}[moment]
    if keys == ["log_std_param"]:
        return ("opt.log_std" if moment else "model.log_std") + tail, "log_std"
    module = (ACTOR if owner == "actor_state" else CRITIC)[keys[-2]]
    leaf = "weight" if keys[-1] == "kernel" else "bias"
    return f"{'opt' if moment else 'model'}.{module}.{leaf}{tail}", keys[-1]


def to_ajax(template: Any, arrays: dict, itr: int, dtype: Any) -> Any:
    def put(path: Any, x: Any) -> Any:
        if jax.tree_util.keystr(path) == ".n_updates":
            return jnp.asarray(itr - 1, x.dtype)
        found = reference_names(path)
        if found is None:
            return jnp.asarray(x, dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x
        name, kind = found
        value = np.asarray(arrays[f"in.{name}"], np.float64)
        if kind == "count":
            return jnp.asarray(int(value.reshape(-1)[0]), x.dtype)
        if kind == "scalar":  # NaN before APPO's first call, where Ajax reads n_updates
            value = np.nan_to_num(value)
        value = value.T if kind == "kernel" else value
        return jnp.asarray(value.reshape(x.shape), dtype)

    return jax.tree_util.tree_map_with_path(put, template)


def to_reference(state: Any) -> dict:
    out = {}
    for path, x in jax.tree_util.tree_leaves_with_path(state):
        found = reference_names(path)
        if found is None:
            continue
        name, kind = found
        value = np.asarray(x, np.float64)
        if kind == "kernel":
            value = value.reshape(value.shape[-2:]).T
        elif kind in ("bias", "log_std"):
            value = value.reshape(-1)
        else:
            value = value.reshape(())
        out[name] = value
    return out


def rollout(arrays: dict, dtype: Any) -> Transition:
    """The recorded batch as Ajax's (T, B) rollout, the sampler's Gaussian log-probabilities stored."""
    s = {
        k.removeprefix("samples."): v
        for k, v in arrays.items()
        if k.startswith("samples.")
    }
    done, timeout = s["env.done"].astype(bool), s["env.env_info.timeout"].astype(bool)
    mean, log_std = (s[f"agent.agent_info.dist_info.{k}"] for k in ("mean", "log_std"))
    action = np.asarray(s["agent.action"], np.float64)
    z = (action - mean) / np.exp(log_std)
    log_prob = -0.5 * z**2 - log_std - 0.5 * np.log(2 * np.pi)
    cast = lambda x: jnp.asarray(np.asarray(x, np.float64), dtype)  # noqa: E731
    return Transition(
        obs=cast(s["env.observation"]),
        action=cast(action),
        reward=cast(s["env.reward"][..., None]),
        terminated=cast((done & ~timeout)[..., None]),
        truncated=cast(timeout[..., None]),
        next_obs=cast(arrays["next_obs"]),
        raw_obs=cast(s["env.observation"]),
        log_prob=cast(log_prob),
    )


def ajax_perms(perm: np.ndarray) -> list:
    """rlpyt's flat row b T + t as Ajax's t B + b, epoch by epoch."""
    return [((p % T) * B + p // T).astype(np.int32) for p in perm]


def run(records: Any, precision: str, fault: str | None = None) -> dict[str, dict]:
    """Ajax's state after one update from each record, in rlpyt's names, by record name."""
    dtype = np.float64 if precision == "f64" else np.float32
    envs = sorted({p.name.rsplit("_", 1)[0] for p in records.glob("*.npz")})
    out: dict = {}
    for env_id in envs:
        out.update(_env_updates(records, env_id, dtype, fault))
    return out


def _env_updates(records: Any, env_id: str, dtype: Any, fault: str | None) -> dict:
    settings = dict(SETTINGS)
    settings.update(FAULTS.get(fault or "", {}))
    agent = ajax.APO(env_id, **settings)
    template = kit.capture_step(agent, 8 * T * B).fresh(jax.random.PRNGKey(0))
    _, context = agent.make_context(8 * T * B)

    def update(state: Any, roll: Transition) -> Any:
        return agent.update(state, roll, state, context)[0]

    arrays = {p.stem: dict(np.load(p)) for p in sorted(records.glob(f"{env_id}_*.npz"))}
    keys: dict = {}
    for n, a in arrays.items():
        itr = int(n.rsplit("_", 1)[1])
        with kit.recording() as rec:
            state = to_ajax(template, a, itr, np.float32)
            jax.block_until_ready(jax.jit(update)(state, rollout(a, np.float32)))
        draws = kit.by_role(rec.log)
        assert {r[0] for r in draws} == {"permutation"}, sorted(draws)
        keys[n] = draws
    table: dict = {}

    def with_perms(state: Any, roll: Transition, tape: dict) -> Any:
        table.update(tape)
        return update(state, roll)

    out = {}
    with jax.enable_x64(dtype == np.float64), kit.replaying(table) as replay:
        replayed = jax.jit(with_perms)
        for n, a in arrays.items():
            itr = int(n.rsplit("_", 1)[1])
            [role] = keys[n]
            tape = kit.key_table(keys[n], {role: ajax_perms(a["perm"])})
            new = replayed(to_ajax(template, a, itr, dtype), rollout(a, dtype), tape)
            out[n] = to_reference(new)
    assert not replay.misses
    return out
