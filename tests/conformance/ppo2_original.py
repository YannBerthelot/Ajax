"""Ajax's side of the PPO check against OpenAI Baselines' ppo2 (test_ppo2_original.py): one update
from each of ppo2's recorded states (benchmarks/reference/conformance_ppo2.py), written back in
ppo2's names.

ppo2's variables map one to one onto Ajax's, kernels in the same (in, out) and HWIO layouts:

* ``mlp``: the policy MLP (pi/mlp_fc*) onto the actor's encoder, pi onto its mean head, pi/logstd
  onto the state-independent log-std; the value MLP (vf/mlp_fc*, value_network='copy') onto the
  critic's encoder and vf onto its head. Ajax keeps the two networks apart for MLPs, as ppo2's
  MuJoCo defaults do.
* ``cnn``: nature_cnn (pi/c1-c3, pi/fc1), which both of ppo2's heads read, onto the one encoder
  the critic owns (share_encoder, Ajax's default with images); pi onto the actor's head, vf onto
  the critic's.

ppo2's one Adam maps onto Ajax's two, per coordinate, stepped on one clipped gradient; its step is
its beta1_power's exponent. ppo2 ends an episode at every done, a time limit included, and leaves
the step out of the bootstrap: each done is a termination. Its images (0-255 frames, which
nature_cnn divides by 255) reach Ajax's encoder divided by 255. Each epoch's shuffle is the
record's, from ppo2's rows (b T + t) to Ajax's (t B + b).
"""

from __future__ import annotations

import math
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from conformance_start import start_value
from flax import struct
from gymnax.environments import environment, spaces

import ajax
from ajax.networks.networks import MNIH_CNN
from ajax.state import Transition
from tests.conformance import kit

#: ppo2's settings (ppo2/defaults.py, and learn's: vf_coef 0.5, max_grad_norm 0.5), in Ajax's names;
#: the learning rate is each record's. No normalisation: ppo2's VecNormalize sits in its env, outside
#: the update, and the records feed the update the batch it read.
PPO2 = {
    "normalize_observations": False,
    "normalize_rewards": False,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "adam_eps": 1e-5,
}
MLP = ("64", "tanh", "64", "tanh")  # baselines/common/models.py mlp(): 2 x 64, tanh
CASES: dict[str, dict[str, Any]] = {
    "mlp": {  # mujoco()
        "n_envs": 1,
        "n_steps": 2048,
        "num_minibatches": 32,
        "n_epochs": 10,
        "ent_coef": 0.0,
        "clip_range": 0.2,
        "actor_architecture": MLP,
        "critic_architecture": MLP,
    },
    "cnn": {  # atari(), 2 envs of 64 steps
        "n_envs": 2,
        "n_steps": 64,
        "num_minibatches": 4,
        "n_epochs": 4,
        "ent_coef": 0.01,
        "clip_range": 0.1,
        "cnn_image_shape": (84, 84, 4),
        "cnn_spec": MNIH_CNN,
    },
}
#: Planted faults: one in the update's maths, and each case's network topology flipped (the MLP's
#: encoder shared, the CNN's kept apart).
FAULTS: dict[str, dict[str, Any]] = {
    "gae_lambda": {"gae_lambda": 0.96},
    "shared": {"share_encoder": True},
    "apart": {"share_encoder": False},
}
#: Ajax's module path (below params, past an ensemble) -> ppo2's scope, per case and network.
MODULES = {
    "mlp": {
        "actor_state": {
            "encoder/network/layers_0": "pi/mlp_fc0",
            "encoder/network/layers_2": "pi/mlp_fc1",
            "mean": "pi",
        },
        "critic_state": {
            "encoder/network/layers_0": "vf/mlp_fc0",
            "encoder/network/layers_2": "vf/mlp_fc1",
            "model": "vf",
        },
    },
    "cnn": {
        "actor_state": {"model/layers_0": "pi"},
        "critic_state": {
            "encoder/Conv_0": "pi/c1",
            "encoder/Conv_1": "pi/c2",
            "encoder/Conv_2": "pi/c3",
            "encoder/Dense_0": "pi/fc1",
            "model": "vf",
        },
    },
}
LEAF = re.compile(
    r"\.(actor_state|critic_state)\.(params|step|opt_state\[0\]\.(mu|nu|count))"
    r"(?:\['params'\]\['ensemble'\]|\['params'\])?(.*)"
)
STEP = "adam_step"
SLOT = {"mu": "/Adam", "nu": "/Adam_1", None: ""}


@struct.dataclass
class _FrameState(environment.EnvState):
    pass


class AtariFrames(environment.Environment):
    """Breakout's spaces behind ppo2's wrappers, 84 x 84 x 4 frames (flattened, as Ajax's CNN reads
    them) and 4 actions: what Ajax builds its networks from. The update reads only the recorded
    batch, so the frames themselves are never drawn."""

    @property
    def default_params(self) -> environment.EnvParams:
        return environment.EnvParams()

    def reset_env(self, key: jax.Array, params: Any) -> tuple[jax.Array, _FrameState]:
        return jnp.zeros(84 * 84 * 4), _FrameState(time=0)

    def step_env(
        self, key: jax.Array, state: _FrameState, action: Any, params: Any
    ) -> tuple:
        state = _FrameState(time=state.time + 1)
        return jnp.zeros(84 * 84 * 4), state, jnp.float32(0.0), jnp.bool_(False), {}

    def observation_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(0.0, 1.0, (84 * 84 * 4,), jnp.float32)

    def action_space(self, params: Any = None) -> spaces.Discrete:
        return spaces.Discrete(4)


def reference_name(case: str, path: Any) -> str | None:
    """An Ajax leaf's name in ppo2's record (STEP for Adam's counts); None for leaves the record
    has no counterpart of (the actor's copy of a shared encoder)."""
    m = LEAF.fullmatch(path if isinstance(path, str) else jax.tree_util.keystr(path))
    if m is None:
        return None
    owner, kind, moment, rest = m.groups()
    if kind == "step" or moment == "count":
        return STEP
    keys = re.findall(r"\['([^']+)'\]", rest)
    if keys == ["log_std_param"]:
        return "ppo2_model/pi/logstd" + SLOT[moment]
    scope = MODULES[case][owner].get("/".join(keys[:-1]))
    if scope is None:
        return None
    leaf = "w" if keys[-1] == "kernel" else "b"
    return f"ppo2_model/{scope}/{leaf}{SLOT[moment]}"


def steps_of(arrays: dict, prefix: str) -> int:
    """Adam's steps from its beta2_power, beta2 ** (steps + 1) (beta1_power underflows float32
    within a thousand steps); the start's is beta2."""
    power = float(arrays.get(prefix + "beta2_power", 0.999))
    return round(math.log(power) / math.log(0.999)) - 1


def reference_state(arrays: dict, prefix: str, sampled: bool = True) -> dict:
    """The record's state ``prefix`` (in., f32. or f64.) by ppo2 name, in float64, Adam's steps
    under STEP. A leaf the state before the update leaves out is its start (start_value for a
    variable, zeros for Adam's moments), whole, or at the record's positions when ``sampled``."""
    out: dict[str, Any] = {STEP: np.asarray(steps_of(arrays, prefix))}
    for key in arrays:
        name = key.removeprefix("f32.")
        if not key.startswith("f32.") or name.startswith("beta"):
            continue
        if prefix + name in arrays:
            out[name] = np.asarray(arrays[prefix + name], np.float64)
            continue
        shape = tuple(arrays["shape." + name])
        moment = name.endswith(("/Adam", "/Adam_1"))
        value = np.zeros(shape) if moment else start_value(name, shape)
        value = np.asarray(value, np.float64)
        out[name] = value.reshape(-1)[arrays["at." + name]] if sampled else value
    return out


#: The actor's copy of a shared encoder (LoadedTrainState.shared_params), which the losses do not
#: read (they read the critic's): given the encoder's values all the same.
COPY = re.compile(r"\.actor_state\.shared_params(.*)")


def to_ajax(case: str, template: Any, old: dict, call: int, dtype: Any) -> Any:
    def put(path: Any, x: Any) -> Any:
        key = jax.tree_util.keystr(path)
        if key == ".n_updates":
            return jnp.asarray(call - 1, x.dtype)
        copy = COPY.fullmatch(key)
        if copy is not None:
            name = reference_name(case, f".critic_state.params['params']{copy[1]}")
        else:
            name = reference_name(case, path)
        if name is None:
            return jnp.asarray(x, dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x
        if name == STEP:
            return jnp.asarray(int(old[STEP]), x.dtype)
        return jnp.asarray(np.reshape(old[name], x.shape), dtype)

    return kit.map_with_path(put, template)


def to_reference(case: str, state: Any, arrays: dict) -> dict:
    out = {}
    for path, x in kit.leaves_with_path(state):
        name = reference_name(case, path)
        if name is None:
            continue
        value = np.asarray(x, np.float64)
        if name == STEP:
            out[name] = np.asarray(int(value.reshape(-1)[0]))
        elif "at." + name in arrays:
            out[name] = value.reshape(-1)[arrays["at." + name]]
        else:
            out[name] = value.reshape(np.shape(arrays["f32." + name]))
    return out


def rollout(arrays: dict, case: str, dtype: Any) -> Transition:
    """The recorded batch as Ajax's (T, B) rollout: ppo2's env-major rows b T + t to (t, b), each
    done a termination, the stored log-probabilities ppo2's own."""
    rewards = arrays["env.rewards"]
    T, B = rewards.shape

    def steps(x: np.ndarray) -> np.ndarray:
        return x.reshape(B, T, *x.shape[1:]).swapaxes(0, 1)

    obs = np.concatenate([steps(arrays["batch.obs"]), arrays["env.last_obs"][None]])
    if case == "cnn":  # nature_cnn's own scaling, flattened for Ajax's encoder
        obs = obs.reshape(T + 1, B, -1) / 255.0
    cast = lambda x: jnp.asarray(np.asarray(x, np.float64), dtype)  # noqa: E731
    action = steps(arrays["batch.actions"])
    done = arrays["env.dones"][..., None]
    return Transition(
        obs=cast(obs[:-1]),
        action=jnp.asarray(action) if case == "cnn" else cast(action),
        reward=cast(rewards[..., None]),
        terminated=cast(done),
        truncated=cast(np.zeros_like(done)),
        next_obs=cast(obs[1:]),
        raw_obs=cast(obs[:-1]),
        log_prob=cast(-steps(arrays["batch.neglogpacs"])),
    )


def ajax_perms(perm: np.ndarray, T: int, B: int) -> list:
    """ppo2's row b T + t as Ajax's t B + b, epoch by epoch."""
    return [((p % T) * B + p // T).astype(np.int32) for p in perm]


def agent_for(case: str, lr: float, fault: str | None = None) -> Any:
    settings = {**PPO2, **CASES[case], "actor_learning_rate": lr}
    settings["critic_learning_rate"] = lr
    settings.update(FAULTS.get(fault or "", {}))
    env = "Pendulum-v1" if case == "mlp" else AtariFrames()
    return ajax.PPO(env, **settings)


def run(records: Any, case: str, precision: str, fault: str | None = None) -> dict:
    """Ajax's state after one update from each record of ``case``, in ppo2's names, by record
    name; with ``fault``, a planted fault in Ajax's settings."""
    dtype = np.float64 if precision == "f64" else np.float32
    out = {}
    for path in sorted(records.glob(f"{case}_*.npz")):
        arrays = dict(np.load(path))
        call = int(path.stem.rsplit("_", 1)[1])
        out[path.stem] = _update(case, arrays, call, dtype, fault)
    assert out, f"no record of {case}"
    return out


def _update(case: str, arrays: dict, call: int, dtype: Any, fault: str | None) -> dict:
    agent = agent_for(case, float(arrays["lr"]), fault)
    n = agent.env_args.n_envs * CASES[case]["n_steps"]
    template = kit.capture_step(agent, 8 * n).fresh(jax.random.PRNGKey(0))
    _, context = agent.make_context(8 * n)
    old = reference_state(arrays, "in.", sampled=False)

    def update(state: Any, roll: Transition) -> Any:
        return agent.update(state, roll, state, context)[0]

    with kit.recording() as rec:
        state = to_ajax(case, template, old, call, np.float32)
        jax.block_until_ready(jax.jit(update)(state, rollout(arrays, case, np.float32)))
    draws = kit.by_role(rec.log)
    assert {r[0] for r in draws} == {"permutation"}, sorted(draws)
    [role] = draws
    T, B = arrays["env.rewards"].shape
    tape = kit.key_table(draws, {role: ajax_perms(arrays["perm"], T, B)})
    with jax.enable_x64(dtype == np.float64), kit.replaying(tape) as replay:
        state = to_ajax(case, template, old, call, dtype)
        new = jax.jit(update)(state, rollout(arrays, case, dtype))
    assert not replay.misses
    return to_reference(case, new, arrays)
