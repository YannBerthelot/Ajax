"""xtma/apo's update, recorded for the conformance check (tests/conformance/test_apo_original.py).

usage: python conformance_apo.py ENV_ID OUT_DIR   (in envs/torch)

Imports nothing from Ajax. rlpyt's MujocoFfAgent (vendor/xtma_rlpyt) collects B x T steps of a gymnax
task (conformance_torch.OldGym) in rlpyt's sampler layout, as its CpuResetCollector does: an env that
ends resets mid-batch, its next previous action and reward zeroed; a time limit is a done that is not a
termination; the bootstrap value is V of the observation after the last step. APPO (vendor/xtma_apo)
with mujoco_appo.py's algo config and the authors' run grid's discount 1.0 optimises on each batch.
Before the optimize_agent calls in CALLS, the call runs on a copy with its minibatch permutations
recorded, and again in float64 with the same permutations. One .npz per call: ``in.*`` (both MLPs,
log_std, Adam, eta and value_bias, NaN before the first call), ``samples.*``, ``next_obs`` (each env's
own observation after the step, before a reset), ``perm`` (epochs, T B), ``f32.*`` and ``f64.*``.
"""

import copy
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path[:0] = [str(HERE / "vendor" / "xtma_apo"), str(HERE / "vendor" / "xtma_rlpyt")]
import apo.algos.apg.appo as appo_module  # noqa: E402
from apo.algos.apg.appo import APPO  # noqa: E402
from conformance_torch import OldGym, write_versions  # noqa: E402
from rlpyt.agents.pg.mujoco import MujocoFfAgent  # noqa: E402
from rlpyt.samplers.collections import (  # noqa: E402
    AgentSamplesBsv,
    BatchSpec,
    EnvSamples,
    Samples,
)
from rlpyt.utils.buffer import buffer_func  # noqa: E402
from rlpyt.utils.collections import namedarraytuple  # noqa: E402

CALLS = (1, 6)
T, B, SEED = 200, 10, 0
#: mujoco_appo.py's algo config with the run grid's discount (run_exp.py: [1.0]); gae_lambda,
#: lr_eta and rm_vbias_coeff from the grid (Ajax's defaults).
ALGO = {
    "longrun": True,
    "learning_rate": 3e-4,
    "lr_eta": 0.1,
    "rm_vbias_coeff": 0.1,
    "clip_grad_norm": 10.0,
    "entropy_loss_coeff": 0.01,
    "discount": 1.0,
    "gae_lambda": 0.95,
    "minibatches": 20,
    "epochs": 10,
    "ratio_clip": 0.2,
    "normalize_advantage": False,
    "linear_lr_schedule": False,
    "bootstrap_timelimit": True,
}
EnvInfo = namedarraytuple("EnvInfo", ["timeout"])
AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
DistInfo = namedarraytuple("DistInfoStd", ["mean", "log_std"])


class Permutations:
    """iterate_mb_idxs with its shuffles logged, or given."""

    def __init__(self):
        self.given: list = []
        self.drawn: list = []

    def __call__(self, data_length, minibatch_size, shuffle=False):
        indexes = np.arange(data_length)
        if shuffle:
            if self.given:
                indexes = np.asarray(self.given.pop(0))
            else:
                np.random.shuffle(indexes)
        self.drawn.append(indexes.copy())
        for start in range(0, data_length - minibatch_size + 1, minibatch_size):
            yield indexes[start : start + minibatch_size]


PERMS = Permutations()


def collect(agent, envs, carry):
    """One batch of T steps of the B envs, as rlpyt's CpuResetCollector: (samples, the env's own
    observations after each step, before any reset, carry)."""
    obs, prev_action, prev_reward = carry
    rows = []
    for _ in range(T):
        step = agent.step(
            *(torch.as_tensor(x) for x in (obs, prev_action, prev_reward))
        )
        action = step.action.numpy()
        reward, done, timeout = (
            np.zeros(B, np.float32),
            np.zeros(B, bool),
            np.zeros(B, bool),
        )
        next_obs, seen = np.empty_like(obs), np.empty_like(obs)
        for b, env in enumerate(envs):
            seen[b], reward[b], done[b], _ = env.step(action[b])
            timeout[b] = done[b] and not env.terminated
            next_obs[b] = env.reset() if done[b] else seen[b]
        info = step.agent_info
        rows.append(
            (obs, prev_action, prev_reward, action, reward, done, timeout, info, seen)
        )
        next_action = np.where(done[:, None], 0.0, action).astype(np.float32)
        obs, prev_action, prev_reward = (
            next_obs,
            next_action,
            np.where(done, 0.0, reward),
        )
        prev_reward = prev_reward.astype(np.float32)
    o, pa, pr, a, r, d, to, info, seen = zip(*rows, strict=True)
    stack = lambda xs: torch.as_tensor(np.stack(xs))  # noqa: E731
    bootstrap = agent.value(
        *(torch.as_tensor(x) for x in (obs, prev_action, prev_reward))
    )
    dist = DistInfo(
        mean=torch.stack([i.dist_info.mean for i in info]),
        log_std=torch.stack([i.dist_info.log_std for i in info]),
    )
    samples = Samples(
        agent=AgentSamplesBsv(
            action=stack(a),
            prev_action=stack(pa),
            agent_info=AgentInfo(
                dist_info=dist, value=torch.stack([i.value for i in info])
            ),
            bootstrap_value=bootstrap,
        ),
        env=EnvSamples(
            observation=stack(o),
            reward=stack(r),
            prev_reward=stack(pr),
            done=stack(d),
            env_info=EnvInfo(timeout=stack(to)),
        ),
    )
    return samples, np.stack(seen), (obs, prev_action, prev_reward)


def state(algo) -> dict:
    out = {}
    for name, p in algo.agent.model.named_parameters():
        out[f"model.{name}"] = p.detach().double().numpy().copy()
        st = algo.optimizer.state.get(p, {})
        for k in ("exp_avg", "exp_avg_sq"):
            v = st[k].detach().double().numpy() if st else np.zeros(tuple(p.shape))
            out[f"opt.{name}.{k}"] = v.copy()
        out[f"opt.{name}.step"] = np.asarray(float(st["step"]) if st else 0.0)
    for k in ("eta", "value_bias"):
        v = getattr(algo, k)
        out[k] = np.asarray(np.nan if v is None else v, np.float64)
    return out


def aside(algo, itr, samples, perms: list, double: bool) -> dict:
    """optimize_agent on a copy, ``perms`` as its shuffles when given."""
    a = copy.copy(algo)
    a.agent = copy.copy(algo.agent)
    a.agent.model, a.optimizer = copy.deepcopy((algo.agent.model, algo.optimizer))
    if double:
        a.agent.model.double()
        for st in a.optimizer.state.values():
            for k in ("exp_avg", "exp_avg_sq"):
                st[k] = st[k].double()
        samples = buffer_func(
            samples, lambda x: x.double() if torch.is_floating_point(x) else x
        )
    PERMS.given, PERMS.drawn = list(perms), []
    default = torch.get_default_dtype()
    if double:
        torch.set_default_dtype(torch.float64)
    try:
        a.optimize_agent(itr, samples)
    finally:
        torch.set_default_dtype(default)
    drawn, PERMS.drawn = PERMS.drawn, []
    return state(a) | {"perm": np.stack(drawn)}


def flat(samples) -> dict:
    """The samples' leaves by their field path."""
    out = {}

    def walk(x, prefix):
        if hasattr(x, "_fields"):
            for f in x._fields:
                walk(getattr(x, f), f"{prefix}.{f}" if prefix else f)
        else:
            out[prefix] = x.numpy()

    walk(samples, "")
    return out


def record(env_id: str, out: pathlib.Path) -> None:
    torch.set_num_threads(1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    envs = [OldGym(env_id, SEED + b) for b in range(B)]
    act_dim, high = envs[0].shape[0], float(envs[0].high[0])
    spaces = SimpleNamespace(
        observation=SimpleNamespace(shape=(envs[0].obs_dim,)),
        action=SimpleNamespace(
            shape=(act_dim,), high=np.full(act_dim, high), low=np.full(act_dim, -high)
        ),
    )
    agent = MujocoFfAgent(model_kwargs={"normalize_observation": False})
    agent.initialize(spaces)
    algo = APPO(**ALGO)
    algo.initialize(
        agent, n_itr=max(CALLS), batch_spec=BatchSpec(T, B), mid_batch_reset=True
    )
    appo_module.iterate_mb_idxs = PERMS
    out.mkdir(parents=True, exist_ok=True)
    zeros = np.zeros((B, act_dim), np.float32)
    carry = (
        np.stack([env.reset() for env in envs]).astype(np.float32),
        zeros,
        np.zeros(B, np.float32),
    )
    for itr in range(max(CALLS)):
        samples, seen, carry = collect(agent, envs, carry)
        if itr + 1 in CALLS:
            f32 = aside(algo, itr, samples, [], double=False)
            perm = f32.pop("perm")
            f64 = aside(algo, itr, samples, list(perm), double=True)
            f64.pop("perm")
            arrays = {f"in.{k}": v for k, v in state(algo).items()}
            arrays |= {f"samples.{k}": v for k, v in flat(samples).items()}
            arrays |= {"perm": perm, "next_obs": seen}
            arrays |= {f"f32.{k}": v for k, v in f32.items()}
            arrays |= {f"f64.{k}": v for k, v in f64.items()}
            np.savez(out / f"{env_id}_{itr + 1}.npz", **arrays)
        algo.optimize_agent(itr, samples)
        PERMS.drawn = []
    write_versions(out, xtma_apo="bfcd997", xtma_rlpyt="e28bb20")


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
