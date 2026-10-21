"""BobMak/asac's update, recorded for the conformance check (tests/conformance/test_asac_original.py).

usage: python conformance_asac.py ENV_ID OUT_DIR   (in envs/asac: CPU-only PyTorch 2.8,
stable-baselines3 2.4, gymnasium 1.0, the authors' pins)

Imports nothing from Ajax. ASAC's own agent (vendor/bobmak_asac/darer) learns on gymnasium's ENV_ID by
its own learn() loop with InvertedPendulum-v5's asac.yaml, the closest task the authors tune;
``PendulumEnds`` is Pendulum-v1 with a two-dimensional torque (their mean drives it) that terminates
every 20th step, so that batches hold terminations and nA is 2 (darer/ASAC.py's log(1/nA)). Before
the gradient_descent calls in CALLS, the call and the target update after it run on a copy with their
draws recorded (Normal.rsample: the actor's sample, then the next actions'), and again in float64 with
the same draws. One .npz per call: ``in.*`` (the actor, the two critics and their targets, the three
Adams, theta and the penalty), ``batch.*``, ``noise``, ``max_action``, ``f32.*`` and ``f64.*``.
"""

import copy
import importlib.metadata
import json
import pathlib
import platform
import sys

import gymnasium
import numpy as np
import stable_baselines3.common.policies as sb3_policies
import torch
import yaml

VENDOR = pathlib.Path(__file__).parent / "vendor" / "bobmak_asac"
sys.path.insert(0, str(VENDOR / "darer"))
from ASAC import ASAC  # noqa: E402 (darer's modules import each other by name)

CALLS = (10, 1_000)
SEED = 0
COMMIT = "4b68862c1634c6fcf929064c653efb97d4b72eb5"
HPARAMS = VENDOR / "hparams" / "InvertedPendulum-v5" / "asac.yaml"
#: What gradient_descent reads and writes, copied for a step run aside (the optimizers with the
#: networks whose parameters they hold).
STATE = ("actor", "actor_optimizer", "online_critics", "target_critics", "q_optimizers")
TENSORS = ("theta", "penalty", "baseline", "ent_coef_tensor", "logpi0")


class PendulumEnds(gymnasium.Wrapper):
    """Pendulum-v1 driven by the mean of a two-dimensional torque, terminated every 20th step."""

    def __init__(self):
        super().__init__(gymnasium.make("Pendulum-v1"))
        box = self.env.action_space
        self.action_space = gymnasium.spaces.Box(
            box.low[0], box.high[0], (2,), np.float32
        )
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.t += 1
        torque = np.asarray(action, np.float32).mean(keepdims=True)
        obs, reward, terminated, truncated, info = self.env.step(torque)
        return obs, reward, terminated or self.t % 20 == 0, truncated, info


ENVS = {"PendulumEnds": PendulumEnds}


class Noise:
    """Normal.rsample with its standard-normal draws logged, or given."""

    def __init__(self):
        self.given: list = []
        self.drawn: list = []

    def rsample(self, dist, sample_shape=()):
        shape = dist._extended_shape(torch.Size(sample_shape))
        if self.given:
            eps = torch.as_tensor(self.given.pop(0), dtype=dist.loc.dtype)
        else:
            eps = torch.randn(shape, dtype=dist.loc.dtype)
        self.drawn.append(eps.detach().numpy().copy())
        return dist.loc + eps * dist.scale


NOISE = Noise()


def adam(opt, p, prefix: str, out: dict) -> None:
    st = opt.state.get(p, {})
    zeros = np.zeros(tuple(p.shape), np.float32)
    for k in ("exp_avg", "exp_avg_sq"):
        out[f"{prefix}.{k}"] = st[k].detach().numpy().copy() if st else zeros
    out[f"{prefix}.step"] = (
        st["step"].detach().numpy().copy() if st else np.zeros((), np.float32)
    )


def state(a) -> dict:
    out = {}
    for name, p in a.actor.named_parameters():
        out[f"actor.{name}"] = p.detach().numpy().copy()
        adam(a.actor_optimizer, p, f"opt.actor.{name}", out)
    for i, (q, t, opt) in enumerate(
        zip(
            a.online_critics.nets,
            a.target_critics.nets,
            a.q_optimizers.optimizers,
            strict=True,
        )
    ):
        for (name, p), (_, tp) in zip(
            q.named_parameters(), t.named_parameters(), strict=True
        ):
            out[f"q{i}.{name}"] = p.detach().numpy().copy()
            out[f"q{i}_target.{name}"] = tp.detach().numpy().copy()
            adam(opt, p, f"opt.q{i}.{name}", out)
    out["theta"] = np.asarray(a.theta.detach().numpy(), np.float64)
    out["penalty"] = np.asarray(a.penalty.detach().numpy(), np.float64)
    return out


def aside(agent, batch, noise: list, double: bool) -> dict:
    """gradient_descent then _update_target on a copy, ``noise`` as its draws when given."""
    a = copy.copy(agent)  # shares the logger and the replay
    for k, v in zip(
        STATE, copy.deepcopy(tuple(getattr(agent, k) for k in STATE)), strict=True
    ):
        setattr(a, k, v)
    for k in TENSORS:  # theta carries a graph; its value is what the step reads
        setattr(a, k, getattr(agent, k).detach().clone())
    a.model = a.online_critics
    default = torch.get_default_dtype()
    if double:
        for net in (a.actor, *a.online_critics.nets, *a.target_critics.nets):
            net.double()
        for opt in (a.actor_optimizer, *a.q_optimizers.optimizers):
            for st in opt.state.values():
                for k in ("exp_avg", "exp_avg_sq"):
                    st[k] = st[k].double()
        for k in TENSORS:
            setattr(a, k, getattr(a, k).double())
        batch = tuple(t.double() for t in batch)
        torch.set_default_dtype(torch.float64)  # the baseline's torch.zeros
    # SB3's policies cast a Box observation to float32 (preprocess_obs, its `obs.float()`): for the
    # float64 run the cast keeps the dtype, an identity on the float32 run's inputs.
    preprocess = sb3_policies.preprocess_obs
    if double:
        sb3_policies.preprocess_obs = lambda obs, *_, **__: obs
    NOISE.given, NOISE.drawn = list(noise), []
    try:
        ASAC.gradient_descent(a, batch, 0)
        ASAC._update_target(a)
    finally:
        torch.set_default_dtype(default)
        sb3_policies.preprocess_obs = preprocess
    drawn, NOISE.drawn = NOISE.drawn, []
    return state(a) | {"draws": drawn}


def record(env_id: str, out: pathlib.Path) -> None:
    torch.set_num_threads(1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    params = yaml.safe_load(HPARAMS.read_text())
    env = ENVS[env_id]() if env_id in ENVS else gymnasium.make(env_id)
    # No log folder: ASAC's logger then writes to stdout, without tensorboard.
    agent = ASAC(env, **params, device="cpu", tensorboard_log=None, log_interval=10**9)
    agent.env.reset(seed=SEED)  # learn()'s resets continue this generator
    agent.env.action_space.seed(SEED)
    out.mkdir(parents=True, exist_ok=True)
    max_action = float(np.asarray(env.action_space.high).reshape(-1)[0])
    calls, original = [0], agent.gradient_descent

    def recorded(batch, grad_step):
        calls[0] += 1
        if calls[0] in CALLS:
            f32 = aside(agent, batch, [], double=False)
            noise = f32.pop("draws")
            f64 = aside(agent, batch, noise, double=True)
            f64.pop("draws")
            arrays = {f"in.{k}": v for k, v in state(agent).items()}
            names = ("obs", "action", "next_obs", "done", "reward")
            arrays |= {
                f"batch.{n}": t.numpy() for n, t in zip(names, batch, strict=True)
            }
            arrays |= {"noise": np.stack(noise), "max_action": np.asarray(max_action)}
            arrays |= {f"f32.{k}": v for k, v in f32.items()}
            arrays |= {f"f64.{k}": v for k, v in f64.items()}
            np.savez(out / f"{env_id}_{calls[0]}.npz", **arrays)
        NOISE.drawn = []
        return original(batch, grad_step)

    agent.gradient_descent = recorded
    torch.distributions.Normal.rsample = lambda self, s=(): NOISE.rsample(self, s)
    agent.learn(total_timesteps=max(CALLS) + 1)
    libs = ("torch", "stable-baselines3", "gymnasium", "numpy")
    versions = {lib: importlib.metadata.version(lib) for lib in libs}
    meta = versions | {"python": platform.python_version(), "bobmak_asac": COMMIT}
    (out / "versions.json").write_text(json.dumps(meta))


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
