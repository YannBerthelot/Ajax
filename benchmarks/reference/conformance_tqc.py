"""bayesgroup/tqc_pytorch's update, recorded for the conformance check (tests/conformance/test_tqc_original.py).

usage: python conformance_tqc.py ENV_ID OUT_DIR   (in envs/torch14: torch 1.4 on x86 Python 3.8)

Imports nothing from Ajax. The authors' repository is no longer on GitHub: vendor/bayesgroup_tqc_pytorch
is Software Heritage's archive of it (SOURCE below), unmodified. Their Actor, Critic and Trainer train
on a gymnasium task by main.py's loop, written out below: actions in [-1, 1] mapped onto the env's box
(their RescaleAction), the policy's sampled action from the first step, done at a termination only (a
time limit resets without one), one train() per step from t >= batch_size. Before the train() calls in
CALLS (counted from the first), the call runs on a copy with its draws recorded (the batch indices, and
every noise of the policy's rsample: the next actions', then the actor's), and again in float64 with
the same draws. One .npz per call: ``in.*``, ``batch.*``, ``noise``, ``f32.*`` and ``f64.*``.

Their Trainer.train runs unchanged. It backpropagates the actor's loss after the critics' optimizer
step; their torch (1.3.0) stepped parameters through ``.data``, which autograd does not track, so that
backward pass reads the stepped critics' weights, and torch >= 1.5 refuses it. torch 1.4 steps the same
way and is the oldest with a wheel for a Python uv installs, so this runs there; CI cannot (no arm64
wheels), and the records are committed (tests/conformance/records). One recorded change: the networks
are WIDTH units wide (theirs: critics 3 x 512, actor 2 x 256), their own Mlp class built narrower, as
the maths does not depend on the width and the records are kept small.
"""

import copy
import pathlib
import sys
import types

import numpy as np
import torch

# structures.py imports gym for its RescaleAction, which this loop writes out: an empty stand-in.
gym = types.ModuleType("gym")
gym.ActionWrapper = object
gym.spaces = types.SimpleNamespace(Box=object)
sys.modules.setdefault("gym", gym)
sys.modules.setdefault("gym.spaces", gym.spaces)
sys.path.insert(
    0, str(pathlib.Path(__file__).parent / "vendor" / "bayesgroup_tqc_pytorch")
)
from conformance_old_torch import OldGym, write_versions  # noqa: E402
from tqc import structures  # noqa: E402
from tqc.trainer import Trainer  # noqa: E402

CALLS = (1, 25)
SEED, WIDTH, BATCH = 0, 16, 64
N_QUANTILES, N_NETS, TOP_DROP_PER_NET = 25, 5, 2  # main.py's defaults
SOURCE = "swh:1:rev:65114eba5d868e9f0355e33b96acab7530191f16"


class Critic(structures.Critic):
    """Their Critic, its nets WIDTH wide."""

    def __init__(self, state_dim, action_dim, n_quantiles, n_nets):
        torch.nn.Module.__init__(self)
        self.nets, self.n_quantiles, self.n_nets = [], n_quantiles, n_nets
        for i in range(n_nets):
            net = structures.Mlp(state_dim + action_dim, [WIDTH] * 3, n_quantiles)
            self.add_module(f"qf{i}", net)
            self.nets.append(net)


class Actor(structures.Actor):
    """Their Actor, WIDTH wide."""

    def __init__(self, state_dim, action_dim):
        torch.nn.Module.__init__(self)
        self.action_dim = action_dim
        self.net = structures.Mlp(state_dim, [WIDTH] * 2, 2 * action_dim)


def named(trainer) -> dict:
    """Every trained tensor by name: the actor's last layer split into its mean and log-std rows."""
    a = trainer.actor.action_dim
    out = {}
    for name, p in trainer.actor.net.named_parameters():
        if name.startswith("last_fc."):
            leaf = name.split(".")[1]
            out[f"actor.mean.{leaf}"] = (p, slice(0, a))
            out[f"actor.log_std.{leaf}"] = (p, slice(a, 2 * a))
        else:
            out[f"actor.{name}"] = (p, slice(None))
    for prefix, net in (("critic", trainer.critic), ("target", trainer.critic_target)):
        out.update(
            {f"{prefix}.{n}": (p, slice(None)) for n, p in net.named_parameters()}
        )
    out["log_alpha"] = (trainer.log_alpha, slice(None))
    return out


def state(trainer) -> dict:
    """The update's state: the actor, the critics and their targets, their Adams, log_alpha and its
    Adam (float32 as trained; the float64 run's output as float64)."""
    opts = {"actor": trainer.actor_optimizer, "critic": trainer.critic_optimizer}
    opts["log_alpha"] = trainer.alpha_optimizer
    out = {}
    for name, (p, rows) in named(trainer).items():
        out[name] = p.detach().numpy()[rows].copy()
        owner = name.split(".")[0]
        if owner == "target":
            continue
        st = opts[owner].state.get(p, {})
        zeros = np.zeros(tuple(p.shape), p.detach().numpy().dtype)
        for moment in ("exp_avg", "exp_avg_sq"):
            value = st[moment].detach().numpy() if st else zeros
            out[f"opt.{name}.{moment}"] = value[rows].copy()
        out[f"opt.{name}.step"] = (
            np.asarray(float(st["step"]), np.float32)  # an int on torch 1.4
            if st
            else np.zeros((), np.float32)
        )
    return out


def one_call(trainer, rb, draws: dict, double: bool) -> dict:
    """``trainer.train`` once on a copy: recording its draws into the empty lists of ``draws``, or
    replaying the filled ones (in float64 with ``double``)."""
    t = copy.deepcopy(trainer)
    replay = bool(draws["idxs"])
    dtype = torch.float64 if double else torch.float32
    if double:
        for net in (t.actor, t.critic, t.critic_target):
            net.double()
        with torch.no_grad():
            t.log_alpha.data = t.log_alpha.data.double()
        for opt in (t.actor_optimizer, t.critic_optimizer, t.alpha_optimizer):
            for st in opt.state.values():
                for k in ("exp_avg", "exp_avg_sq"):
                    st[k] = st[k].double()
    counts = {"idxs": 0, "noise": 0}

    def taken(name: str, fresh):
        if not replay:
            draws[name].append(fresh())
            return draws[name][-1]
        counts[name] += 1
        return draws[name][counts[name] - 1]

    def sample(
        batch_size,
    ):  # ReplayBuffer.sample (structures.py:84-87), its indices taped
        ind = taken("idxs", lambda: np.random.randint(0, rb.size, size=batch_size))
        names = rb.transition_names
        return (torch.as_tensor(getattr(rb, n)[ind], dtype=dtype) for n in names)

    def rsample(self):  # TanhNormal.rsample (structures.py:147-149), its noise taped
        eps = taken("noise", lambda: self.standard_normal.sample().numpy())
        pretanh = self.normal_mean + self.normal_std * torch.as_tensor(eps, dtype=dtype)
        return torch.tanh(pretanh), pretanh

    original = structures.TanhNormal.rsample
    structures.TanhNormal.rsample = rsample
    try:
        t.train(types.SimpleNamespace(sample=sample), BATCH)
    finally:
        structures.TanhNormal.rsample = original
    return state(t)


def record(env_id: str, out: pathlib.Path) -> None:
    torch.set_num_threads(1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = OldGym(env_id, SEED)
    state_dim, action_dim = env.obs_dim, env.shape[0]
    rb = structures.ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim, N_QUANTILES, N_NETS)
    trainer = Trainer(
        actor=actor,
        critic=critic,
        critic_target=copy.deepcopy(critic),
        top_quantiles_to_drop=TOP_DROP_PER_NET * N_NETS,
        discount=0.99,
        tau=0.005,
        target_entropy=-float(action_dim),
    )
    out.mkdir(parents=True, exist_ok=True)
    o, calls = env.reset(), 0
    actor.train()
    for t in range(10**9):
        a = actor.select_action(o)
        # RescaleAction's map of [-1, 1] onto the env's box.
        a_env = np.clip(
            env.low + (env.high - env.low) * ((a + 1.0) / 2.0), env.low, env.high
        )
        o2, r, d, _ = env.step(a_env)
        rb.add(o, a, o2, r, float(env.terminated))
        o = o2
        if t >= BATCH:
            calls += 1
            if calls in CALLS:
                draws: dict = {"idxs": [], "noise": []}
                f32 = one_call(trainer, rb, draws, double=False)
                f64 = one_call(trainer, rb, draws, double=True)
                [idxs] = draws["idxs"]
                arrays = {f"in.{k}": v for k, v in state(trainer).items()}
                arrays.update(
                    {
                        "batch.obs": rb.state[idxs],
                        "batch.action": rb.action[idxs],
                        "batch.next_obs": rb.next_state[idxs],
                        "batch.reward": rb.reward[idxs],
                        "batch.not_done": rb.not_done[idxs],
                        "noise": np.stack(draws["noise"]),
                    }
                )
                arrays.update({f"f32.{k}": v for k, v in f32.items()})
                arrays.update({f"f64.{k}": v for k, v in f64.items()})
                # Committed records: compressed (a fifth smaller here).
                np.savez_compressed(out / f"{env_id}_{calls}.npz", **arrays)
            if calls >= max(CALLS):
                break
            trainer.train(rb, BATCH)
        if d:
            o = env.reset()
    write_versions(out, bayesgroup_tqc_pytorch=SOURCE)


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
