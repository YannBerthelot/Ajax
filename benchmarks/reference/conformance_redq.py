"""watchernyu/REDQ's update, recorded for the conformance check (tests/conformance/test_redq_original.py).

usage: python conformance_redq.py ENV_ID OUT_DIR   (in envs/torch: CPU-only PyTorch, gymnax, jax)

Imports nothing from Ajax. REDQ's own REDQSACAgent (vendor/watchernyu_redq) trains on a gymnax task by
experiments/train_redq_sac.py's loop, written out below: uniform random actions for START steps, then
the policy's sampled action, done False at a time limit, one train() per step. Before the train() calls
in CALLS (counted among those that update), the call is run on a copy with its draws recorded (each of
its 20 updates' batch indices and 2-of-10 subset, and every noise of the policy's rsample: 20 next
actions, then the actor's), and again in float64 with the same draws. One .npz per call: ``in.*`` (with
``alpha_used``, the temperature REDQ reads), ``batch.*`` (the 20 batches, stacked), ``subset``,
``noise``, ``max_action``, ``f32.*`` and ``f64.*``.
"""

import copy
import pathlib
import sys
import types

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "vendor" / "watchernyu_redq"))
from conformance_torch import OldGym, write_versions
from redq.algos.redq_sac import REDQSACAgent

CALLS = (1, 25)
START, SEED = 1_000, 0
COMMIT = "7b5d1bff39291a57325a2836bd397a55728960bb"


class Quiet:
    def store(self, **_):
        pass


def state(agent) -> dict:
    """The update's state: the policy, the 10 critics and their targets, their Adams, log_alpha and
    its Adam (float32 as trained; the float64 run's output as float64)."""
    nets = {"policy": agent.policy_net}
    nets |= {f"q{i}": q for i, q in enumerate(agent.q_net_list)}
    nets |= {f"q{i}_target": q for i, q in enumerate(agent.q_target_net_list)}
    out = {
        f"{prefix}.{name}": p.detach().numpy().copy()
        for prefix, net in nets.items()
        for name, p in net.named_parameters()
    }
    opts = {"policy": (agent.policy_net, agent.policy_optimizer)}
    opts |= {
        f"q{i}": pair
        for i, pair in enumerate(
            zip(agent.q_net_list, agent.q_optimizer_list, strict=False)
        )
    }
    pairs = [
        (f"opt.{prefix}.{name}", p, opt)
        for prefix, (net, opt) in opts.items()
        for name, p in net.named_parameters()
    ]
    out["log_alpha"] = agent.log_alpha.detach().numpy().copy()
    # The temperature its targets and actor loss read: exp(log_alpha) once it has stepped, the alpha
    # argument (0.2) before, which its __init__ sets last (redq_sac.py:80).
    out["alpha_used"] = np.asarray(agent.alpha, np.float64)
    pairs.append(("opt.alpha", agent.log_alpha, agent.alpha_optim))
    for prefix, p, opt in pairs:
        # An Adam that has not stepped holds no state: its moments are zeros, its step 0.
        st = opt.state.get(p, {})
        zeros = np.zeros(tuple(p.shape), p.detach().numpy().dtype)
        out[f"{prefix}.exp_avg"] = (
            st["exp_avg"].detach().numpy().copy() if st else zeros
        )
        out[f"{prefix}.exp_avg_sq"] = (
            st["exp_avg_sq"].detach().numpy().copy() if st else zeros
        )
        out[f"{prefix}.step"] = (
            st["step"].detach().numpy().copy() if st else np.zeros((), np.float32)
        )
    return out


def one_call(agent, draws: dict, double: bool) -> dict:
    """``agent.train`` once on a copy: recording its draws into the empty lists of ``draws``, or
    replaying the filled ones (cast to float64 with ``double``)."""
    a = copy.deepcopy(agent)
    replay = bool(draws["idxs"])
    if double:
        for net in (a.policy_net, *a.q_net_list, *a.q_target_net_list):
            net.double()
        with torch.no_grad():
            a.log_alpha.data = a.log_alpha.data.double()
        for opt in (a.policy_optimizer, *a.q_optimizer_list, a.alpha_optim):
            for st in opt.state.values():
                for k in ("exp_avg", "exp_avg_sq"):
                    st[k] = st[k].double()
    counts = {"idxs": 0, "subset": 0, "noise": 0}

    def taken(name: str, fresh):
        if not replay:
            draws[name].append(fresh())
            return draws[name][-1]
        counts[name] += 1
        return draws[name][counts[name] - 1]

    sample_batch, choice = a.replay_buffer.sample_batch, np.random.choice

    def sampled(batch_size=32, idxs=None):  # REDQ's own call passes no idxs
        del idxs
        size = a.replay_buffer.size
        return sample_batch(
            batch_size, taken("idxs", lambda: np.random.randint(0, size, batch_size))
        )

    a.replay_buffer.sample_batch = sampled
    sample_data = a.sample_data
    if double:
        a.sample_data = lambda n: tuple(t.double() for t in sample_data(n))
    rsample = torch.distributions.Normal.rsample

    def noisy(self, sample_shape=()):
        shape = self._extended_shape(torch.Size(sample_shape))
        eps = taken("noise", lambda: torch.randn(shape).numpy())
        return self.loc + torch.as_tensor(eps, dtype=self.loc.dtype) * self.scale

    np.random.choice = lambda *args, **kw: taken("subset", lambda: choice(*args, **kw))
    torch.distributions.Normal.rsample = noisy
    try:
        a.train(Quiet())
    finally:
        np.random.choice = choice
        torch.distributions.Normal.rsample = rsample
    return state(a)


def record(env_id: str, out: pathlib.Path) -> None:
    torch.set_num_threads(1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = OldGym(env_id, SEED)
    env.action_space = types.SimpleNamespace(
        sample=env.sample
    )  # get_exploration_action's
    max_action = float(env.high[0])
    agent = REDQSACAgent(
        env_id,
        env.obs_dim,
        env.shape[0],
        max_action,
        torch.device("cpu"),
        target_entropy="auto",
        start_steps=START,
    )  # train_redq_sac.py's defaults, but the two named
    max_ep_len = min(1000, env._max_episode_steps)
    out.mkdir(parents=True, exist_ok=True)
    o, ep_len, calls = env.reset(), 0, 0
    while calls < max(CALLS):
        a = agent.get_exploration_action(o, env)
        o2, r, d, _ = env.step(a)
        ep_len += 1
        d = False if ep_len == max_ep_len else d
        agent.store_data(o, a, r, o2, d)
        if agent.replay_buffer.size > agent.delay_update_steps:  # train() updates
            calls += 1
            if calls in CALLS:
                draws: dict = {"idxs": [], "subset": [], "noise": []}
                f32 = one_call(agent, draws, double=False)
                f64 = one_call(agent, draws, double=True)
                rb, idxs = agent.replay_buffer, np.stack(draws["idxs"])
                arrays = {f"in.{k}": v for k, v in state(agent).items()}
                arrays |= {
                    "batch.obs": rb.obs1_buf[idxs],
                    "batch.next_obs": rb.obs2_buf[idxs],
                    "batch.action": rb.acts_buf[idxs],
                    "batch.reward": rb.rews_buf[idxs],
                    "batch.done": rb.done_buf[idxs],
                    "subset": np.stack(draws["subset"]),
                    "noise": np.stack(draws["noise"]),
                    "max_action": np.asarray(max_action),
                }
                arrays |= {f"f32.{k}": v for k, v in f32.items()}
                arrays |= {f"f64.{k}": v for k, v in f64.items()}
                np.savez(out / f"{env_id}_{calls}.npz", **arrays)
        agent.train(Quiet())
        o = o2
        if d or ep_len == max_ep_len:
            o, ep_len = env.reset(), 0
    write_versions(out, watchernyu_redq=COMMIT)


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
