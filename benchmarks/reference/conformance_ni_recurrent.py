"""Ni et al.'s recurrent SAC and TD3 update, recorded for the conformance check
(tests/conformance/test_ni_recurrent_original.py).

usage: python conformance_ni_recurrent.py ALGO OUT_DIR   (ALGO: sac or td3; in envs/torch)

Imports nothing from Ajax. Builds twni2016/pomdp-baselines' ModelFreeOffPolicy_Separate_RNN
(vendor/twni2016_pomdp_baselines, unchanged) with a GRU and its MuJoCo POMDP configs' settings
(configs/pomdp/*_blt/*/rnn.yml), for gymnax's Pendulum-v1 (3 observations, 1 action). Its windows
are cut from episodes of uniform random actions in [-1, 1] (Ni et al.'s normalised action space):
64 windows of 64 steps, each wholly inside its episode (mask 1), starting mid-episode, as the
update's batch dict holds them. Pendulum never terminates, so the recorder ends every fourth
window's episode at its last step (term 1), as the ASAC recorder's PendulumEnds does: the batch
then covers the done mask. WARMUP updates on windows of their own come first, so that the Adams'
moments, the targets and the temperature are not at their start; the temperature starts at 0.2
(the config's entropy_alpha) rather than SAC's tuned start of 1, where alpha and log alpha have
the same gradient and the check could not tell which one the temperature steps.

The recorded update runs aside, the agent put back after it, its draws logged
(torch.distributions.Normal.sample, which TanhNormal.rsample calls: SAC's next actions, then the
actor's; torch.randn_like: TD3's target noise), and again in float64 with the same draws. It runs
twice more, in either precision, with each GRU given flax's one bias per r and z gate: torch's
nn.GRU has two (b_ih, b_hh), each stepped as far as flax's one, so the original moves its gate
biases twice as far as Ajax's (hold_hidden_gate_biases; tests/conformance/ni_recurrent_original.py).
One .npz per algorithm: ``in.*`` (the actor, the critic and both targets, both Adams, SAC's log
alpha and its Adam), ``batch.*`` (the update's dict, and each window's stored previous action and
reward at its first row, which Ni et al.'s update replaces by zeros), ``noise``, ``f32.*`` and
``f64.*``, ``one_gate_bias.f32.*`` and ``one_gate_bias.f64.*``.
"""

import codecs
import copy
import math
import pathlib
import sys
import types

import gymnasium
import numpy as np
import torch
from conformance_torch import OldGym, write_versions

VENDOR = pathlib.Path(__file__).parent / "vendor" / "twni2016_pomdp_baselines"
sys.path.insert(0, str(VENDOR))
# utils/helpers.py imports gym.spaces for helpers the update never calls: gymnasium's spaces stand
# in. utils/logger.py, which policy_rnn.py imports and never calls in the update, is not vendored:
# it imports collections.Set, removed in Python 3.10, and tensorboard.
sys.modules.setdefault("gym", gymnasium)
sys.modules.setdefault("gym.spaces", gymnasium.spaces)
sys.modules["utils.logger"] = types.ModuleType("utils.logger")
# torchkit/pytorch_utils.py declares the future-fstrings codec, a backport of f-strings to Python
# before 3.6 whose package decodes as UTF-8 on later Pythons: registered here as UTF-8.
codecs.register(
    lambda name: codecs.lookup("utf-8") if name == "future_fstrings" else None
)
import torchkit.pytorch_utils as ptu  # noqa: E402 (after the stand-ins above)
from policies.models.policy_rnn import ModelFreeOffPolicy_Separate_RNN  # noqa: E402

COMMIT = "e7c19c32a20033d75414b29fbc466c77c211e968"
SEED = 0
ENV_ID = "Pendulum-v1"
#: rnn.yml's policy section (configs/pomdp/hopper_blt/p/rnn.yml:29-54, as every MuJoCo POMDP config).
POLICY = {
    "action_embedding_size": 16,
    "observ_embedding_size": 32,
    "reward_embedding_size": 16,
    "rnn_hidden_size": 128,
    "dqn_layers": [256, 256],
    "policy_layers": [256, 256],
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "sac": {"entropy_alpha": 0.2, "automatic_entropy_tuning": True, "alpha_lr": 3e-4},
    "td3": {"exploration_noise": 0.1, "target_noise": 0.2, "target_noise_clip": 0.5},
}
WINDOWS, STEPS = 64, 64  # batch_size, sampled_seq_len (rnn.yml:19-20)
EPISODES, EPISODE_STEPS = 8, 200  # Pendulum-v1's time limit
WARMUP = 3
NETWORKS = ("actor", "actor_target", "critic", "critic_target")
OPTIMIZERS = (("actor", "actor_optimizer"), ("critic", "critic_optimizer"))


class Noise:
    """The update's standard-normal draws, logged, or given."""

    def __init__(self):
        self.given: list = []
        self.drawn: list = []

    def draw(self, shape, dtype) -> torch.Tensor:
        if self.given:
            eps = torch.as_tensor(self.given.pop(0), dtype=dtype)
        else:
            eps = torch.randn(shape, dtype=dtype)
        self.drawn.append(eps.numpy().copy())
        return eps


NOISE = Noise()


def normal_sample(self, sample_shape=()):
    with torch.no_grad():
        eps = NOISE.draw(self._extended_shape(sample_shape), self.loc.dtype)
        return self.loc + eps * self.scale


torch.distributions.Normal.sample = normal_sample
torch.randn_like = lambda tensor, **_: NOISE.draw(tensor.shape, tensor.dtype)


def rollouts(env: OldGym, rng: np.random.Generator) -> dict:
    """EPISODES episodes of uniform random actions in [-1, 1], (episode, step, dim)."""
    obs, act, rew = [], [], []
    for _ in range(EPISODES):
        o = [env.reset()]
        a = rng.uniform(-1.0, 1.0, (EPISODE_STEPS, 1)).astype(np.float32)
        r = []
        for action in a:
            next_obs, reward, _, _ = env.step(env.high * action)
            o.append(next_obs)
            r.append([reward])
        obs.append(o)
        act.append(a)
        rew.append(r)
    return {
        "obs": np.asarray(obs, np.float32),  # (episode, step + 1, 3)
        "act": np.asarray(act, np.float32),
        "rew": np.asarray(rew, np.float32),
    }


def windows(data: dict, rng: np.random.Generator) -> dict:
    """The update's batch dict, (T, B, dim) as numpy, and each window's stored previous action and
    reward at its first row, (B, dim)."""
    episode = rng.integers(0, EPISODES, WINDOWS)
    start = rng.integers(1, EPISODE_STEPS - STEPS + 1, WINDOWS)
    steps = start[:, None] + np.arange(STEPS)  # (B, T)

    def take(name: str, offset: int = 0) -> np.ndarray:
        return np.swapaxes(data[name][episode[:, None], steps + offset], 0, 1)

    term = np.zeros((STEPS, WINDOWS, 1), np.float32)
    term[-1, 3::4] = 1.0
    return {
        "obs": take("obs"),
        "act": take("act"),
        "rew": take("rew"),
        "term": term,
        "obs2": take("obs", 1),
        "mask": np.ones((STEPS, WINDOWS, 1), np.float32),
        "prev_act": data["act"][episode, start - 1],
        "prev_rew": data["rew"][episode, start - 1],
    }


def adam(optimizer, parameter, prefix: str, out: dict) -> None:
    state = optimizer.state.get(parameter, {})
    zeros = np.zeros(tuple(parameter.shape), np.float32)
    for k in ("exp_avg", "exp_avg_sq"):
        out[f"{prefix}.{k}"] = state[k].detach().numpy().copy() if state else zeros
    out[f"{prefix}.step"] = (
        state["step"].detach().numpy().copy() if state else np.zeros((), np.float32)
    )


def state(agent) -> dict:
    out = {}
    for net in NETWORKS:
        for name, p in getattr(agent, net).named_parameters():
            out[f"{net}.{name}"] = p.detach().numpy().copy()
    for net, optimizer in OPTIMIZERS:
        for name, p in getattr(agent, net).named_parameters():
            adam(getattr(agent, optimizer), p, f"opt.{net}.{name}", out)
    algo = agent.algo
    if hasattr(algo, "log_alpha_entropy"):
        out["log_alpha"] = algo.log_alpha_entropy.detach().numpy().copy()
        adam(algo.alpha_entropy_optim, algo.log_alpha_entropy, "opt.alpha", out)
    return out


def as_tensors(batch: dict, dtype) -> dict:
    names = ("obs", "act", "rew", "term", "obs2", "mask")
    return {n: torch.as_tensor(batch[n], dtype=dtype) for n in names}


def optimizers(agent) -> list:
    algo = agent.algo
    alpha = [algo.alpha_entropy_optim] if hasattr(algo, "log_alpha_entropy") else []
    return [agent.actor_optimizer, agent.critic_optimizer, *alpha]


def snapshot(agent) -> dict:
    """What an update changes, to put back after a step run aside. Not a deepcopy of the agent:
    torchkit's PyTorchModule (the Mlps) copies by pickling, which rebuilds each Mlp with new
    parameters (core.py __setstate__), so a copied optimizer would step parameters the copied
    networks no longer hold."""
    algo = agent.algo
    return {
        "networks": {
            n: copy.deepcopy(getattr(agent, n).state_dict()) for n in NETWORKS
        },
        "optimizers": [copy.deepcopy(o.state_dict()) for o in optimizers(agent)],
        "alpha": (
            (algo.log_alpha_entropy.detach().clone(), algo.alpha_entropy)
            if hasattr(algo, "log_alpha_entropy")
            else None
        ),
    }


def restore(agent, saved: dict) -> None:
    agent.float()
    for name, values in saved["networks"].items():
        getattr(agent, name).load_state_dict(values)
    if saved["alpha"] is not None:
        log_alpha, agent.algo.alpha_entropy = saved["alpha"]
        agent.algo.log_alpha_entropy.data = log_alpha.clone()
    for optimizer, values in zip(optimizers(agent), saved["optimizers"], strict=True):
        optimizer.load_state_dict(values)  # its moments cast to the parameters' dtype


def hold_hidden_gate_biases(agent) -> list:
    """Give each GRU flax's one bias per r and z gate for the update: b_hh's r and z rows held
    where they are, their gradient and Adam moments zeroed, so that b_ih's rows alone step (Adam
    moves nothing whose moments and gradient are zero). The hooks' handles, to remove."""
    rows = 2 * POLICY["rnn_hidden_size"]  # r, then z
    handles = []
    for net, optimizer in OPTIMIZERS:
        bias = getattr(agent, net).rnn.bias_hh_l0
        for k in ("exp_avg", "exp_avg_sq"):
            getattr(agent, optimizer).state[bias][k][:rows] = 0.0
        keep = torch.ones_like(bias)
        keep[:rows] = 0.0
        handles.append(bias.register_hook(lambda grad, keep=keep: grad * keep))
    return handles


def aside(
    agent, batch: dict, noise: list, double: bool, one_gate_bias: bool = False
) -> tuple[dict, list]:
    """agent.update(batch), ``noise`` as its draws when given, then the agent put back; with
    one_gate_bias, its GRUs given flax's one bias per r and z gate (hold_hidden_gate_biases)."""
    saved = snapshot(agent)
    default = torch.get_default_dtype()
    handles = []
    if double:
        agent.double()  # in place (param.data), so the optimizers keep their parameters
        algo = agent.algo
        if hasattr(algo, "log_alpha_entropy"):
            algo.log_alpha_entropy.data = algo.log_alpha_entropy.data.double()
            algo.alpha_entropy = algo.log_alpha_entropy.exp().item()
        for optimizer in optimizers(agent):
            for st in optimizer.state.values():
                for k in ("exp_avg", "exp_avg_sq"):
                    st[k] = st[k].double()
        torch.set_default_dtype(torch.float64)  # ptu.zeros, TanhNormal's Normal(0, 1)
    if one_gate_bias:
        handles = hold_hidden_gate_biases(agent)
    NOISE.given, NOISE.drawn = list(noise), []
    try:
        agent.update(as_tensors(batch, torch.float64 if double else torch.float32))
        after = state(agent)
    finally:
        torch.set_default_dtype(default)
        for handle in handles:
            handle.remove()
        restore(agent, saved)
    drawn, NOISE.drawn = NOISE.drawn, []
    return after, drawn


def record(algo: str, out: pathlib.Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    ptu.set_gpu_mode(False)
    env = OldGym(ENV_ID, SEED)
    data = rollouts(env, rng)
    agent = ModelFreeOffPolicy_Separate_RNN(
        obs_dim=env.obs_dim,
        action_dim=env.shape[0],
        encoder="gru",
        algo_name=algo,
        **POLICY,
    )
    if algo == "sac":
        start = POLICY["sac"]["entropy_alpha"]
        agent.algo.log_alpha_entropy.data.fill_(math.log(start))
        agent.algo.alpha_entropy = agent.algo.log_alpha_entropy.exp().item()
    for _ in range(WARMUP):
        agent.update(as_tensors(windows(data, rng), torch.float32))
    NOISE.drawn = []
    batch = windows(data, rng)
    f32, noise = aside(agent, batch, [], double=False)
    arrays = {f"in.{k}": v for k, v in state(agent).items()}
    arrays.update({f"batch.{k}": v for k, v in batch.items()})
    arrays["noise"] = np.stack(noise)
    for tag, double, one_gate_bias in (
        ("f32", False, False),
        ("f64", True, False),
        ("one_gate_bias.f32", False, True),
        ("one_gate_bias.f64", True, True),
    ):
        after = (
            f32
            if tag == "f32"
            else aside(agent, batch, noise, double, one_gate_bias)[0]
        )
        arrays.update({f"{tag}.{k}": v for k, v in after.items()})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{algo}_{WARMUP + 1}.npz", **arrays)
    write_versions(out, pomdp_baselines=COMMIT)


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
