"""gauthamvasan/avg's update, recorded for the conformance check (tests/conformance/test_avg_original.py).

usage: python conformance_avg.py ENV_ID OUT_DIR   (in envs/torch: CPU-only PyTorch, gymnasium, gymnax)

Imports nothing from Ajax. AVG's own avg.py (vendor/gauthamvasan_avg) learns on a gymnax task by
avg.py's main() loop, written out below, under its gymnasium wrappers (FlattenObservation,
NormalizeObservation, ClipAction); avg.py's video helper, unused here, is stubbed for its cv2 import.
Before the updates of the steps in STEPS, the step's update is run on a copy with its draws given (the
action's noise, which compute_action draws, then the next action's), as is and in float64. One .npz per
step: ``in.*`` (the actor, the critic, both Adams, the TD-error scaler's statistics and G), the
transition as the update reads it (``obs``, ``action``, ``action_pre``, ``next_obs``, ``reward``,
``terminated``), ``noise.action``, ``noise.next``, ``f32.*`` and ``f64.*``.
"""

import copy
import pathlib
import sys
import types

import gymnasium
import gymnax
import jax
import numpy as np
import torch
from torch.distributions import MultivariateNormal

sys.path.insert(0, str(pathlib.Path(__file__).parent / "vendor" / "gauthamvasan_avg"))
sys.modules["incremental_rl.experiment_tracker"] = types.SimpleNamespace(
    record_video=None
)
import avg  # noqa: E402 (after the stub above)
from conformance_torch import write_versions  # noqa: E402

STEPS = (10, 1_000)
SEED = 0
COMMIT = "6b6822cf2ae63c49a8344738e41ffc37c7998355"
#: avg.py main()'s defaults.
DEFAULTS = {
    "actor_lr": 0.0063,
    "critic_lr": 0.0087,
    "betas": [0.0, 0.999],
    "gamma": 0.99,
    "alpha_lr": 0.07,
    "nhid_actor": 256,
    "nhid_critic": 256,
    "device": "cpu",
}
SCALER = ("reward_rms", "gamma_rms", "return_sq_rms", "return_rms")


class GymnaxEnv(gymnasium.Env):
    """A gymnax task as a gymnasium.Env, for avg.py's wrappers."""

    def __init__(self, env_id: str, seed: int):
        self.env, self.params = gymnax.make(env_id)
        space = self.env.action_space(self.params)
        low = np.broadcast_to(np.asarray(space.low, np.float32), space.shape)
        high = np.broadcast_to(np.asarray(space.high, np.float32), space.shape)
        self.action_space = gymnasium.spaces.Box(low, high, dtype=np.float32)
        dim = self.env.observation_space(self.params).shape[0]
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, (dim,), np.float32
        )
        self.key = jax.random.PRNGKey(seed)
        self._step = jax.jit(lambda k, s, a: self.env.step(k, s, a, self.params))
        self._reset = jax.jit(lambda k: self.env.reset(k, self.params))

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        self.key, k = jax.random.split(self.key)
        obs, self.state = self._reset(k)
        return np.asarray(obs), {}

    def step(self, action):
        self.key, k = jax.random.split(self.key)
        obs, self.state, reward, term, trunc, _ = self._step(
            k, self.state, np.asarray(action, np.float32)
        )
        return np.asarray(obs), float(reward), bool(term), bool(trunc), {}


class Noise:
    """MultivariateNormal.rsample with its standard-normal draws logged, or given."""

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
        tril = dist._unbroadcasted_scale_tril
        return dist.loc + torch.matmul(tril, eps.unsqueeze(-1)).squeeze(-1)


NOISE = Noise()
MultivariateNormal.rsample = lambda self, sample_shape=(): NOISE.rsample(
    self, sample_shape
)


def state(agent) -> dict:
    """The update's state: the actor, the critic, both Adams (zeros before a step), the TD-error
    scaler's four running statistics (n, m, s) and the return accumulator G."""
    out = {}
    for prefix, net, opt in (
        ("actor", agent.actor, agent.popt),
        ("Q", agent.Q, agent.qopt),
    ):
        for name, p in net.named_parameters():
            out[f"{prefix}.{name}"] = p.detach().numpy().copy()
            st = opt.state.get(p, {})
            zeros = np.zeros(tuple(p.shape), p.detach().numpy().dtype)
            for k in ("exp_avg", "exp_avg_sq"):
                out[f"opt.{prefix}.{name}.{k}"] = (
                    st[k].detach().numpy().copy() if st else zeros
                )
            step = (
                st["step"].detach().numpy().copy() if st else np.zeros((), np.float32)
            )
            out[f"opt.{prefix}.{name}.step"] = step
    for name in SCALER:
        rs = getattr(agent.td_error_scaler, name)
        out[f"scaler.{name}.n"] = np.asarray(rs.n, np.float64)
        out[f"scaler.{name}.m"] = np.asarray(rs.m if rs.n else 0.0, np.float64)
        out[f"scaler.{name}.s"] = np.asarray(rs.s if rs.n else 0.0, np.float64)
    out["G"] = np.asarray(agent.G, np.float64)
    return out


def one_update(
    agent, obs, next_obs, reward, terminated, noise: list, double: bool
) -> dict:
    """compute_action then update once, on a copy, with ``noise`` as its draws (the action's, then
    the next action's, when given), in float64 with ``double``."""
    a = copy.deepcopy(agent)
    default = torch.get_default_dtype()
    if double:
        for net in (a.actor, a.Q):
            net.double()
        for opt in (a.popt, a.qopt):
            for st in opt.state.values():
                for k in ("exp_avg", "exp_avg_sq"):
                    st[k] = st[k].double()
        torch.set_default_dtype(torch.float64)  # the update's torch.Tensor(obs)
    NOISE.given, NOISE.drawn = list(noise), []
    try:
        action, info = a.compute_action(obs)
        a.update(obs, action, next_obs, reward, terminated, **info)
    finally:
        torch.set_default_dtype(default)
    drawn, NOISE.drawn = NOISE.drawn, []
    return state(a) | {"draws": drawn}


def record(env_id: str, out: pathlib.Path) -> None:
    env = GymnaxEnv(env_id, SEED)
    env = gymnasium.wrappers.FlattenObservation(env)
    env = gymnasium.wrappers.NormalizeObservation(env)
    env = gymnasium.wrappers.ClipAction(env)
    env.reset(seed=SEED)
    env.action_space.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = avg.AVG(
        types.SimpleNamespace(obs_dim=obs_dim, action_dim=action_dim, **DEFAULTS)
    )
    out.mkdir(parents=True, exist_ok=True)
    obs, _ = env.reset()
    for t in range(max(STEPS) + 1):
        action, action_info = agent.compute_action(obs)
        sim_action = action.detach().cpu().view(-1).numpy()
        next_obs, reward, terminated, truncated, _ = env.step(sim_action)
        if t in STEPS:
            noise = [NOISE.drawn[-1]]  # compute_action's
            f32 = one_update(agent, obs, next_obs, reward, terminated, noise, False)
            noise = f32.pop("draws")  # the action's, then the update's next action's
            f64 = one_update(agent, obs, next_obs, reward, terminated, noise, True)
            f64.pop("draws")
            arrays = {f"in.{k}": v for k, v in state(agent).items()}
            arrays |= {
                "obs": np.asarray(obs, np.float32),
                "action": sim_action,
                "action_pre": action_info["action_pre"].detach().numpy().reshape(-1),
                "next_obs": np.asarray(next_obs, np.float32),
                "reward": np.asarray(reward, np.float64),
                "terminated": np.asarray(float(terminated)),
                "noise.action": noise[0].reshape(-1),
                "noise.next": noise[1].reshape(-1),
            }
            arrays |= {f"f32.{k}": v for k, v in f32.items()}
            arrays |= {f"f64.{k}": v for k, v in f64.items()}
            np.savez(out / f"{env_id}_{t}.npz", **arrays)
        NOISE.drawn = []
        agent.update(obs, action, next_obs, reward, terminated, **action_info)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    write_versions(out, gauthamvasan_avg=COMMIT)


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
