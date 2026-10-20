"""sfujim/TD3's update, recorded for the conformance check (tests/conformance/test_td3_sfujim.py).

usage: python conformance_sfujim_td3.py ENV_ID OUT_DIR   (in envs/sb3: PyTorch, gymnax, jax)

Imports nothing from Ajax. sfujim's own TD3.py and ReplayBuffer (vendor/sfujim_td3) train on a gymnax
task by main.py's loop, written out below: random actions for START steps, then the actor's action plus
N(0, 0.1 max_action), done_bool 0 at a time limit, one train() per step. Before the updates that bring
total_it to SNAPSHOTS + 1, the state is recorded, with the update's own draws (its batch indices, then
its target noise), and the update is run from it twice: as is, and in float64, whose difference is the
float32 rounding the check tolerates. One .npz per snapshot: ``in.*`` (the state), ``batch.*``,
``noise``, ``max_action``, ``f32.*`` and ``f64.*`` (the states after the update), and ``versions.json``.
"""

import copy
import importlib.metadata
import json
import pathlib
import platform
import sys

import gymnax
import jax
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent / "vendor" / "sfujim_td3"))
import TD3
import utils

SNAPSHOTS = (9, 10, 499, 500)
START, BATCH, SEED = 1_000, 256, 0
COMMIT = "34770ccdcb51df6f6c7d85c1ba7e40d71b940d16"


class OldGym:
    """A gymnax task with the gym API main.py uses: reset() -> obs, step(a) -> (obs, reward, done,
    info), _max_episode_steps, and a box to sample."""

    def __init__(self, env_id: str, seed: int):
        self.env, self.params = gymnax.make(env_id)
        space = self.env.action_space(self.params)
        self.shape = space.shape
        self.low = np.broadcast_to(np.asarray(space.low, np.float32), self.shape)
        self.high = np.broadcast_to(np.asarray(space.high, np.float32), self.shape)
        self.obs_dim = self.env.observation_space(self.params).shape[0]
        self._max_episode_steps = int(self.params.max_steps_in_episode)
        self.key = jax.random.PRNGKey(seed)
        self._step = jax.jit(lambda k, s, a: self.env.step(k, s, a, self.params))
        self._reset = jax.jit(lambda k: self.env.reset(k, self.params))

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, self.shape).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.key, k = jax.random.split(self.key)
        obs, self.state = self._reset(k)
        return np.asarray(obs)

    def step(self, action):
        self.key, k = jax.random.split(self.key)
        obs, self.state, reward, term, trunc, _ = self._step(
            k, self.state, np.asarray(action, np.float32)
        )
        return np.asarray(obs), float(reward), bool(term | trunc), {}


def named(policy) -> dict:
    """The update's state in float64 arrays: the four networks, both Adams, total_it."""
    out = {}
    for prefix in ("actor", "actor_target", "critic", "critic_target"):
        for name, p in getattr(policy, prefix).named_parameters():
            out[f"{prefix}.{name}"] = p.detach().double().numpy().copy()
    for prefix, module, opt in (
        ("opt.actor", policy.actor, policy.actor_optimizer),
        ("opt.critic", policy.critic, policy.critic_optimizer),
    ):
        for name, p in module.named_parameters():
            state = opt.state.get(p, {})
            for k in ("exp_avg", "exp_avg_sq", "step"):
                if k in state:
                    out[f"{prefix}.{name}.{k}"] = np.asarray(
                        state[k].detach().double().numpy()
                    ).copy()
    out["total_it"] = np.asarray(policy.total_it)
    return out


def one_update(policy, batch, noise, double: bool) -> dict:
    """``policy.train`` once from a copy, with ``batch`` as its sample and ``noise`` as its randn_like."""
    p = copy.deepcopy(policy)
    cast = (lambda t: t.double()) if double else (lambda t: t)
    if double:
        for module in (p.actor, p.actor_target, p.critic, p.critic_target):
            module.double()
        for opt in (p.actor_optimizer, p.critic_optimizer):
            for state in opt.state.values():
                for k in ("exp_avg", "exp_avg_sq"):
                    state[k] = state[k].double()

    class Fixed:
        def sample(self, _):
            return tuple(cast(torch.as_tensor(b)) for b in batch)

    randn_like = torch.randn_like
    torch.randn_like = lambda _: cast(torch.as_tensor(noise))
    try:
        p.train(Fixed(), BATCH)
    finally:
        torch.randn_like = randn_like
    return named(p)


def record(env_id: str, out: pathlib.Path) -> None:
    torch.set_num_threads(1)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    env = OldGym(env_id, SEED)
    max_action = float(env.high[0])
    policy = TD3.TD3(
        state_dim=env.obs_dim,
        action_dim=env.shape[0],
        max_action=max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2 * max_action,
        noise_clip=0.5 * max_action,
        policy_freq=2,
    )  # main.py's kwargs
    rb = utils.ReplayBuffer(env.obs_dim, env.shape[0])
    out.mkdir(parents=True, exist_ok=True)
    state, done, episode_timesteps, t = env.reset(), False, 0, 0
    while policy.total_it <= max(SNAPSHOTS):
        episode_timesteps += 1
        if t < START:
            action = env.sample()
        else:
            noise = np.random.normal(0, max_action * 0.1, size=env.shape[0])
            action = (policy.select_action(np.array(state)) + noise).clip(
                -max_action, max_action
            )
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
        rb.add(state, action, next_state, reward, done_bool)
        state = next_state
        if done:
            state, done, episode_timesteps = env.reset(), False, 0
        if t >= START:
            if policy.total_it in SNAPSHOTS:
                ind = np.random.randint(0, rb.size, size=BATCH)  # what rb.sample draws
                batch = tuple(
                    np.asarray(x[ind], np.float32)
                    for x in (
                        rb.state,
                        rb.action,
                        rb.next_state,
                        rb.reward,
                        rb.not_done,
                    )
                )
                noise = torch.randn(
                    (BATCH, env.shape[0])
                ).numpy()  # what train's randn_like draws
                arrays = {f"in.{k}": v for k, v in named(policy).items()}
                arrays |= {
                    f"batch.{k}": v
                    for k, v in zip(
                        ("state", "action", "next_state", "reward", "not_done"),
                        batch,
                        strict=False,
                    )
                }
                arrays |= {"noise": noise, "max_action": np.asarray(max_action)}
                for tag, double in (("f32", False), ("f64", True)):
                    arrays |= {
                        f"{tag}.{k}": v
                        for k, v in one_update(policy, batch, noise, double).items()
                    }
                np.savez(out / f"{env_id}_{policy.total_it}.npz", **arrays)
            policy.train(rb, BATCH)
        t += 1
    versions = {
        lib: importlib.metadata.version(lib)
        for lib in ("torch", "numpy", "jax", "gymnax")
    }
    (out / "versions.json").write_text(
        json.dumps(
            versions | {"python": platform.python_version(), "sfujim_td3": COMMIT}
        )
    )


if __name__ == "__main__":
    record(sys.argv[1], pathlib.Path(sys.argv[2]))
