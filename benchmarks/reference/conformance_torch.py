"""What the PyTorch originals' recorders share (conformance_*.py): a gymnax task behind the old gym
API their loops use, a record's library versions, and, for sfujim's TD3 and TD3_BC (the same
networks, optimisers and ``train(replay_buffer, batch_size)``), the update's state and one update
from it in either precision. Imports nothing from Ajax.
"""

import copy
import importlib.metadata
import json
import pathlib
import platform

import gymnax
import jax
import numpy as np
import torch


class OldGym:
    """A gymnax task with the gym API the originals' loops use: reset() -> obs, step(a) -> (obs,
    reward, done, info), _max_episode_steps, and a box to sample."""

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
        self.terminated = bool(term)  # a true termination, not a time limit
        return np.asarray(obs), float(reward), bool(term | trunc), {}


def write_versions(out: pathlib.Path, **original: str) -> None:
    versions = {
        lib: importlib.metadata.version(lib)
        for lib in ("torch", "numpy", "jax", "gymnax")
    }
    (out / "versions.json").write_text(
        json.dumps(versions | {"python": platform.python_version()} | original)
    )


def sfujim_state(policy) -> dict:
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


def sfujim_update(policy, batch, noise, double: bool, batch_size: int) -> dict:
    """``policy.train`` once from a copy, with ``batch`` as its sample and ``noise`` as its
    randn_like."""
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
        p.train(Fixed(), batch_size)
    finally:
        torch.randn_like = randn_like
    return sfujim_state(p)


def sfujim_snapshot(policy, rb, batch_size: int, noise_shape: tuple) -> dict:
    """The arrays of one record: the state (``in.``), the update's batch as ``rb.sample`` draws it
    (``batch.``), the target noise ``train``'s randn_like draws, and the state after the update in
    float32 and float64 (``f32.``, ``f64.``)."""
    ind = np.random.randint(0, rb.size, size=batch_size)  # what rb.sample draws
    batch = tuple(
        np.asarray(x[ind], np.float32)
        for x in (rb.state, rb.action, rb.next_state, rb.reward, rb.not_done)
    )
    noise = torch.randn(noise_shape).numpy()  # what train's randn_like draws
    arrays = {f"in.{k}": v for k, v in sfujim_state(policy).items()}
    names = ("state", "action", "next_state", "reward", "not_done")
    arrays |= {f"batch.{k}": v for k, v in zip(names, batch, strict=True)}
    arrays["noise"] = noise
    for tag, double in (("f32", False), ("f64", True)):
        arrays |= {
            f"{tag}.{k}": v
            for k, v in sfujim_update(policy, batch, noise, double, batch_size).items()
        }
    return arrays
