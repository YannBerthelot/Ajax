"""One agent factory: each agent's preset, plus overrides.

``make`` builds the agent from its preset with the overrides on top, then
reads every override back from the agent's config: one that did not take
(a subclass forwarding ``**kwargs`` can swallow it) raises RuntimeError.
The presets are frozen and pinned by ``DIGEST``: a calibration holds only
for the configuration it was made on.
"""

from __future__ import annotations

import hashlib
import importlib
from types import MappingProxyType
from typing import Any, Mapping

import ajax

GAMMA = 0.62
"""At 0.62 the discounting probe's wrong answers (gamma squared, 1 - gamma,
no discount) sit at least 0.24 from the right one; at 0.5 two coincide."""
NET = ("64", "relu", "64", "relu")
SMALL = ("16", "relu", "16", "relu")


def _split(net: tuple[str, ...]) -> dict[str, Any]:
    lr = {"actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3}
    return {**lr, "actor_architecture": net, "critic_architecture": net}


_NORM = {"normalize_observations": False, "normalize_rewards": False}
_REPLAY = {"learning_starts": 100, "buffer_size": 10_000, "batch_size": 64}
_SAC = {**_split(NET), **_NORM, **_REPLAY, "gamma": GAMMA, "n_envs": 1}
_PPO = {**_split(NET), **_NORM, "n_steps": 32, "batch_size": 32, "n_epochs": 4}
_PPO |= {"n_envs": 1}
_Q = {"learning_rate": 1e-3, "architecture": NET, "gamma": GAMMA, "n_envs": 1}
_PROBE = {
    **dict.fromkeys(("SAC", "TD3", "REDQ"), _SAC),
    "ASAC": {k: v for k, v in _SAC.items() if k != "gamma"},
    "AVG": {"actor_architecture": NET, "critic_architecture": NET}
    | {"gamma": GAMMA, "n_envs": 1},
    "PPO": {**_PPO, "gamma": GAMMA},
    "APO": _PPO,
    "DQN": {**_Q, **_REPLAY, "target_update_interval": 100},
    # PQN learns from parallel envs, not a replay buffer: with one env the
    # value of the action it stops exploring drifts.
    "PQN": {**_Q, "n_envs": 8, "n_steps": 16, "n_epochs": 4, "num_minibatches": 1},
    "APG": {"actor_architecture": NET, "n_envs": 64},
}

# One preset for every bookkeeping probe (decision D2): their answers do not
# depend on learning, so small nets, a small buffer, and update schedules
# whose step counts tell their parts apart. P2 was calibrated on it; Q3, Q5,
# Q8, P3's counter, metric and E10 cells and P4's non-resumable cells are
# re-certified on it as they port (their world models: Q3's small ones).
_B_REPLAY = {"learning_starts": 100, "buffer_size": 2000, "batch_size": 16}
_B_SAC = {**_split(SMALL), **_NORM, **_B_REPLAY, "gamma": GAMMA, "n_envs": 1}
_B_PPO = {**_split(SMALL), **_NORM, "n_steps": 32, "batch_size": 16, "n_epochs": 2}
_B_PPO |= {"n_envs": 1}
_B_Q = {**_NORM, "learning_rate": 1e-3, "architecture": SMALL, "gamma": GAMMA}
_B_Q |= {"n_envs": 1}
_B_SACS = {**_B_SAC, "policy_update_start": 200, "alpha_update_start": 300}
_BOOKKEEPING = {
    "SAC": _B_SACS,
    "TD3": {**_B_SAC, "policy_delay": 3},
    "REDQ": {**_B_SAC, "num_critics": 4, "subset_size": 2, "num_critic_updates": 3},
    "ASAC": {k: v for k, v in _B_SAC.items() if k != "gamma"},
    # AVG always normalises observations and never rewards (AVG.py:80-93).
    "AVG": {**_split(SMALL), "gamma": GAMMA, "learning_starts": 0, "n_envs": 1},
    "PPO": {**_B_PPO, "gamma": GAMMA},
    "APO": _B_PPO,
    "DQN": {**_B_Q, **_B_REPLAY, "target_update_interval": 100},
    "PQN": {**_B_Q, "n_steps": 16, "n_epochs": 2, "num_minibatches": 2},
    "UDRL": {"actor_architecture": SMALL, "n_steps": 32, "batch_size": 32}
    | {"n_epochs": 2, "n_updates_per_iter": 4, "n_envs": 1},
    "APG": {"actor_architecture": SMALL, "horizon": 16, "n_envs": 1},
    "DreamerV3": {"model_size": "1m", "units": 16, "hidden": 16, "deter": 32}
    | {"classes": 4, "stoch": 4, "blocks": 4, "imag_horizon": 3, "batch_size": 4}
    | {"batch_length": 8, "warmup": 10, "train_ratio": 8, "replay_capacity": 1000}
    | {"n_envs": 1},
    "TDMPC2": {"enc_dim": 32, "mlp_dim": 32, "latent_dim": 16, "num_q": 2}
    | {"batch_size": 16, "num_samples": 32, "num_elites": 4, "num_pi_trajs": 4}
    | {"iterations": 2, "gamma": 0.8, "n_envs": 1},
}

# The world models' learning probes (Q10, Q12, Q13): the tiny models of
# tests/agents (test_tdmpc2_probes.py:35-49, test_probing.py's DreamerV3),
# copied so their retuning does not move these calibrations.
_T_CORE: dict[str, Any] = {"enc_dim": 32, "mlp_dim": 32, "latent_dim": 16}
_T_CORE |= {"num_q": 2, "batch_size": 32, "num_samples": 64, "num_elites": 8}
_T_CORE |= {"num_pi_trajs": 4, "iterations": 4, "tau": 0.1, "learning_rate": 1e-3}
_TINY: dict[str, dict[str, Any]] = {
    "DreamerV3": {"model_size": "1m", "units": 32, "hidden": 32, "deter": 64}
    | {"classes": 4, "stoch": 8, "blocks": 4, "batch_size": 8, "batch_length": 8}
    | {"train_ratio": 8, "imag_horizon": 5, "warmup": 100, "learning_rate": 1e-3}
    | {"n_envs": 1},
    "TDMPC2": {**_T_CORE, "n_envs": 1},
    "TDMPC2MultiTask": {**_T_CORE, "task_dim": 4},
}

PRESETS: Mapping[str, Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {"probe": _PROBE, "bookkeeping": _BOOKKEEPING, "tiny": _TINY}
)
DIGEST = "8442f391cf52"
FAMILY = {"PPO": "v", "APO": "v", "DQN": "dqn", "PQN": "dqn"}
"""Which value an agent's readout reads (others: Q at the mean action)."""


def digest() -> str:
    return hashlib.sha256(repr(sorted(PRESETS.items())).encode()).hexdigest()[:12]


def agent_class(name: str) -> type:
    if name in ajax.__all__:
        return getattr(ajax, name)
    return getattr(importlib.import_module(f"ajax.agents.{name}.{name}"), name)


_MISSING = object()


def _read_back(agent: Any, key: str) -> Any:
    """An override's value as the agent holds it, where it holds one."""
    for name in ("agent_config", "env_args", "network_args"):
        if hasattr(getattr(agent, name, None), key):
            return getattr(getattr(agent, name), key)
    return getattr(agent, key, _MISSING)


def make(
    name: str,
    env: Any,
    params: Any = None,
    *,
    preset: str | None = "probe",
    extensions: tuple = (),
    cls: type | None = None,
    **overrides: Any,
) -> Any:
    """``preset`` None builds from the class defaults; ``cls`` stands in
    for the agent's class (a subclass); ``env`` None builds an offline
    agent (TDMPC2MultiTask: its ``dataset`` an override). Scalar and tuple
    overrides must read back."""
    kwargs = {**(PRESETS[preset].get(name, {}) if preset else {}), **overrides}
    kwargs |= {} if env is None else {"env_id": env, "env_params": params}
    agent = (cls or agent_class(name))(extensions=extensions, **kwargs)
    for key, value in overrides.items():
        took = _read_back(agent, key)
        if took is _MISSING or not isinstance(value, (int, float, str, tuple)):
            continue
        if not (took is value or took == value):
            raise RuntimeError(f"{name} took {key}={took!r}, not {value!r}")
    return agent
