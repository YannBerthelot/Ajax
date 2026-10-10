"""Behaviour pins for SAC's descendants (ASAC, REDQ, AVG), TD3 and
recurrent SAC.

Each case trains a tiny fixed run and compares a fingerprint (the sum of
squares of each parameter tree, and alpha) with the one recorded before
refactoring roadmap step 16 moved these agents onto SAC's core helpers: a
restructure must reproduce it. The relative tolerance absorbs CPU platform
drift (fp32 reduction order differs between CI's Linux x86 and macOS ARM);
an algorithmic change moves a fingerprint far more.
"""

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.networks.memory import MemoryConfig

_SMALL: dict[str, Any] = {
    "env_id": "Pendulum-v1",
    "n_envs": 2,
    "actor_architecture": ("32", "relu"),
    "critic_architecture": ("32", "relu"),
}
_REPLAY: dict[str, Any] = {"learning_starts": 20, "batch_size": 16, "buffer_size": 400}
_CASES: dict[str, tuple[type, dict[str, Any]]] = {
    "ASAC": (ASAC, _REPLAY),
    "REDQ": (REDQ, {**_REPLAY, "num_critics": 4, "num_critic_updates": 3}),
    "AVG": (AVG, {"learning_starts": 20}),
    "TD3": (TD3, _REPLAY),
    "SAC-gru": (
        SAC,
        {
            "n_envs": 1,
            "learning_starts": 40,
            "batch_size": 8,
            "buffer_size": 400,
            "burn_in": 2,
            "sequence_length": 4,
            "memory": MemoryConfig(kind="gru", hidden_size=8),
            "policy_update_start": 40,
            "alpha_update_start": 40,
        },
    ),
}
_TIMESTEPS = 200
_TOL = 1e-3
# Recorded on macOS ARM CPU at the parent of the step-16 lineage commit.
_GOLDEN: dict[str, dict[str, float]] = {
    "ASAC": {
        "actor": 37.438865661621094,
        "critic": 74.46932983398438,
        "alpha": 0.9724646210670471,
    },
    "AVG": {
        "actor": 5.874258518218994,
        "critic": 18.981433868408203,
        "alpha": 0.07000000029802322,
    },
    "REDQ": {
        "actor": 37.609249114990234,
        "critic": 158.7569122314453,
        "alpha": 0.9725564122200012,
    },
    "SAC-gru": {
        "actor": 87.06092834472656,
        "critic": 173.4002227783203,
        "alpha": 0.951093316078186,
    },
    "TD3": {"actor": 4.087213039398193, "critic": 79.58155059814453},
}


def _checksum(tree: Any) -> float:
    return float(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree)))


def fingerprint(name: str) -> dict[str, float]:
    agent_cls, kwargs = _CASES[name]
    state, _ = agent_cls(**{**_SMALL, **kwargs}).train(seed=0, n_timesteps=_TIMESTEPS)
    out = {
        "actor": _checksum(state.actor_state.params),
        "critic": _checksum(state.critic_state.params),
    }
    if hasattr(state, "alpha"):
        out["alpha"] = float(jnp.exp(state.alpha.params["log_alpha"]).reshape(-1)[0])
    return out


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_tiny_run_reproduces_its_recorded_fingerprint(name: str) -> None:
    got = fingerprint(name)
    for key, golden in _GOLDEN[name].items():
        rel = abs(got[key] - golden) / max(abs(golden), 1.0)
        assert rel < _TOL, f"{name} {key}: got {got[key]!r}, golden {golden!r}"
