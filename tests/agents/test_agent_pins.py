"""Behaviour pins for the agents a refactoring step restructures: SAC's
descendants (ASAC, REDQ, AVG), TD3, recurrent SAC, PPO on each minibatch
geometry, APO, PQN and DQN.

Each case trains a tiny fixed run and compares a fingerprint (the sum of
squares of each parameter tree, alpha, the ``Nudge`` extension's state) with
the one recorded before the restructure: a restructure must reproduce it.
The relative tolerance absorbs CPU platform drift (fp32 reduction order
differs between CI's Linux x86 and macOS ARM); an algorithmic change moves a
fingerprint far more.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.DQN.DQN import DQN
from ajax.agents.DQN.networks import DuelingQNetwork
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig


def _checksum(tree: Any) -> jax.Array:
    return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree))


@dataclass(frozen=True)
class Nudge(Extension):
    """Touches every phase the on-policy and value agents fold, its state
    drawn from the keys they hand it."""

    name: str = "nudge"

    def init_state(self, agent_state, rng):
        return jax.random.normal(rng)

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target + 0.01 * ext_state

    def critic_loss(self, agent_state, ext_state, batch, ctx):
        params = batch.get("critic_params")
        return 0.0 if params is None else 1e-3 * _checksum(params)

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        params = batch.get("actor_params")
        return 0.0 if params is None else 1e-3 * _checksum(params)

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + 0.01 * jax.random.normal(ctx.rng)

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {"nudge": ext_state}


_SMALL: dict[str, Any] = {
    "env_id": "Pendulum-v1",
    "n_envs": 2,
    "actor_architecture": ("32", "relu"),
    "critic_architecture": ("32", "relu"),
}
_REPLAY: dict[str, Any] = {"learning_starts": 20, "batch_size": 16, "buffer_size": 400}
_PENDULUM: dict[str, Any] = {
    "env_id": "Pendulum-v1",
    "actor_architecture": ("16", "tanh"),
    "critic_architecture": ("16", "tanh"),
}
_CARTPOLE: dict[str, Any] = {"env_id": "CartPole-v1", "architecture": ("16", "relu")}
_SPLIT: dict[str, Any] = {"n_envs": 4, "n_steps": 16, "num_minibatches": 2}
_PQN: dict[str, Any] = {**_CARTPOLE, **_SPLIT, "n_epochs": 2}
_DQN: dict[str, Any] = {**_CARTPOLE, **_REPLAY, "target_update_interval": 7}
_CASES: dict[str, tuple[type, dict[str, Any], int]] = {
    "ASAC": (ASAC, {**_SMALL, **_REPLAY}, 200),
    "REDQ": (
        REDQ,
        {**_SMALL, **_REPLAY, "num_critics": 4, "num_critic_updates": 3},
        200,
    ),
    "AVG": (AVG, {**_SMALL, "learning_starts": 20}, 200),
    "TD3": (TD3, {**_SMALL, **_REPLAY}, 200),
    "SAC-gru": (
        SAC,
        {
            **_SMALL,
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
        200,
    ),
    # Flat shuffle, GAE on the whole rollout: 4 minibatches, 2 envs.
    "PPO-flat": (PPO, {**_PENDULUM, "n_envs": 2, "n_steps": 32, "batch_size": 8}, 256),
    # The env axis split, GAE per minibatch; brax's knobs.
    "PPO-env-split": (
        PPO,
        {
            **_PENDULUM,
            **_SPLIT,
            "squash": True,
            "log_std_state_independent": True,
            "log_std_init": 0.0,
            "normalize_observations": True,
            "ent_coef": 0.01,
            "fused_grad_clip": True,
            "num_resets_per_eval": 1,
            "num_evals": 3,
        },
        256,
    ),
    "PPO-unroll": (
        PPO,
        {
            **_PENDULUM,
            "n_envs": 2,
            "n_steps": 32,
            "num_minibatches": 4,
            "unroll_length": 8,
        },
        256,
    ),
    "PPO-gru": (
        PPO,
        {
            **_PENDULUM,
            "n_envs": 2,
            "n_steps": 16,
            "num_minibatches": 2,
            "bptt_length": 4,
            "memory": MemoryConfig(kind="gru", hidden_size=8),
        },
        256,
    ),
    "PPO-discrete-nudge": (
        PPO,
        {
            "env_id": "CartPole-v1",
            "actor_architecture": ("16", "tanh"),
            "critic_architecture": ("16", "tanh"),
            "n_envs": 2,
            "n_steps": 32,
            "batch_size": 16,
            "ent_coef": 0.01,
            "extensions": (Nudge(),),
        },
        256,
    ),
    "PPO-env-split-nudge": (
        PPO,
        {**_PENDULUM, **_SPLIT, "extensions": (Nudge(),)},
        256,
    ),
    "APO-nudge": (
        APO,
        {
            **_PENDULUM,
            "n_envs": 2,
            "n_steps": 32,
            "batch_size": 16,
            "extensions": (Nudge(),),
        },
        256,
    ),
    "PQN": (PQN, _PQN, 512),
    "PQN-nudge": (PQN, {**_PQN, "extensions": (Nudge(),)}, 512),
    "DQN": (DQN, _DQN, 200),
    "DQN-dueling-nudge": (
        DQN,
        {**_DQN, "q_network_cls": DuelingQNetwork, "extensions": (Nudge(),)},
        200,
    ),
}
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
# Recorded on macOS ARM CPU once PPO and PQN took TrainLoop's init keys,
# before step 17 restructured them. DQN's and PQN's critic state is a
# never-updated copy of the initial Q-network: not pinned.
_GOLDEN |= {
    "PPO-flat": {"actor": 20.46432876586914, "critic": 21.161226272583008},
    "PPO-env-split": {"actor": 19.116395950317383, "critic": 20.26976776123047},
    "PPO-unroll": {"actor": 20.528013229370117, "critic": 21.03093910217285},
    "PPO-gru": {"actor": 69.79020690917969, "critic": 68.853759765625},
    "PPO-discrete-nudge": {
        "actor": 21.793909072875977,
        "critic": 21.164554595947266,
        "nudge": 0.29222556948661804,
    },
    "PPO-env-split-nudge": {
        "actor": 19.185230255126953,
        "critic": 20.3308048248291,
        "nudge": 0.3099551498889923,
    },
    "APO-nudge": {
        "actor": 3.837249994277954,
        "critic": 4.151569843292236,
        "nudge": 0.33622848987579346,
    },
    "PQN": {"actor": 25.90884017944336},
    "PQN-nudge": {"actor": 25.898910522460938, "nudge": 0.33566388487815857},
    "DQN": {"actor": 28.83551025390625, "target": 28.488685607910156},
    "DQN-dueling-nudge": {
        "actor": 25.996488571166992,
        "target": 25.84494400024414,
        "nudge": 0.22634370625019073,
    },
}


def fingerprint(name: str) -> dict[str, float]:
    agent_cls, kwargs, steps = _CASES[name]
    state, _ = agent_cls(**kwargs).train(seed=0, n_timesteps=steps)
    out = {
        "actor": float(_checksum(state.actor_state.params)),
        "critic": float(_checksum(state.critic_state.params)),
    }
    if getattr(state.actor_state, "target_params", None) is not None:
        out["target"] = float(_checksum(state.actor_state.target_params))
    if hasattr(state, "alpha"):
        out["alpha"] = float(jnp.exp(state.alpha.params["log_alpha"]).reshape(-1)[0])
    if state.ext_state:
        out["nudge"] = float(jnp.asarray(state.ext_state[0]).reshape(-1)[0])
    return out


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_tiny_run_reproduces_its_recorded_fingerprint(name: str) -> None:
    got = fingerprint(name)
    for key, golden in _GOLDEN[name].items():
        rel = abs(got[key] - golden) / max(abs(golden), 1.0)
        assert rel < _TOL, f"{name} {key}: got {got[key]!r}, golden {golden!r}"
