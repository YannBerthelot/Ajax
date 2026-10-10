"""Behaviour pins for the agents a refactoring step restructures: SAC's
descendants (ASAC, REDQ, AVG), TD3, recurrent SAC, PPO on each minibatch
geometry, APO, PQN, DQN, the world models (DreamerV3, TD-MPC2 single- and
multi-task), UDRL and APG.

Each case trains a tiny fixed run and compares a fingerprint (the sum of
squares of each parameter tree, alpha, the ``Nudge`` or ``Drift``
extension's state) with the one recorded before the restructure: a
restructure must reproduce it.
The relative tolerance absorbs CPU platform drift (fp32 reduction order
differs between CI's Linux x86 and macOS ARM); an algorithmic change moves a
fingerprint far more.
"""

from dataclasses import dataclass
from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.APG.APG import APG
from ajax.agents.APG.networks import PIDHeadConfig
from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.DQN.DQN import DQN
from ajax.agents.DQN.networks import DuelingQNetwork
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.agents.TDMPC2.dataset import TaskEpisodes, pool_tasks
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.agents.TDMPC2.TDMPC2MultiTask import TDMPC2MultiTask
from ajax.agents.UDRL.UDRL import UDRL
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig


def _checksum(tree: Any) -> jax.Array:
    return sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree))


@dataclass(frozen=True)
class Nudge(Extension):
    """Touches every phase the on-policy agents fold, its state drawn from
    the keys they hand it (agents reject the phases they do not fold)."""

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


@dataclass(frozen=True)
class ValueNudge(Nudge):
    """Nudge on the value agents (DQN, PQN), which fold no actor loss."""

    actor_loss = Extension.actor_loss


@dataclass(frozen=True)
class ActorNudge(Nudge):
    """Nudge on the agents training an actor alone (UDRL, APG), which fold
    no target and no critic loss."""

    on_target = Extension.on_target
    critic_loss = Extension.critic_loss


@dataclass(frozen=True)
class Drift(Extension):
    """The phases the world models fold: its state drawn at init, drifted
    on the key of every update."""

    name: str = "drift"

    def init_state(self, agent_state, rng):
        return jax.random.normal(rng)

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + 0.01 * jax.random.normal(ctx.rng)

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {"drift": ext_state}


def _multitask(**kwargs: Any) -> TDMPC2MultiTask:
    """Two toy tasks of 6-row episodes, dims (3, 2) and (2, 1)."""
    rng = np.random.default_rng(0)
    tasks = [
        TaskEpisodes(
            obs=rng.normal(size=(4, 6, obs)).astype(np.float32),
            action=rng.uniform(-1, 1, (4, 6, act)).astype(np.float32),
            reward=rng.uniform(0, 1, (4, 6)).astype(np.float32),
            episode_length=5,
            name=name,
        )
        for name, obs, act in (("a", 3, 2), ("b", 2, 1))
    ]
    return TDMPC2MultiTask(pool_tasks(tasks), **kwargs)


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
_TDMPC2_TINY: dict[str, Any] = {
    "enc_dim": 16,
    "mlp_dim": 16,
    "latent_dim": 8,
    "num_q": 2,
    "batch_size": 8,
    "num_samples": 16,
    "num_elites": 4,
    "num_pi_trajs": 2,
    "iterations": 2,
    "extensions": (Drift(),),
}
_DREAMER_TINY: dict[str, Any] = {
    "env_id": "CartPole-v1",
    "n_envs": 2,
    "model_size": "1m",
    "units": 16,
    "hidden": 16,
    "deter": 32,
    "classes": 4,
    "stoch": 4,
    "blocks": 4,
    "imag_horizon": 3,
    "batch_size": 4,
    "batch_length": 8,
    "warmup": 10,
    "train_ratio": 32,
    "replay_capacity": 80,
    "extensions": (Drift(),),
}
_APG: dict[str, Any] = {
    "env_id": "Pendulum-v1",
    "n_envs": 2,
    "horizon": 8,
    "actor_architecture": ("16", "relu"),
    "extensions": (ActorNudge(),),
}
_CASES: dict[str, tuple[Any, dict[str, Any], int]] = {
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
    "PQN-nudge": (PQN, {**_PQN, "extensions": (ValueNudge(),)}, 512),
    "DQN": (DQN, _DQN, 200),
    "DQN-dueling-nudge": (
        DQN,
        {**_DQN, "q_network_cls": DuelingQNetwork, "extensions": (ValueNudge(),)},
        200,
    ),
    # 40 ticks of 2 rows; the first update after tick 9, one per row after.
    "DreamerV3-drift": (DreamerV3, _DREAMER_TINY, 80),
    # Episodes of 10 steps; the 20-update burst on step 20, one per step after.
    "TDMPC2-drift": (
        TDMPC2,
        {
            "env_id": "Pendulum-v1",
            "env_params": gymnax.make("Pendulum-v1")[1].replace(
                max_steps_in_episode=10
            ),
            "seed_steps": 20,
            **_TDMPC2_TINY,
        },
        40,
    ),
    "TDMPC2MultiTask-drift": (_multitask, {**_TDMPC2_TINY, "task_dim": 4}, 12),
    "UDRL-nudge": (
        UDRL,
        {
            "env_id": "CartPole-v1",
            "n_envs": 2,
            "actor_architecture": ("16", "relu"),
            "n_steps": 16,
            "batch_size": 16,
            "buffer_capacity": 8,
            "n_updates_per_iter": 4,
            "extensions": (ActorNudge(),),
        },
        128,
    ),
    "APG-nudge": (APG, _APG, 80),
    "APG-pid-nudge": (APG, {**_APG, "pid": PIDHeadConfig()}, 80),
}
_TOL = 1e-3
# Recorded on macOS ARM CPU at the parent of the step-16 lineage commit.
_GOLDEN: dict[str, dict[str, float]] = {
    "AVG": {
        "actor": 5.874258518218994,
        "critic": 18.981433868408203,
        "alpha": 0.07000000029802322,
    },
}
# Recorded on macOS ARM CPU once TD3's critic dropped its encoder LayerNorm:
# the critic tree lost the norm's scales (64 of the old 79.6) and the actor
# moved 3.6%.
_GOLDEN |= {"TD3": {"actor": 3.94071626663208, "critic": 11.506895065307617}}
# Recorded on macOS ARM CPU once PPO and APO drew a fresh minibatch
# partition every epoch (0.1-0.5% on the actors, under 0.3% on the critics).
_GOLDEN |= {
    "PPO-flat": {"actor": 20.36318016052246, "critic": 21.18039321899414},
    "PPO-env-split": {"actor": 19.09400177001953, "critic": 20.218984603881836},
    "PPO-unroll": {"actor": 20.432720184326172, "critic": 21.03376007080078},
    "PPO-gru": {"actor": 69.55516052246094, "critic": 68.83757019042969},
    "PPO-discrete-nudge": {
        "actor": 21.77309799194336,
        "critic": 21.17990493774414,
        "nudge": 0.29222556948661804,
    },
    "PPO-env-split-nudge": {
        "actor": 19.13752555847168,
        "critic": 20.32254409790039,
        "nudge": 0.3099551498889923,
    },
    "APO-nudge": {
        "actor": 3.8527090549468994,
        "critic": 4.158751964569092,
        "nudge": 0.33622848987579346,
    },
}
# Recorded on macOS ARM CPU once PQN took TrainLoop's init keys, before
# step 17 restructured it. DQN's and PQN's critic state is a never-updated
# copy of the initial Q-network: not pinned.
_GOLDEN |= {
    "PQN": {"actor": 25.90884017944336},
    "PQN-nudge": {"actor": 25.898910522460938, "nudge": 0.33566388487815857},
}
# Recorded on macOS ARM CPU once the shared encoder's output LayerNorm became
# opt-in (SAC, REDQ, ASAC, PPO and DQN build plain MLPs, as in their papers):
# each tree lost the norm's scales, one per hidden unit (16 or 32 per encoder).
# PPO's entries re-recorded once this met the per-epoch reshuffle (#81):
# 0.05-2.6% on the actors, under 0.4% on the critics.
_GOLDEN |= {
    "ASAC": {
        "actor": 4.500450611114502,
        "critic": 9.956274032592773,
        "alpha": 0.972590446472168,
    },
    "REDQ": {
        "actor": 4.491269588470459,
        "critic": 23.677350997924805,
        "alpha": 0.9729597568511963,
    },
    "SAC-gru": {
        "actor": 57.4627571105957,
        "critic": 109.36508178710938,
        "alpha": 0.9514929056167603,
    },
    "PPO-flat": {"actor": 4.138375282287598, "critic": 4.506361961364746},
    "PPO-env-split": {"actor": 3.0261974334716797, "critic": 4.079187870025635},
    "PPO-unroll": {"actor": 4.224678993225098, "critic": 4.548887729644775},
    "PPO-gru": {"actor": 53.321449279785156, "critic": 52.78514099121094},
    "PPO-discrete-nudge": {
        "actor": 5.820460796356201,
        "critic": 5.029939651489258,
        "nudge": 0.29222556948661804,
    },
    "PPO-env-split-nudge": {
        "actor": 3.742928981781006,
        "critic": 4.225018501281738,
        "nudge": 0.3099551498889923,
    },
    "DQN": {"actor": 9.462096214294434, "target": 9.303977966308594},
    "DQN-dueling-nudge": {
        "actor": 10.129926681518555,
        "target": 9.98266887664795,
        "nudge": 0.22634370625019073,
    },
}
# Recorded on macOS ARM CPU before step 17 moved the world models onto
# TrainLoop and tidied APG and UDRL (the world models once they took
# TrainLoop's init and post_update keys). UDRL's critic and its actor's
# target are never-updated copies: not pinned.
_GOLDEN |= {
    "DreamerV3-drift": {
        "actor": 92.46167755126953,
        "critic": 95.06945037841797,
        "world_model": 548.8908081054688,
        "drift": 0.23474201560020447,
    },
    "TDMPC2-drift": {
        "actor": 32.17240905761719,
        "world_model": 160.75112915039062,
        "drift": 0.3190682828426361,
    },
    "TDMPC2MultiTask-drift": {
        "actor": 32.23477554321289,
        "world_model": 161.12486267089844,
        "drift": 0.2877737283706665,
    },
    "UDRL-nudge": {"actor": 23.97211456298828, "nudge": 0.3160724639892578},
}
# Recorded on macOS ARM CPU once APG took TrainLoop's init and post_update
# keys, before step 17 moved it onto the shared evaluation.
_GOLDEN |= {
    "APG-nudge": {"actor": 35.82789611816406, "nudge": 0.27262088656425476},
    "APG-pid-nudge": {"actor": 36.82305145263672, "nudge": 0.27262088656425476},
}


def fingerprint(name: str) -> dict[str, float]:
    agent_cls, kwargs, steps = _CASES[name]
    state, _ = agent_cls(**kwargs).train(seed=0, n_timesteps=steps)
    out = {"actor": float(_checksum(state.actor_state.params))}
    if state.critic_state is not None:
        out["critic"] = float(_checksum(state.critic_state.params))
    if getattr(state, "world_model_state", None) is not None:
        out["world_model"] = float(_checksum(state.world_model_state.params))
    if getattr(state.actor_state, "target_params", None) is not None:
        out["target"] = float(_checksum(state.actor_state.target_params))
    if hasattr(state, "alpha"):
        out["alpha"] = float(jnp.exp(state.alpha.params["log_alpha"]).reshape(-1)[0])
    for ext, ext_state in zip(kwargs.get("extensions", ()), state.ext_state):
        out[ext.name] = float(jnp.asarray(ext_state).reshape(-1)[0])
    return out


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_tiny_run_reproduces_its_recorded_fingerprint(name: str) -> None:
    got = fingerprint(name)
    for key, golden in _GOLDEN[name].items():
        rel = abs(got[key] - golden) / max(abs(golden), 1.0)
        assert rel < _TOL, f"{name} {key}: got {got[key]!r}, golden {golden!r}"
