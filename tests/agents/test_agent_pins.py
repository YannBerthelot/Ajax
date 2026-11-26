"""Behaviour pins for the agents a refactoring step restructures: SAC's
descendants (ASAC, REDQ, AVG, TQC, CrossQ), SAC itself (plain, with an
expert's prefill and warm-up mix, recurrent), TD3, PPO on each minibatch
geometry, APO, PQN, DQN, the world models (DreamerV3, TD-MPC2 single- and
multi-task), UDRL, APG, DAgger and IQL.

Each case trains a tiny fixed run and compares a fingerprint (the sum of
squares of each parameter tree, alpha, the ``Nudge`` or ``Drift``
extension's state) with the one recorded before the restructure: a
restructure must reproduce it.
The relative tolerance absorbs CPU platform drift (fp32 reduction order
differs between CI's Linux x86 and macOS ARM); an algorithmic change moves a
fingerprint far more.

Characterization tests: they pin current behaviour, not verified against a
spec, and may encode bugs; a refactor guard, never proof of correctness.
"""

from dataclasses import dataclass
from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optax.tree_utils import tree_norm

from ajax.agents.APG.APG import APG
from ajax.agents.APG.networks import PIDHeadConfig
from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.CrossQ.CrossQ import CrossQ
from ajax.agents.DAgger.DAgger import DAgger
from ajax.agents.DQN.DQN import DQN
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.IQL.IQL import IQL
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.REINFORCE.REINFORCE import REINFORCE
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.agents.TDMPC2.dataset import TaskEpisodes, pool_tasks
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.agents.TDMPC2.TDMPC2MultiTask import TDMPC2MultiTask
from ajax.agents.TQC.TQC import TQC
from ajax.agents.UDRL.UDRL import UDRL
from ajax.extensions.base import Extension
from ajax.extensions.expert import ExpertWarmup
from ajax.extensions.pretrain import ExpertPrefill
from ajax.networks.memory import MemoryConfig
from ajax.networks.q_networks import DuelingQNetwork


@dataclass(frozen=True)
class Nudge(Extension):
    """Touches every phase the on-policy agents fold, its state drawn from
    the keys they hand it (agents reject the phases they do not fold)."""

    name: str = "nudge"

    def init_state(self, agent_state, rng):
        return jax.random.normal(rng)

    def on_target(self, agent_state, ext_state, batch, target, context):
        return target + 0.01 * ext_state

    def critic_loss(self, agent_state, ext_state, batch, context):
        params = batch.get("critic_params")
        return 0.0 if params is None else 1e-3 * tree_norm(params, squared=True)

    def actor_loss(self, agent_state, ext_state, batch, context):
        params = batch.get("actor_params")
        return 0.0 if params is None else 1e-3 * tree_norm(params, squared=True)

    def post_update(self, agent_state, ext_state, context):
        return agent_state, ext_state + 0.01 * jax.random.normal(context.rng)

    def eval_metrics(self, agent_state, ext_state, rng, context):
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

    def post_update(self, agent_state, ext_state, context):
        return agent_state, ext_state + 0.01 * jax.random.normal(context.rng)

    def eval_metrics(self, agent_state, ext_state, rng, context):
        return {"drift": ext_state}


def _push(obs: jax.Array) -> jax.Array:
    """DAgger's expert: 0.5 in env units on Pendulum's [-2, 2] torque."""
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


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
# A [-1, 1] action box: mapping agent actions onto the env's bounds leaves it.
_MCC: dict[str, Any] = {"env_id": "MountainCarContinuous-v0"}
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
        100,
    ),
    "AVG": (AVG, {**_SMALL, "learning_starts": 20}, 200),
    "TD3": (TD3, {**_SMALL, **_REPLAY}, 200),
    "DAgger-nudge": (
        DAgger,
        {
            "env_id": "Pendulum-v1",
            "n_envs": 2,
            "actor_architecture": ("32", "relu"),
            "batch_size": 16,
            "buffer_size": 400,
            # Fits at 100 and 200: the default fit runs once per iteration.
            "iteration_steps": 100,
            "expert_policy": _push,
            "extensions": (ActorNudge(),),
        },
        200,
    ),
    "IQL-nudge": (IQL, {**_SMALL, **_REPLAY, "extensions": (Nudge(),)}, 200),
    "TQC-nudge": (
        TQC,
        {
            **_SMALL,
            **_REPLAY,
            "num_critics": 3,
            "n_quantiles": 5,
            "extensions": (Nudge(),),
        },
        200,
    ),
    "TQC": (TQC, {**_SMALL, **_REPLAY, "num_critics": 3, "n_quantiles": 5}, 200),
    "CrossQ": (
        CrossQ,
        {
            "env_id": "Pendulum-v1",
            "n_envs": 2,
            "actor_width": 32,
            "critic_width": 32,
            **_REPLAY,
        },
        200,
    ),
    "SAC": (
        SAC,
        {**_SMALL, **_REPLAY, "policy_update_start": 40, "alpha_update_start": 40},
        200,
    ),
    # Expert prefill (40 expert steps written to replay) and the warm-up mix
    # (the expert acts 70% of the warm-up's steps).
    "SAC-expert": (
        SAC,
        {
            **_SMALL,
            **_REPLAY,
            "policy_update_start": 40,
            "alpha_update_start": 40,
            "expert_policy": _push,
            "extensions": (ExpertWarmup(fraction=0.7), ExpertPrefill(n_steps=40)),
        },
        200,
    ),
    "SAC-mcc": (
        SAC,
        {
            **_SMALL,
            **_REPLAY,
            **_MCC,
            "policy_update_start": 40,
            "alpha_update_start": 40,
        },
        200,
    ),
    "TD3-mcc": (TD3, {**_SMALL, **_REPLAY, **_MCC}, 200),
    "PPO-mcc": (
        PPO,
        {**_PENDULUM, **_MCC, "n_envs": 2, "n_steps": 32, "batch_size": 8},
        128,
    ),
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
        120,
    ),
    # Flat shuffle, GAE on the whole rollout: 4 minibatches, 2 envs.
    "PPO-flat": (PPO, {**_PENDULUM, "n_envs": 2, "n_steps": 32, "batch_size": 8}, 128),
    # One policy step and three critic steps a rollout, 2 envs.
    "REINFORCE": (
        REINFORCE,
        {**_PENDULUM, "n_envs": 2, "n_steps": 32, "value_steps": 3},
        128,
    ),
    # The env axis split, GAE per minibatch (brax's time geometry, opted
    # into: unroll_length = n_steps); brax's knobs.
    "PPO-env-split": (
        PPO,
        {
            **_PENDULUM,
            **_SPLIT,
            "unroll_length": 16,
            "squash": True,
            "log_std_state_independent": True,
            "log_std_init": 0.0,
            "normalize_observations": True,
            "ent_coef": 0.01,
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
        128,
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
        128,
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
        {**_PENDULUM, **_SPLIT, "unroll_length": 16, "extensions": (Nudge(),)},
        256,
    ),
    "APO-nudge": (
        APO,
        {
            **_PENDULUM,
            "n_envs": 2,
            "n_steps": 32,
            "num_minibatches": 2,
            "extensions": (Nudge(),),
        },
        256,
    ),
    "PQN": (PQN, _PQN, 512),
    "PQN-nudge": (PQN, {**_PQN, "extensions": (ValueNudge(),)}, 512),
    "DQN": (DQN, _DQN, 128),
    "DQN-dueling-nudge": (
        DQN,
        {**_DQN, "q_network_cls": DuelingQNetwork, "extensions": (ValueNudge(),)},
        128,
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
# Each case's recorded fingerprint. A re-recording replaces its case's entry
# and says why in its commit message (git log -p keeps the earlier values
# and their reasons).
_GOLDEN: dict[str, dict[str, float]] = {
    "APG-nudge": {
        "actor": 35.84657669067383,
        "n_updates": 5.0,
        "nudge": 0.8785082697868347,
    },
    "APG-pid-nudge": {
        "actor": 36.84217834472656,
        "n_updates": 5.0,
        "nudge": 0.8785082697868347,
    },
    "APO-nudge": {
        "actor": 9.22077751159668,
        "critic": 8.006710052490234,
        "target": 9.211957931518555,
        "n_updates": 4.0,
        "nudge": 0.8760150671005249,
    },
    "ASAC": {
        "actor": 17.27933692932129,
        "critic": 27.503591537475586,
        "target": 17.321802139282227,
        "n_updates": 0.0,
        "alpha": 0.19999998807907104,
    },
    "AVG": {
        "actor": 5.643533706665039,
        "critic": 17.76322364807129,
        "target": 5.000000476837158,
        "n_updates": 0.0,
        "alpha": 0.07000000029802322,
    },
    "CrossQ": {
        "actor": 132.8204803466797,
        "critic": 271.1706237792969,
        "target": 133.31524658203125,
        "n_updates": 91.0,
        "alpha": 0.9723634719848633,
    },
    "DAgger-nudge": {
        "actor": 4.547504901885986,
        "n_updates": 2.0,
        "nudge": 0.8891514539718628,
    },
    "DQN": {
        "actor": 8.627391815185547,
        "critic": 8.621588706970215,
        "target": 8.624855041503906,
        "n_updates": 11.0,
    },
    "DQN-dueling-nudge": {
        "actor": 9.089839935302734,
        "critic": 9.092710494995117,
        "target": 9.089208602905273,
        "n_updates": 11.0,
        "nudge": 0.8379592299461365,
    },
    "DreamerV3-drift": {
        "actor": 92.456298828125,
        "critic": 95.05998229980469,
        "world_model": 548.9129638671875,
        "n_updates": 61.0,
        "drift": 0.5100106000900269,
    },
    "IQL-nudge": {
        "actor": 7.999928951263428,
        "critic": 19.837133407592773,
        "value": 7.922294616699219,
        "target": 8.0,
        "n_updates": 91.0,
        "nudge": 1.0078387260437012,
    },
    "PPO-discrete-nudge": {
        "actor": 7.548346996307373,
        "critic": 8.936015129089355,
        "target": 8.000201225280762,
        "n_updates": 4.0,
        "nudge": 0.8760150671005249,
    },
    "PPO-env-split": {
        "actor": 6.042834758758545,
        "critic": 7.037717819213867,
        "target": 6.0001020431518555,
        "n_updates": 4.0,
    },
    "PPO-env-split-nudge": {
        "actor": 5.540170669555664,
        "critic": 6.926347732543945,
        "target": 6.0001020431518555,
        "n_updates": 4.0,
        "nudge": 0.8760150671005249,
    },
    "PPO-flat": {
        "actor": 6.084283828735352,
        "critic": 7.008105754852295,
        "target": 6.0001020431518555,
        "n_updates": 2.0,
    },
    "PPO-gru": {
        "actor": 89.6171875,
        "critic": 86.63198852539062,
        "target": 88.98634338378906,
        "n_updates": 4.0,
    },
    "PPO-mcc": {
        "actor": 3.9840002059936523,
        "critic": 4.984694957733154,
        "target": 4.000099182128906,
        "n_updates": 2.0,
    },
    "PPO-unroll": {
        "actor": 6.055914878845215,
        "critic": 6.934496879577637,
        "target": 6.0001020431518555,
        "n_updates": 2.0,
    },
    "PQN": {
        "actor": 34.021881103515625,
        "critic": 34.03536605834961,
        "target": 34.03536605834961,
        "n_updates": 8.0,
    },
    "PQN-nudge": {
        "actor": 34.021873474121094,
        "critic": 34.03536605834961,
        "target": 34.03536605834961,
        "n_updates": 8.0,
        "nudge": 0.8321770429611206,
    },
    "REDQ": {
        "actor": 10.62346363067627,
        "critic": 36.932762145996094,
        "target": 10.68629264831543,
        "n_updates": 0.0,
        "alpha": 0.9872119426727295,
    },
    "REINFORCE": {
        "actor": 7.6761860847473145,
        "critic": 8.163204193115234,
        "target": 7.68051815032959,
        "n_updates": 2.0,
    },
    "SAC": {
        "actor": 10.58867359161377,
        "critic": 17.932748794555664,
        "target": 10.68629264831543,
        "n_updates": 0.0,
        "alpha": 0.9744771122932434,
    },
    "SAC-expert": {
        "actor": 10.587984085083008,
        "critic": 17.91284942626953,
        "target": 10.68629264831543,
        "n_updates": 0.0,
        "alpha": 0.9746769666671753,
    },
    "SAC-gru": {
        "actor": 91.2584457397461,
        "critic": 150.99508666992188,
        "target": 90.58779907226562,
        "n_updates": 0.0,
        "alpha": 0.9761014580726624,
    },
    "SAC-mcc": {
        "actor": 8.320406913757324,
        "critic": 15.497808456420898,
        "target": 8.329099655151367,
        "n_updates": 0.0,
        "alpha": 0.9759957194328308,
    },
    "TD3": {
        "actor": 15.418506622314453,
        "critic": 27.09539031982422,
        "target": 15.265251159667969,
        "n_updates": 91.0,
    },
    "TD3-mcc": {
        "actor": 17.516809463500977,
        "critic": 28.63946533203125,
        "target": 17.338119506835938,
        "n_updates": 91.0,
    },
    "TDMPC2-drift": {
        "actor": 32.238319396972656,
        "world_model": 162.49891662597656,
        "n_updates": 39.0,
        "drift": 0.5062060952186584,
    },
    "TDMPC2MultiTask-drift": {
        "actor": 32.23353576660156,
        "world_model": 161.12286376953125,
        "n_updates": 12.0,
        "drift": 0.4896028935909271,
    },
    "TQC": {
        "actor": 17.295066833496094,
        "critic": 45.290157318115234,
        "target": 17.321802139282227,
        "n_updates": 0.0,
        "alpha": 0.972418487071991,
    },
    "TQC-nudge": {
        "actor": 17.15433692932129,
        "critic": 43.00996398925781,
        "target": 17.321802139282227,
        "n_updates": 0.0,
        "alpha": 0.9724184274673462,
        "nudge": 0.8450943231582642,
    },
    "UDRL-nudge": {
        "actor": 21.741743087768555,
        "critic": 23.000001907348633,
        "target": 22.000200271606445,
        "n_updates": 4.0,
        "nudge": 0.8760150671005249,
    },
}


def fingerprint(name: str) -> dict[str, float]:
    agent_cls, kwargs, steps = _CASES[name]
    agent = agent_cls(**kwargs)
    state, _ = agent.train(seed=0, n_timesteps=steps, logging_config=None)
    out = {"actor": float(tree_norm(state.actor_state.params, squared=True))}
    if state.critic_state is not None:
        out["critic"] = float(tree_norm(state.critic_state.params, squared=True))
    if getattr(state, "world_model_state", None) is not None:
        out["world_model"] = float(
            tree_norm(state.world_model_state.params, squared=True)
        )
    if getattr(state, "value_state", None) is not None:  # IQL's V
        out["value"] = float(tree_norm(state.value_state.params, squared=True))
    if getattr(state.actor_state, "target_params", None) is not None:
        out["target"] = float(tree_norm(state.actor_state.target_params, squared=True))
    if getattr(state, "n_updates", None) is not None:
        out["n_updates"] = float(jnp.asarray(state.n_updates).reshape(-1)[0])
    if hasattr(state, "alpha"):
        out["alpha"] = float(jnp.exp(state.alpha.params["log_alpha"]).reshape(-1)[0])
    # The learner's extensions hold the states; a controller's (ExpertWarmup)
    # or a stateless one (ExpertPrefill) records nothing.
    extensions = agent.extension_stack.extensions
    for ext, ext_state in zip(extensions, state.ext_state, strict=True):
        leaves = jnp.asarray(ext_state).reshape(-1)
        if leaves.size:
            out[ext.name] = float(leaves[0])
    return out


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_tiny_run_reproduces_its_recorded_fingerprint(name: str) -> None:
    got = fingerprint(name)
    for key, golden in _GOLDEN[name].items():
        rel = abs(got[key] - golden) / max(abs(golden), 1.0)
        assert rel < _TOL, f"{name} {key}: got {got[key]!r}, golden {golden!r}"
