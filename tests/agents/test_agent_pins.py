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
from optax.tree_utils import tree_norm

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
        return 0.0 if params is None else 1e-3 * tree_norm(params, squared=True)

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        params = batch.get("actor_params")
        return 0.0 if params is None else 1e-3 * tree_norm(params, squared=True)

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
        200,
    ),
    "AVG": (AVG, {**_SMALL, "learning_starts": 20}, 200),
    "TD3": (TD3, {**_SMALL, **_REPLAY}, 200),
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
        256,
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
        200,
    ),
    # Flat shuffle, GAE on the whole rollout: 4 minibatches, 2 envs.
    "PPO-flat": (PPO, {**_PENDULUM, "n_envs": 2, "n_steps": 32, "batch_size": 8}, 256),
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
        {**_PENDULUM, **_SPLIT, "unroll_length": 16, "extensions": (Nudge(),)},
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
# AVG (final observation normalised) and PPO-env-split (PPO trains on
# observations as normalised at collection, on the plain-MLP default)
# re-recorded once batch/env-eval-logging met main: -0.6%/+0.7%, +0.3%/+0.5%.
# Recorded on macOS ARM CPU at the parent of the step-16 lineage commit.
_GOLDEN: dict[str, dict[str, float]] = {
    "AVG": {
        "actor": 5.839523792266846,
        "critic": 19.103961944580078,
        "alpha": 0.07000000029802322,
    },
}
# Recorded on macOS ARM CPU once TD3's critic dropped its encoder LayerNorm:
# the critic tree lost the norm's scales (64 of the old 79.6) and the actor
# moved 3.6%.
_GOLDEN |= {"TD3": {"actor": 3.94071626663208, "critic": 11.506895065307617}}
# Recorded on macOS ARM CPU once a recurrent Q-critic embedded the
# observation and the previous action before its memory and read the
# current one at a head after it: new critic parameters (against the
# head-only critic before it: +42.9% critic, +0.17% actor, -0.045% alpha).
_GOLDEN |= {}
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
    "SAC-gru": {
        "actor": 57.405818939208984,
        "critic": 134.20553588867188,
        "alpha": 0.9515719413757324,
    },
    "PPO-flat": {"actor": 4.138375282287598, "critic": 4.506361961364746},
    "PPO-env-split": {"actor": 3.0352354049682617, "critic": 4.100451469421387},
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
# Recorded on macOS ARM CPU once ASAC's temperature and theta steps read one
# sample of the updated policy (one actor pass fewer, one key fewer per
# update): actor +0.034%, critic -0.015%, alpha +0.037%.
_GOLDEN |= {
    "ASAC": {
        "actor": 4.501960754394531,
        "critic": 9.954827308654785,
        "alpha": 0.9729517102241516,
    },
}
# Recorded on macOS ARM CPU once REDQ's critic loss summed each critic's
# error and its actor read the critics from before their last update, both
# as REDQ's official code. The second moved the actor -7.1e-6, the critic
# -4.0e-7 and alpha +6.1e-8 (relative).
_GOLDEN |= {
    "REDQ": {
        "actor": 4.491237640380859,
        "critic": 23.677995681762695,
        "alpha": 0.9729598164558411,
    },
}
# Recorded on macOS ARM CPU before step 17 moved the world models onto
# TrainLoop and tidied APG and UDRL (the world models once they took
# TrainLoop's init and post_update keys).
_GOLDEN |= {
    "DreamerV3-drift": {
        "actor": 92.46167755126953,
        "critic": 95.06945037841797,
        "world_model": 548.8908081054688,
        "drift": 0.23474201560020447,
    },
    "TDMPC2MultiTask-drift": {
        "actor": 32.23477554321289,
        "world_model": 161.12486267089844,
        "drift": 0.2877737283706665,
    },
}
# Recorded on macOS ARM CPU once UDRL's sampler drew intervals across
# segment boundaries (actor -1.3e-4; the nudge unchanged). UDRL's critic
# and its actor's target are never-updated copies: not pinned.
_GOLDEN |= {"UDRL-nudge": {"actor": 23.968936920166016, "nudge": 0.3160724639892578}}
# Recorded on macOS ARM CPU once prepare_env kept the env_params given with
# an env id: TDMPC2-drift had run on Pendulum's default T = 200, so its 40
# steps ended before the seed tick and it pinned no update (actor +0.11 %,
# world model +1.0 %, drift -27 % on re-recording). With T = 10 it runs the
# burst of 20 and 19 single updates.
_GOLDEN |= {
    "TDMPC2-drift": {
        "actor": 32.20726776123047,
        "world_model": 162.42007446289062,
        "drift": 0.2332032173871994,
        "n_updates": 39,
    },
}
# Recorded on macOS ARM CPU once APG took TrainLoop's init and post_update
# keys, before step 17 moved it onto the shared evaluation.
_GOLDEN |= {
    "APG-nudge": {"actor": 35.82789611816406, "nudge": 0.27262088656425476},
    "APG-pid-nudge": {"actor": 36.82305145263672, "nudge": 0.27262088656425476},
}

# Recorded on macOS ARM CPU once the action map took its centre/half-range
# form, exact on a [-1, 1] box: on MountainCarContinuous's, mapping the
# agents' actions onto the env's bounds must leave these unchanged.
_GOLDEN |= {
    "SAC-mcc": {
        "actor": 3.1000170707702637,
        "critic": 8.203842163085938,
        "alpha": 0.9754154682159424,
    },
    "TD3-mcc": {"actor": 2.397678852081299, "critic": 7.9359588623046875},
    "PPO-mcc": {"actor": 3.6173574924468994, "critic": 3.1998236179351807},
}
# Re-recorded on macOS ARM CPU once get_action_and_log_probs lost its dead
# uniform draw (0 * u + 1 * a): XLA compiles the GRU collector without it,
# and the actor moved by float rounding alone (-1.4e-7; 53.321449279785156
# before). Every other pin is bitwise unchanged.
_GOLDEN |= {"PPO-gru": {"actor": 53.321441650390625, "critic": 52.78514099121094}}
# Re-recorded on macOS ARM CPU once every continuous agent's [-1, 1] action
# was mapped onto the env's box: on Pendulum's [-2, 2] the torque, the
# uniform warm-up and TD3's noise double, PPO's Gaussian is read in
# normalised units and APG's gradients scale by 2. Relative shifts (actor,
# critic): AVG -2.63%, -2.39%; TD3 -0.61%, -0.36%; PPO-flat -0.52%,
# -0.15%; PPO-env-split -0.52%, +0.08%; PPO-unroll -0.42%, -0.31%;
# PPO-env-split-nudge +0.29%, +0.12%; PPO-gru +0.08%, -0.11%; ASAC -0.13%,
# -0.04%; REDQ -0.10%, -0.15%; SAC-gru -0.05%, +0.13%; APO-nudge +0.003%,
# +0.18%; APG-nudge and APG-pid-nudge actor +0.03%. Alphas and nudges move
# under 1e-5. Every [-1, 1]-box and discrete pin (the -mcc ones, CartPole's,
# the world models') is bitwise unchanged.
_GOLDEN |= {
    "AVG": {
        "actor": 5.686222076416016,
        "critic": 18.646684646606445,
        "alpha": 0.07000000029802322,
    },
    "TD3": {"actor": 3.9167749881744385, "critic": 11.465590476989746},
    "PPO-flat": {"actor": 4.116726398468018, "critic": 4.499772548675537},
    "PPO-env-split": {"actor": 3.0195863246917725, "critic": 4.1037750244140625},
    "PPO-unroll": {"actor": 4.207005023956299, "critic": 4.5349884033203125},
    "PPO-env-split-nudge": {
        "actor": 3.753653049468994,
        "critic": 4.229918479919434,
        "nudge": 0.3099551498889923,
    },
    "PPO-gru": {"actor": 53.36158752441406, "critic": 52.72870635986328},
    "ASAC": {
        "actor": 4.495896339416504,
        "critic": 9.951025009155273,
        "alpha": 0.972956120967865,
    },
    "REDQ": {
        "actor": 4.486619472503662,
        "critic": 23.64264488220215,
        "alpha": 0.9729592204093933,
    },
    "SAC-gru": {
        "actor": 57.37752151489258,
        "critic": 134.38070678710938,
        "alpha": 0.95156329870224,
    },
    "APO-nudge": {
        "actor": 3.852809429168701,
        "critic": 4.166390419006348,
        "nudge": 0.33622848987579346,
    },
    "APG-nudge": {"actor": 35.83795928955078, "nudge": 0.27262088656425476},
    "APG-pid-nudge": {"actor": 36.834327697753906, "nudge": 0.27262088656425476},
}
# Re-recorded on macOS ARM CPU once APG's Adam eps became torch's 1e-8 (was
# 1e-5). Relative shifts (actor): APG -2.4e-5, APG-pid -2.4e-5.
_GOLDEN |= {
    "APG-nudge": {"actor": 35.83708953857422, "nudge": 0.27262088656425476},
    "APG-pid-nudge": {"actor": 36.83342742919922, "nudge": 0.27262088656425476},
}
# Re-recorded on macOS ARM CPU once PPO's default log-std became its
# reference code's: one unclipped parameter per action dimension at 0, not
# a Dense head at -1. The actor loses the head's kernel; relative shifts
# (actor, critic): PPO-mcc -44.1%, +3.9%; PPO-flat -27.0%, -0.55%;
# PPO-unroll -25.4%, -0.72%; PPO-env-split-nudge -29.8%, -0.05%; PPO-gru
# -2.0%, -0.12%. PPO-env-split (the head set explicitly) is bitwise
# unchanged, PPO-discrete-nudge within 1e-5.
_GOLDEN |= {
    "PPO-mcc": {"actor": 2.020819902420044, "critic": 3.3231139183044434},
    "PPO-flat": {"actor": 3.0036025047302246, "critic": 4.475032806396484},
    "PPO-unroll": {"actor": 3.136448621749878, "critic": 4.502480506896973},
    "PPO-env-split-nudge": {
        "actor": 2.6335089206695557,
        "critic": 4.227860450744629,
        "nudge": 0.3099551498889923,
    },
    "PPO-gru": {"actor": 52.301292419433594, "critic": 52.665245056152344},
}
# Re-recorded on macOS ARM CPU once feedforward PPO shuffled its rollout flat
# by default (GAE once, as in the paper): with 2 envs and 2 minibatches this
# run took brax's time geometry before. Actor +0.52%, critic +0.98%, nudge
# +6.6%. Every other PPO pin is unchanged.
_GOLDEN |= {
    "PPO-discrete-nudge": {
        "actor": 5.850821495056152,
        "critic": 5.0791544914245605,
        "nudge": 0.31162577867507935,
    },
}
# Re-recorded on macOS ARM CPU once APO seeded its reward rate and value
# bias with the first rollout's means, as xtma/apo does, instead of an EMA
# from 0: actor -0.67%, critic +1.64%, nudge unchanged.
_GOLDEN |= {
    "APO-nudge": {
        "actor": 3.8270583152770996,
        "critic": 4.234889984130859,
        "nudge": 0.33622848987579346,
    },
}
# Re-recorded on macOS ARM CPU once AVG scaled its TD error as the authors'
# TDErrorScaler does (a 1e-4 floor, scaled from the reward's variance before
# any return, each return counted once): actor +0.41%, critic +6.4%.
_GOLDEN |= {
    "AVG": {
        "actor": 5.7092366218566895,
        "critic": 19.835189819335938,
        "alpha": 0.07000000029802322,
    },
}
# Re-recorded on macOS ARM CPU once the discrete logits head started at
# orthogonal(0.01), not 1.0: the head's kernel loses most of its initial
# squared norm (2.0 at gain 1 for 16 x 2). Relative shifts: PPO-discrete-nudge
# (from the flat-geometry pin above) actor -39.9%, critic +0.78%; UDRL-nudge
# actor -9.3%; both nudges unchanged.
_GOLDEN |= {
    "PPO-discrete-nudge": {
        "actor": 3.514505386352539,
        "critic": 5.118714809417725,
        "nudge": 0.31162577867507935,
    },
    "UDRL-nudge": {"actor": 21.728490829467773, "nudge": 0.3160724639892578},
}

# Re-recorded on macOS ARM CPU once PPO's hidden layers became Baselines' ppo2's:
# orthogonal with gain sqrt(2) (were orthogonal(1.0)); PPO-gru, recurrent,
# keeps its. Relative shifts (actor, critic): PPO-mcc +99.2%, +58.9%;
# PPO-flat +99.9%, +66.2%; PPO-env-split +99.3%, +74.2%; PPO-unroll +96.8%,
# +66.7%; PPO-discrete-nudge +105.4%, +76.8%; PPO-env-split-nudge +103.5%,
# +71.4%. The nudges are bitwise unchanged.
_GOLDEN |= {
    "PPO-mcc": {"actor": 4.02446985244751, "critic": 5.281169891357422},
    "PPO-flat": {"actor": 6.003925800323486, "critic": 7.437717914581299},
    "PPO-env-split": {"actor": 6.0180745124816895, "critic": 7.147911071777344},
    "PPO-unroll": {"actor": 6.1724958419799805, "critic": 7.504622459411621},
    "PPO-discrete-nudge": {
        "actor": 7.218266487121582,
        "critic": 9.050727844238281,
        "nudge": 0.31162577867507935,
    },
    "PPO-env-split-nudge": {
        "actor": 5.358870983123779,
        "critic": 7.248394966125488,
        "nudge": 0.3099551498889923,
    },
}

# Re-recorded on macOS ARM CPU once recurrent PPO's hidden layers became
# PureJaxRL's recurrent PPO's: orthogonal with gain sqrt(2) (were
# orthogonal(1.0)). PPO-gru moved +5.6% (actor), +5.6% (critic).
_GOLDEN |= {"PPO-gru": {"actor": 55.2531852722168, "critic": 55.617679595947266}}
# Re-recorded on macOS ARM CPU once PPO's optimiser became ppo2's: Adam eps
# 1e-5 (was 1e-8), one clip at 0.5 over both networks (was each its own;
# PPO-unroll's fused_grad_clip, which added a joint clip at 1.0, is gone)
# and vf_coef 0.5 (was 1.0). Relative shifts (actor, critic): PPO-mcc -0.19%,
# -0.34%; PPO-flat -0.21%, -0.02%; PPO-env-split -0.23%, -0.003%; PPO-unroll
# -1.9%, -0.03%; PPO-gru -0.04%, +0.001%; PPO-discrete-nudge +1.2%, -0.19%;
# PPO-env-split-nudge +5.9%, -0.02%; the nudges are unchanged.
_GOLDEN |= {
    "PPO-mcc": {"actor": 4.016942501068115, "critic": 5.263054847717285},
    "PPO-flat": {"actor": 5.99110221862793, "critic": 7.436330318450928},
    "PPO-env-split": {"actor": 6.004455089569092, "critic": 7.147724151611328},
    "PPO-unroll": {"actor": 6.052582740783691, "critic": 7.502388000488281},
    "PPO-gru": {"actor": 55.23348617553711, "critic": 55.618282318115234},
    "PPO-discrete-nudge": {
        "actor": 7.301796913146973,
        "critic": 9.033950805664062,
        "nudge": 0.31162577867507935,
    },
    "PPO-env-split-nudge": {
        "actor": 5.6761155128479,
        "critic": 7.247241973876953,
        "nudge": 0.3099551498889923,
    },
}
# Re-recorded on macOS ARM CPU once APO's continuous actor became its
# reference code's: an unsquashed Gaussian around tanh of the mean layer,
# its log-std unclipped (the same parameters). Relative shifts (actor,
# critic): APO-nudge +0.042%, +0.0066%; its nudge is bitwise unchanged.
_GOLDEN |= {
    "APO-nudge": {
        "actor": 3.8544418811798096,
        "critic": 4.166664123535156,
        "nudge": 0.33622848987579346,
    },
}
# Re-recorded on macOS ARM CPU once APO's hidden layers became its reference
# code's: two of 64 tanh units (were 128), starting as torch's nn.Linear
# (kernel and bias uniform in +-1/sqrt(fan_in); were orthogonal, zero bias).
# Relative shifts (actor, critic): APO-nudge +137%, +91%; its nudge is
# bitwise unchanged.
_GOLDEN |= {
    "APO-nudge": {
        "actor": 9.124271392822266,
        "critic": 7.960622310638428,
        "nudge": 0.33622848987579346,
    },
}
# Re-recorded on macOS ARM CPU once APO's optimiser became its reference
# code's: one step of both networks on their summed loss, the gradient
# clipped to norm 10 over both at once (was each network's own Adam step,
# clipped to 0.5 on its own). Relative shifts (actor, critic): APO-nudge
# -0.087%, +0.0016%; its nudge is bitwise unchanged.
_GOLDEN |= {
    "APO-nudge": {
        "actor": 9.116347312927246,
        "critic": 7.960748672485352,
        "nudge": 0.33622848987579346,
    },
}
# Re-recorded on macOS ARM CPU once APO's entropy bonus became its reference
# code's, ent_coef 0.01 (was 0). Relative shifts (actor, critic): APO-nudge
# -3.0e-6, +1.1e-6; its nudge is bitwise unchanged.
_GOLDEN |= {
    "APO-nudge": {
        "actor": 9.11631965637207,
        "critic": 7.960757732391357,
        "nudge": 0.33622848987579346,
    },
}
# Re-recorded on macOS ARM CPU on the batch that brought APO's reference stack
# onto main (batch/apo-reference). The stack's pins above were recorded on
# its base, 04e2249; of main's commits since, only b35fa54 (APO seeds its
# reward rate and value bias with the first rollout's means) re-recorded
# APO-nudge on main. On the merged tree APO-nudge moves +0.97% (actor) and
# +1.21% (critic) from the entropy-bonus pin, its nudge bitwise unchanged. PPO-mcc's critic moves
# -1.8e-7 (relative), float32 rounding, as PPO's joint clip took the shared
# formula g * (max_norm / norm) for optax's (g / norm) * max_norm.
_GOLDEN |= {
    "APO-nudge": {
        "actor": 9.204462051391602,
        "critic": 8.057476043701172,
        "nudge": 0.33622848987579346,
    },
    "PPO-mcc": {"actor": 4.016942501068115, "critic": 5.263053894042969},
}

# Re-recorded on macOS ARM CPU once AVG's layers became its reference code's:
# every kernel orthogonal with gain 1 and every bias zero, the mean (was
# orthogonal(0.01)) and log-std head (was a zero kernel, bias -1) included.
# Relative shifts (actor, critic), from the golden with the authors' TD-error
# scaling: AVG -2.64%, -6.00%; alpha unchanged.
_GOLDEN |= {
    "AVG": {
        "actor": 5.558319091796875,
        "critic": 18.645687103271484,
        "alpha": 0.07000000029802322,
    },
}

# Re-recorded on macOS ARM CPU once ASAC's layers became its reference code's:
# torch's nn.Linear default, every kernel and bias uniform in +-1/sqrt(fan_in)
# (were orthogonal(1.0) hidden and Q layers, an orthogonal(0.01) mean and a
# zero-kernel log-std head with bias -1, zero biases). Relative shifts
# (actor, critic, alpha): ASAC +283.9%, +171.6%, -0.021%.
_GOLDEN |= {
    "ASAC": {
        "actor": 17.258983612060547,
        "critic": 27.022438049316406,
        "alpha": 0.9727535247802734,
    },
}
# Re-recorded on macOS ARM CPU once ASAC fixed its temperature at 0.2 by
# default (was learned from 1.0). Relative shifts (actor, critic, alpha):
# ASAC -0.1%, +0.0%, -79.4%.
_GOLDEN |= {
    "ASAC": {
        "actor": 17.239709854125977,
        "critic": 27.024330139160156,
        "alpha": 0.19999998807907104,
    },
}
# Re-recorded on macOS ARM CPU once ASAC's learning rates (actor 1e-4,
# critics 5e-4) and gradient clip (10, each critic alone) became its
# authors'. Relative shifts (actor, critic, alpha): +0.24%, +1.27%, 0.
_GOLDEN |= {
    "ASAC": {
        "actor": 17.280670166015625,
        "critic": 27.36642837524414,
        "alpha": 0.19999998807907104,
    },
}

# Re-recorded on macOS ARM CPU once PQN's layers became its reference code's:
# flax's lecun_normal kernels (were orthogonal(sqrt(2)) hidden layers and an
# orthogonal(1.0) head), zero biases. Relative shifts (actor): PQN +30.9%,
# PQN-nudge +30.9%; its nudge is bitwise unchanged.
_GOLDEN |= {
    "PQN": {"actor": 33.92633056640625},
    "PQN-nudge": {"actor": 33.91205596923828, "nudge": 0.33566388487815857},
}

# Re-recorded on macOS ARM CPU once TD3's layers became its reference code's:
# torch's nn.Linear default, every kernel and bias uniform in +-1/sqrt(fan_in)
# (were orthogonal(1.0) hidden and Q layers and a uniform(+-3e-3) actor head,
# zero biases). Relative shifts (actor, critic): TD3 +299.6%, +153.4%;
# TD3-mcc +640.8%, +258.6%.
_GOLDEN |= {
    "TD3": {"actor": 15.652060508728027, "critic": 29.05111312866211},
    "TD3-mcc": {"actor": 17.76249885559082, "critic": 28.460254669189453},
}
# Re-recorded on macOS ARM CPU once TD3's defaults became its authors' 2019
# code: the run takes the default learning rate, now 3e-4 (was 1e-3).
# Relative shifts (actor, critic): TD3 -1.4%, -6.8%; TD3-mcc -1.5%, +0.4%.
_GOLDEN |= {
    "TD3": {"actor": 15.433494567871094, "critic": 27.067062377929688},
    "TD3-mcc": {"actor": 17.501985549926758, "critic": 28.56537628173828},
}

# Re-recorded on macOS ARM CPU once REDQ's layers became its reference code's:
# xavier_uniform kernels and zero biases, the mean and log-std layers included
# (were orthogonal(1.0) hidden and Q layers, an orthogonal(0.01) mean and a
# zero-kernel log-std head with bias -1). Relative shifts (actor, critic,
# alpha): REDQ +135.4%, +66.0%, -0.079%.
_GOLDEN |= {
    "REDQ": {
        "actor": 10.559710502624512,
        "critic": 39.23777389526367,
        "alpha": 0.9721929430961609,
    },
}

# Re-recorded on macOS ARM CPU once SAC's layers became its reference code's:
# xavier_uniform kernels and zero biases, the mean and log-std heads included
# (were orthogonal(1.0) hidden and Q layers, an orthogonal(0.01) mean and a
# zero-kernel log-std head with bias -1). Relative shifts (actor, critic,
# alpha): SAC-mcc +168.1%, +90.5%, +0.059%. SAC-gru (recurrent) keeps its.
_GOLDEN |= {
    "SAC-mcc": {
        "actor": 8.31135368347168,
        "critic": 15.628474235534668,
        "alpha": 0.9759861826896667,
    },
}

# Re-recorded on macOS ARM CPU once DQN's layers became its reference code's:
# Torch7's nn.Linear default, kernel and bias uniform in +-1/sqrt(fan_in)
# (were orthogonal(1.0), zero biases), the dueling heads too. Relative shifts
# (actor, target): DQN +27.3%, +28.2%; DQN-dueling-nudge +29.4%, +29.2%; its
# nudge is bitwise unchanged.
_GOLDEN |= {
    "DQN": {"actor": 12.04747200012207, "target": 11.927078247070312},
    "DQN-dueling-nudge": {
        "actor": 13.111983299255371,
        "target": 12.902071952819824,
        "nudge": 0.22634370625019073,
    },
}


def fingerprint(name: str) -> dict[str, float]:
    agent_cls, kwargs, steps = _CASES[name]
    state, _ = agent_cls(**kwargs).train(seed=0, n_timesteps=steps)
    out = {"actor": float(tree_norm(state.actor_state.params, squared=True))}
    if state.critic_state is not None:
        out["critic"] = float(tree_norm(state.critic_state.params, squared=True))
    if getattr(state, "world_model_state", None) is not None:
        out["world_model"] = float(
            tree_norm(state.world_model_state.params, squared=True)
        )
    if getattr(state.actor_state, "target_params", None) is not None:
        out["target"] = float(tree_norm(state.actor_state.target_params, squared=True))
    if getattr(state, "n_updates", None) is not None:
        out["n_updates"] = float(jnp.asarray(state.n_updates).reshape(-1)[0])
    if hasattr(state, "alpha"):
        out["alpha"] = float(jnp.exp(state.alpha.params["log_alpha"]).reshape(-1)[0])
    for ext, ext_state in zip(
        kwargs.get("extensions", ()), state.ext_state, strict=False
    ):
        out[ext.name] = float(jnp.asarray(ext_state).reshape(-1)[0])
    return out


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_tiny_run_reproduces_its_recorded_fingerprint(name: str) -> None:
    got = fingerprint(name)
    for key, golden in _GOLDEN[name].items():
        rel = abs(got[key] - golden) / max(abs(golden), 1.0)
        assert rel < _TOL, f"{name} {key}: got {got[key]!r}, golden {golden!r}"
