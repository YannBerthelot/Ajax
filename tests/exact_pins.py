"""Exact before/after pins for a behaviour-neutral restructure (a tool run by
hand, outside CI): tiny runs whose every final-state leaf and logged
evaluation is saved, with the text of their training program's jaxpr. On one
machine, a restructure that changes nothing reproduces both exactly::

    JAX_PLATFORMS=cpu uv run python -m tests.exact_pins record DIR [CASE ...]
    JAX_PLATFORMS=cpu uv run python -m tests.exact_pins compare DIR [CASE ...]

``compare`` prints per case BITWISE or each differing leaf's largest relative
gap, and whether the program's text is the same. test_agent_pins'
fingerprints hold to 1e-3 across platforms; these hold to the bit on one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from optax.tree_utils import tree_norm

from ajax import APO, ASAC, AVG, DQN, IQL, PPO, PQN, REDQ, SAC, TD3, TQC, CrossQ, DAgger
from ajax.agents.APG.APG import APG
from ajax.agents.APG.networks import PIDHeadConfig
from ajax.agents.critics import compute_mse_td_loss, make_huber_td_loss
from ajax.agents.DQN.DQN import compute_double_dqn_td_target
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.TD3.TD3 import make_default_action_pipeline
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.agents.UDRL.UDRL import UDRL
from ajax.buffers.dataset import TransitionDataset
from ajax.environments.system_class import UniformPerturbation
from ajax.extensions.adversarial import AdversarialReward
from ajax.extensions.base import Extension
from ajax.extensions.expert import (
    ExpertGains,
    ExpertWarmup,
    ImitationLoss,
    JSRLCurriculum,
    ResidualPolicy,
    StorePolicyAction,
)
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.observe import (
    ExpertActionInput,
    ExpertStateInput,
    TrainingProgressInput,
)
from ajax.extensions.offline import TargetBC
from ajax.extensions.pretrain import BellmanPretrain, ExpertPrefill, MCPretrain
from ajax.extensions.target_mods import (
    IBRL,
    LCBGatedBootstrap,
    MCVarianceCorrection,
    ValueBox,
)
from ajax.logging.wandb_logging import LoggingConfig
from ajax.modules.pid_actor import PIDPolicy
from ajax.networks.memory import MemoryConfig
from ajax.networks.q_networks import DuelingQNetwork
from ajax.schedule import linear
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY
from tests.agents.TDMPC2.common import TINY_AGENT as TDMPC2_TINY
from tests.agents.test_offline import episodes
from tests.modules import setpoint_env

SEEDS = [0, 1]


@dataclass(frozen=True)
class Keyed(Extension):
    """Draws on ``context.rng`` in every phase: a moved key moves the numbers."""

    name: str = "keyed"

    def init_state(self, agent_state, rng):
        return jax.random.normal(rng)

    def on_target(self, agent_state, ext_state, batch, target, context):
        return target + 0.01 * jax.random.normal(context.rng)

    def critic_loss(self, agent_state, ext_state, batch, context):
        params = batch.get("critic_params", batch.get("q_state"))
        if params is None:
            return 0.0
        params = getattr(params, "params", params)
        return (
            1e-3
            * (1 + jax.random.normal(context.rng))
            * tree_norm(params, squared=True)
        )

    def actor_loss(self, agent_state, ext_state, batch, context):
        draw = jax.random.normal(context.rng)
        loss = 1e-3 * (1 + draw) * tree_norm(batch["actor_params"], squared=True)
        return loss, {"keyed/draw": draw}

    def post_update(self, agent_state, ext_state, context):
        return agent_state, ext_state + 0.01 * jax.random.normal(context.rng)

    def eval_metrics(self, agent_state, ext_state, rng, context):
        return {"keyed": ext_state}


@dataclass(frozen=True)
class RewardKeyed(Keyed):
    """Keyed, and a rollout's rewards noised on ``context.rng`` (the reward fold)."""

    def reward(self, agent_state, ext_state, batch, rewards, context):
        noise = 0.01 * jax.random.normal(context.rng, rewards.shape)
        return rewards + noise, ext_state, {}


@dataclass(frozen=True)
class ValueKeyed(Keyed):
    actor_loss = Extension.actor_loss


@dataclass(frozen=True)
class ActorKeyed(Keyed):
    on_target = Extension.on_target
    critic_loss = Extension.critic_loss


def push(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


def choose_one(obs: jax.Array) -> jax.Array:
    return jnp.ones(obs.shape[:-1] + (1,), jnp.int32)


class GainExpert:
    """A stateful expert with a learnable gain, in Pendulum's torque units."""

    learnable_fields, anchor_gains = ("kp",), jnp.array([1.0])

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, *args: Any) -> Any:
        state, obs = args[-2:]
        out = self.step_with_gains(state, obs, self.anchor_gains)
        return out if len(args) == 2 else out[0]

    def step_with_gains(self, state: jax.Array, obs: jax.Array, gains: Any) -> tuple:
        return jnp.clip(-gains[0] * obs[..., :1], -2.0, 2.0), state + obs[..., :1]


S: dict[str, Any] = {"env_id": "Pendulum-v1", "n_envs": 2}
S.update({"actor_architecture": ("32", "relu"), "critic_architecture": ("32", "relu")})
R = {"buffer_size": 1000, "batch_size": 16, "learning_starts": 20}
SAC40 = {**S, **R, "policy_update_start": 40, "alpha_update_start": 40}
ON = {"env_id": "Pendulum-v1", "n_envs": 2, "n_steps": 32, "batch_size": 16}
ON.update({"actor_architecture": ("16", "tanh"), "critic_architecture": ("16", "tanh")})
CART = {"env_id": "CartPole-v1", "n_envs": 2, **R, "architecture": ("16", "relu")}
PQN_KW = {"env_id": "CartPole-v1", "n_envs": 4, "n_steps": 16, "num_minibatches": 2}
PQN_KW.update({"n_epochs": 2, "architecture": ("16", "relu")})
SETPOINT = dict(zip(("env_id", "env_params"), setpoint_env.make(), strict=True))
PID = {**SETPOINT, "n_envs": 2, "extensions": (setpoint_env.TERMS,)}
PID.update({"actor_architecture": PIDPolicy(), "critic_architecture": ("16", "relu")})
NORM = {"normalize_observations": True}
GRU = {"learning_starts": 60, "memory": MemoryConfig(kind="gru", hidden_size=8)}
DAGGER = {"env_id": "Pendulum-v1", "n_envs": 2, "actor_architecture": ("32", "relu")}
DAGGER.update({"buffer_size": 1000, "batch_size": 16, "expert_policy": push})
CROSSQ = {"env_id": "Pendulum-v1", "n_envs": 2, "actor_width": 32, "critic_width": 64}
CROSSQ.update({**R, "learning_starts": 40})
# The paths a replay agent's options take: a residual policy (its executed
# and chosen actions), an expert's stored actions and imitation, gain mode,
# observation features (one reading the progress), the normalisers.
RESIDUAL = {"expert_policy": push, "extensions": (ResidualPolicy(), Keyed())}
BC = {"expert_policy": push, "extensions": (ImitationLoss(push, coef=1.0),)}
GAIN = {"expert_policy": GainExpert(), "extensions": (ExpertGains(),)}
EXPERT_PREFILL = {"expert_policy": push, "extensions": (ExpertPrefill(n_steps=40),)}
# The warm-up's expert mix through TD3's explore pipeline and the default one.
WARMUP = {"expert_policy": push, "extensions": (ExpertWarmup(fraction=0.5),)}
OBSERVE: dict[str, Any] = {"expert_policy": push}
OBSERVE.update({"extensions": (ExpertActionInput(), TrainingProgressInput())})
REWARD_NORM = {"normalize_rewards": True, "reward_normalization_gamma": 0.99}
OFFLINE_KW = {**S, **R, "learning_starts": 0, "batch_size": 8}
KEYED = {"extensions": (Keyed(),)}
TIME_RESET = {**ON, "n_envs": 4, "n_steps": 16, "num_minibatches": 2}
TIME_RESET.update(
    {"unroll_length": 16, "num_resets_per_eval": 1, "num_evals": 3, **NORM}
)
PPO_GRU = {"memory": MemoryConfig(kind="gru", hidden_size=8)}
# SAC's expert options: its replay prefill and its warm-up mix, alone and
# with what runs around them: the acting gates before and after the warm-up
# choice, a pretraining reading the prefill, and the actor loss reading
# a_expert. The cases with an expert and no action transform mix 0.7 of the
# warm-up, and those without a prefill of their own prefill 20 000 steps,
# as SAC did by default with an expert when they were recorded.
SAC_EXPERT = {**SAC40, "expert_policy": push}
MIX, PREFILL40 = ExpertWarmup(fraction=0.7), ExpertPrefill(n_steps=40)
DEFAULT_PREFILL = ExpertPrefill(n_steps=20_000)
SMALL_MC = MCPretrain(
    push, n_mc_steps=200, n_mc_episodes=4, n_steps=20, online_light_steps=5
)
GATES = (SMALL_MC, JSRLCurriculum(episode_length=8, decay_frac=0.5), ValueBox())
BELLMAN = (BellmanPretrain(n_steps=8),)
RESIDUAL_BC = (ResidualPolicy(), ImitationLoss(push, coef=1.0))
GAIL_DATA = episodes(3, discrete=False)  # Pendulum's observations
#: Trained offline for their steps (updates on a dataset's rows), then
#: online for ONLINE steps from that state (train_offline, then train).
OFFLINE = {"IQL-offline", "TD3-offline"}
OFFLINE.update({"SAC-offline", "ASAC-offline", "REDQ-offline", "TQC-offline"})
OFFLINE.update({"DQN-offline"})
ONLINE = 40
#: Evaluated only in the last fifth of training (LoggingConfig.sweep).
SWEEP = {"SAC-sweep"}
# The SAC family's paths its declaration on the replay and incremental
# families moves (B4): each agent's extension phases, memory, expert
# options, normalisers and offline training, and an env whose actions
# have two dimensions (every other case's have one).
REACHER = {"env_id": "Reacher-misc"}
TQC_KW = {**S, **R, "num_critics": 3, "n_quantiles": 5}
AVG_KW = {**S, "learning_starts": 20}
SAC_FAMILY_CASES: dict[str, tuple[Any, dict[str, Any], int]] = {
    "SAC-offline": (SAC, {**OFFLINE_KW, "policy_update_start": 40}, 80),
    "SAC-critic-updates": (SAC, {**SAC40, "num_critic_updates": 3}, 160),
    "SAC-store-policy-action": (
        SAC,
        {**SAC_EXPERT, "extensions": (MIX, StorePolicyAction())},
        160,
    ),
    "SAC-gru-stored": (
        SAC,
        {**SAC40, **GRU, "policy_update_start": 80, "stored_state": True},
        160,
    ),
    "SAC-eval-expert": (SAC, {**SAC40, "eval_expert_policy": push}, 160),
    "SAC-clip-decay": (SAC, {**SAC40, "max_grad_norm": 0.5, "weight_decay": 0.01}, 160),
    "SAC-sweep": (SAC, SAC40, 160),
    "SAC-reacher": (SAC, {**SAC40, **REACHER}, 160),
    "ASAC-keyed": (ASAC, {**S, **R, **KEYED}, 160),
    "ASAC-learned-alpha": (ASAC, {**S, **R, "fixed_alpha": False}, 160),
    "ASAC-gru": (ASAC, {**S, **R, **GRU}, 160),
    "ASAC-residual": (ASAC, {**S, **R, **RESIDUAL}, 160),
    "ASAC-bc": (ASAC, {**S, **R, **BC}, 160),
    "ASAC-gain": (ASAC, {**S, **R, **GAIN}, 160),
    "ASAC-observe": (ASAC, {**S, **R, **OBSERVE}, 160),
    "ASAC-norm": (ASAC, {**S, **R, **NORM, **REWARD_NORM}, 160),
    "ASAC-offline": (ASAC, OFFLINE_KW, 80),
    "ASAC-warmup": (ASAC, {**S, **R, **WARMUP}, 160),
    "ASAC-reacher": (ASAC, {**S, **R, **REACHER, "fixed_alpha": False}, 160),
    "REDQ-keyed": (REDQ, {**S, **R, **KEYED}, 120),
    "REDQ-gru": (REDQ, {**S, **R, **GRU}, 120),
    "REDQ-residual": (REDQ, {**S, **R, **RESIDUAL}, 120),
    "REDQ-bc": (REDQ, {**S, **R, **BC}, 120),
    "REDQ-gain": (REDQ, {**S, **R, **GAIN}, 120),
    "REDQ-observe": (REDQ, {**S, **R, **OBSERVE}, 120),
    "REDQ-norm": (REDQ, {**S, **R, **NORM, **REWARD_NORM}, 120),
    "REDQ-offline": (REDQ, {**OFFLINE_KW, "num_critic_updates": 2}, 40),
    "REDQ-prefill": (REDQ, {**S, **R, **EXPERT_PREFILL}, 120),
    "TQC-keyed": (TQC, {**TQC_KW, **KEYED}, 160),
    "TQC-gru": (TQC, {**TQC_KW, **GRU}, 160),
    "TQC-residual": (TQC, {**TQC_KW, **RESIDUAL}, 160),
    "TQC-bc": (TQC, {**TQC_KW, **BC}, 160),
    "TQC-norm": (TQC, {**TQC_KW, **NORM, **REWARD_NORM}, 160),
    "TQC-offline": (TQC, {**OFFLINE_KW, "num_critics": 3, "n_quantiles": 5}, 80),
    "TQC-prefill": (TQC, {**TQC_KW, **EXPERT_PREFILL}, 160),
    "TQC-gain": (TQC, {**TQC_KW, **GAIN}, 160),
    "TQC-observe": (TQC, {**TQC_KW, **OBSERVE}, 160),
    "AVG-residual": (AVG, {**AVG_KW, **RESIDUAL}, 200),
    "AVG-bc": (AVG, {**AVG_KW, **BC}, 200),
    "AVG-gain": (AVG, {**AVG_KW, **GAIN}, 200),
    "AVG-observe": (AVG, {**AVG_KW, **OBSERVE}, 200),
    "AVG-expose": (AVG, {**AVG_KW, "expose_recent_rollout": True}, 200),
    "AVG-clip": (AVG, {**AVG_KW, "max_grad_norm": 0.5, "weight_decay": 0.01}, 200),
}
# The agents with collectors of their own, before the every-agent gaps give
# them features and experts: APG's differentiable rollout (through a residual
# expert), UDRL's command-conditioned collector on both action kinds (its
# actor-loss batch read by ActorKeyed) and the world models' row collector.
OWN = {"env_id": "Pendulum-v1", "n_envs": 2, "actor_architecture": ("16", "relu")}
APG_KW = {**OWN, "horizon": 8}
UDRL_KW = {**OWN, "n_steps": 16, "batch_size": 16, "buffer_capacity": 8}
UDRL_KW.update({"n_updates_per_iter": 4})
ACTOR_RESIDUAL = {"expert_policy": push, "extensions": (ResidualPolicy(),)}
KEYED_RESIDUAL = {"expert_policy": push, "extensions": (ResidualPolicy(), ActorKeyed())}
DREAMER = {**DREAMER_TINY, "env_id": "CartPole-v1", "n_envs": 2}
DREAMER.update({"train_ratio": 32, "replay_capacity": 80})
SHORT_PENDULUM = gymnax.make("Pendulum-v1")[1].replace(max_steps_in_episode=10)
TDMPC2_KW = {**TDMPC2_TINY, "env_id": "Pendulum-v1", "env_params": SHORT_PENDULUM}
TDMPC2_KW.update({"seed_steps": 20})
OWN_COLLECTOR_CASES: dict[str, tuple[Any, dict[str, Any], int]] = {
    "APG": (APG, APG_KW, 80),
    "APG-residual": (APG, {**APG_KW, **KEYED_RESIDUAL}, 80),
    "UDRL": (UDRL, UDRL_KW, 128),
    "UDRL-discrete": (UDRL, {**UDRL_KW, "env_id": "CartPole-v1"}, 128),
    "UDRL-residual": (UDRL, {**UDRL_KW, **ACTOR_RESIDUAL}, 128),
    "UDRL-keyed": (UDRL, {**UDRL_KW, "extensions": (ActorKeyed(),)}, 128),
    "DreamerV3": (DreamerV3, DREAMER, 80),
    "TDMPC2": (TDMPC2, TDMPC2_KW, 40),
}
# The paths B5's declaration of DQN, DAgger, UDRL and APG moves: DQN's
# targets, networks, losses, target refresh, normalisers, offline training
# and reference CNN; DAgger's expert mix, normalisers and a stateful expert
# (its collector moves onto the replay family's); UDRL's normalisers and gain
# mode; APG's memory, PID head, system class, gain mode and both resumes (its
# optimiser rebuilt, then kept with the schedule spanning the run).
DQN_KW = {**CART, "target_update_interval": 10}
MINATAR = {"env_id": "Breakout-MinAtar", "n_envs": 2, "cnn_image_shape": (10, 10, 4)}
MINATAR.update({"buffer_size": 200, "batch_size": 8, "learning_starts": 16})
SYSTEMS = UniformPerturbation(
    gymnax.make("Pendulum-v1")[1], fields=("m", "l"), scale=0.1
)
SCHEDULE = {"lr_schedule": "warmup_cosine", "warmup_steps": 2}
B5_CASES: dict[str, tuple[Any, dict[str, Any], int]] = {
    "DQN-double": (DQN, {**DQN_KW, "td_target_fn": compute_double_dqn_td_target}, 240),
    "DQN-dueling-mse": (
        DQN,
        {**DQN_KW, "q_network_cls": DuelingQNetwork, "td_loss_fn": compute_mse_td_loss},
        240,
    ),
    "DQN-polyak": (
        DQN,
        {**CART, "tau": 0.5, "target_update_interval": 1, "n_gradient_steps": 2},
        240,
    ),
    "DQN-norm-clip": (
        DQN,
        {**DQN_KW, **NORM, **REWARD_NORM, "max_grad_norm": 0.5, "weight_decay": 0.01},
        240,
    ),
    "DQN-offline": (DQN, {**CART, "learning_starts": 0, "batch_size": 8}, 80),
    "DQN-image": (DQN, MINATAR, 64),
    "DAgger": (DAgger, DAGGER, 200),
    "DAgger-mix": (DAgger, {**DAGGER, "beta_decay": 0.5, "iteration_steps": 40}, 200),
    "DAgger-norm-clip": (
        DAgger,
        {**DAGGER, **NORM, "max_grad_norm": 0.5, "weight_decay": 0.01},
        200,
    ),
    "DAgger-stateful": (DAgger, {**DAGGER, "expert_policy": GainExpert()}, 200),
    "UDRL-norm": (UDRL, {**UDRL_KW, **NORM, **REWARD_NORM}, 128),
    "UDRL-gain": (UDRL, {**UDRL_KW, **GAIN}, 128),
    "APG-gru": (APG, {**APG_KW, "memory": MemoryConfig(kind="gru", hidden_size=8)}, 80),
    "APG-pid": (APG, {**APG_KW, "pid": PIDHeadConfig()}, 80),
    "APG-system": (
        APG,
        {**APG_KW, "system_class": SYSTEMS, "env_params": SYSTEMS.nominal},
        80,
    ),
    "APG-gain": (APG, {**APG_KW, **GAIN}, 80),
    "APG-resume": (APG, {**APG_KW, **SCHEDULE}, 80),
    "APG-resume-kept": (APG, {**APG_KW, **SCHEDULE}, 80),
}


def long_episodes(rows: int = 32, length: int = 8) -> TransitionDataset:
    """rows transitions on Pendulum's observations in episodes of length
    steps, so each row but an episode's last has a next action."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(rows, 3))
    action = rng.uniform(-1, 1, (rows, 1))
    ends = np.arange(rows) % length == length - 1
    rewards = rng.normal(size=rows)
    return TransitionDataset(
        obs, action, rewards, ends, np.zeros(rows, bool), np.roll(obs, -1, axis=0)
    )


# The paths the expert plumbing's move touches, pinned before it: the acting
# gates (JSRL; EDGE on phi* and on the live critic), the target extensions
# reading the critics' predictions and an expert's next action
# (MCVarianceCorrection, IBRL on each agent it runs on, LCBGatedBootstrap on
# the agent's stored expert actions) and TargetBC on the target's next
# actions, online (every a' unknown: no penalty) and offline.
JSRL = JSRLCurriculum(episode_length=8, decay_frac=0.5)
IBRL_PUSH = {"extensions": (IBRL(expert_policy=push),)}
PLUMBING_CASES: dict[str, tuple[Any, dict[str, Any], int]] = {
    "SAC-jsrl": (SAC, {**SAC_EXPERT, "extensions": (JSRL,)}, 160),
    "SAC-edge": (
        SAC,
        {**SAC_EXPERT, "extensions": (SMALL_MC, EDGEExploration(gate="lcb"))},
        160,
    ),
    "SAC-edge-live": (
        SAC,
        {**SAC_EXPERT, "extensions": (EDGEExploration(gate="boltzmann"),)},
        160,
    ),
    "SAC-mc-variance": (
        SAC,
        {**SAC_EXPERT, "extensions": (SMALL_MC, MCVarianceCorrection(threshold=0.01))},
        160,
    ),
    "SAC-ibrl": (SAC, {**SAC40, **IBRL_PUSH}, 160),
    # IBRL's expert the agent's: collection and evaluation read its tracked
    # action, not IBRL's call.
    "SAC-ibrl-agent-expert": (SAC, {**SAC_EXPERT, **IBRL_PUSH}, 160),
    "SAC-lcb": (
        SAC,
        {**SAC_EXPERT, "extensions": (LCBGatedBootstrap(expert_policy=push),)},
        160,
    ),
    "TD3-ibrl": (TD3, {**S, **R, **IBRL_PUSH}, 200),
    "REDQ-ibrl": (REDQ, {**S, **R, **IBRL_PUSH}, 120),
    "TQC-ibrl": (TQC, {**TQC_KW, **IBRL_PUSH}, 160),
    "IQL-ibrl": (IQL, {**S, **R, **IBRL_PUSH}, 200),
    "TD3-target-bc": (TD3, {**S, **R, "extensions": (TargetBC(),)}, 200),
    "TD3-target-bc-offline": (TD3, {**OFFLINE_KW, "extensions": (TargetBC(),)}, 80),
    # IBRL's soft choice, drawn in acting and in the target; last in the table,
    # so the cases before it keep their order.
    "TD3-ibrl-soft": (
        TD3,
        {**S, **R, "extensions": (IBRL(push, soft_choice=True),)},
        200,
    ),
}
# IQL's residual policy under the action stage: the replay keeps the
# residual executing the expert's action where JSRL hands it the step. Last
# in the table, so the cases before keep their order.
RESIDUAL_STAGE_CASES: dict[str, tuple[Any, dict[str, Any], int]] = {
    "IQL-residual-jsrl": (
        IQL,
        {**S, **R, "expert_policy": push, "extensions": (ResidualPolicy(), JSRL)},
        200,
    ),
}
OFFLINE.update({"TD3-target-bc-offline"})
#: The dataset an OFFLINE case trains on, where it is not test_offline's.
DATASETS = {"TD3-target-bc-offline": long_episodes()}
#: Trained for their steps, then for as many more from that state (a second
#: train call): (both calls' options, the second call's steps).
RESUMED: dict[str, tuple[dict[str, Any], int]] = {
    "APG-resume": ({}, 80),
    "APG-resume-kept": (
        {"reset_optimizer_on_resume": False, "total_timesteps": 160},
        80,
    ),
}
CASES: dict[str, tuple[Any, dict[str, Any], int]] = {
    "TD3": (TD3, {**S, **R}, 200),
    "TD3-keyed": (TD3, {**S, **R, "extensions": (Keyed(),)}, 200),
    "TD3-gru": (TD3, {**S, **R, **GRU}, 160),
    "TD3-gru-stored": (TD3, {**S, **R, **GRU, "stored_state": True}, 160),
    "TD3-expert-bc": (TD3, {**S, **R, **BC}, 200),
    "TD3-residual": (TD3, {**S, **R, **RESIDUAL}, 200),
    "TD3-gain": (TD3, {**S, **R, **GAIN}, 200),
    "TD3-observe": (TD3, {**S, **R, **OBSERVE}, 200),
    "TD3-norm": (TD3, {**S, **R, **NORM}, 200),
    "TD3-norm-rewards": (TD3, {**S, **R, **REWARD_NORM}, 200),
    "TD3-pipeline": (
        TD3,
        {**S, **R, "action_pipeline": make_default_action_pipeline(False, 0.3)},
        200,
    ),
    "TD3-offline": (TD3, OFFLINE_KW, 80),
    "TD3-prefill": (TD3, {**S, **R, **EXPERT_PREFILL}, 200),
    "TD3-warmup": (TD3, {**S, **R, **WARMUP}, 200),
    "AVG": (AVG, {**S, "learning_starts": 20}, 200),
    "AVG-keyed": (AVG, {**S, "learning_starts": 20, "extensions": (Keyed(),)}, 200),
    "DAgger-keyed": (DAgger, {**DAGGER, "extensions": (ActorKeyed(),)}, 200),
    "DAgger-discrete": (
        DAgger,
        {
            **DAGGER,
            "env_id": "CartPole-v1",
            "actor_architecture": ("16", "relu"),
            "expert_policy": choose_one,
        },
        200,
    ),
    "IQL": (IQL, {**S, **R}, 200),
    "IQL-keyed": (IQL, {**S, **R, "extensions": (Keyed(),)}, 200),
    "IQL-residual": (IQL, {**S, **R, **RESIDUAL}, 200),
    "IQL-gain": (IQL, {**S, **R, **GAIN}, 200),
    "IQL-observe-norm": (IQL, {**S, **R, **OBSERVE, **NORM}, 200),
    "IQL-offline": (IQL, OFFLINE_KW, 80),
    "IQL-prefill": (IQL, {**S, **R, **EXPERT_PREFILL}, 200),
    "DQN": (DQN, {**CART, "target_update_interval": 10}, 240),
    "DQN-keyed": (
        DQN,
        {**CART, "target_update_interval": 10, "extensions": (ValueKeyed(),)},
        240,
    ),
    "PQN-keyed": (PQN, {**PQN_KW, "extensions": (ValueKeyed(),)}, 512),
    "PQN-norm": (PQN, {**PQN_KW, **NORM}, 512),
    "SAC-keyed": (
        SAC,
        {
            **S,
            **R,
            "policy_update_start": 80,
            "alpha_update_start": 120,
            "target_entropy_per_dim": linear(-3.0, -1.0, 120),
            "extensions": (Keyed(),),
        },
        240,
    ),
    "SAC-gru": (
        SAC,
        {**SAC40, **GRU, "policy_update_start": 80, "alpha_update_start": 80},
        160,
    ),
    "SAC-residual": (
        SAC,
        {**SAC_EXPERT, "extensions": (ResidualPolicy(), Keyed(), DEFAULT_PREFILL)},
        160,
    ),
    "SAC-expert-bc": (
        SAC,
        {
            **SAC_EXPERT,
            "extensions": (ImitationLoss(push, coef=1.0), MIX, DEFAULT_PREFILL),
        },
        160,
    ),
    "SAC-expert": (SAC, {**SAC_EXPERT, "extensions": (MIX, PREFILL40)}, 160),
    "SAC-augment": (
        SAC,
        {**SAC_EXPERT, "extensions": (ExpertActionInput(), MIX, DEFAULT_PREFILL)},
        160,
    ),
    "SAC-gain": (
        SAC,
        {
            **SAC40,
            "expert_policy": GainExpert(),
            "extensions": (ExpertGains(), DEFAULT_PREFILL),
        },
        160,
    ),
    "SAC-augment-state": (
        SAC,
        {
            **SAC40,
            "expert_policy": GainExpert(),
            "extensions": (
                ExpertStateInput.for_expert(GainExpert()),
                MIX,
                DEFAULT_PREFILL,
            ),
        },
        160,
    ),
    "SAC-train-frac": (
        SAC,
        {**SAC40, "extensions": (TrainingProgressInput(),)},
        160,
    ),
    "SAC-norm": (SAC, {**SAC40, **NORM}, 160),
    "SAC-fixed": (SAC, {**SAC40, "fixed_alpha": True, "policy_update_start": 60}, 160),
    "SAC-pid-norm": (
        SAC,
        {**PID, **R, **NORM, "policy_update_start": 40, "alpha_update_start": 40},
        160,
    ),
    "SAC-mix": (SAC, {**SAC_EXPERT, "extensions": (MIX,)}, 160),
    "SAC-prefill": (SAC, {**SAC_EXPERT, "extensions": (PREFILL40,)}, 160),
    "SAC-expert-norm": (
        SAC,
        {**SAC_EXPERT, **NORM, **REWARD_NORM, "extensions": (MIX, PREFILL40)},
        160,
    ),
    "SAC-expert-gates": (
        SAC,
        {**SAC_EXPERT, "extensions": (*GATES, MIX, PREFILL40)},
        160,
    ),
    "SAC-expert-bellman": (
        SAC,
        {**SAC_EXPERT, "extensions": (*BELLMAN, MIX, PREFILL40)},
        160,
    ),
    "SAC-residual-bc": (SAC, {**SAC_EXPERT, "extensions": RESIDUAL_BC}, 160),
    "REDQ": (REDQ, {**S, **R}, 120),
    "REDQ-warmup": (REDQ, {**S, **R, **WARMUP}, 120),
    "ASAC": (ASAC, {**S, **R}, 160),
    "ASAC-prefill": (ASAC, {**S, **R, **EXPERT_PREFILL}, 160),
    "CrossQ": (CrossQ, CROSSQ, 160),
    "CrossQ-keyed": (CrossQ, {**CROSSQ, "extensions": (Keyed(),)}, 160),
    "CrossQ-residual": (CrossQ, {**CROSSQ, **RESIDUAL}, 160),
    "CrossQ-expert-bc": (CrossQ, {**CROSSQ, **BC}, 160),
    "CrossQ-gain": (CrossQ, {**CROSSQ, **GAIN}, 160),
    "CrossQ-observe": (CrossQ, {**CROSSQ, **OBSERVE}, 160),
    "CrossQ-norm": (CrossQ, {**CROSSQ, **NORM, **REWARD_NORM}, 160),
    "CrossQ-warmup": (CrossQ, {**CROSSQ, **WARMUP}, 160),
    "PPO-keyed": (PPO, {**ON, "extensions": (Keyed(),)}, 128),
    "PPO-norm": (PPO, {**ON, **NORM}, 128),
    "PPO-norm-brax": (PPO, {**ON, **NORM, "env_id": "fast"}, 128),
    "PPO-pid-norm": (
        PPO,
        {**PID, **NORM, "n_steps": 16, "batch_size": 8, "n_epochs": 1},
        128,
    ),
    "APO-norm": (APO, {**ON, **NORM}, 128),
    # The time geometry with brax's forced resets, and recurrent PPO.
    "PPO-time-reset": (PPO, TIME_RESET, 256),
    "PPO-gru": (
        PPO,
        {**ON, "n_steps": 16, "num_minibatches": 2, "bptt_length": 4, **PPO_GRU},
        128,
    ),
    "TQC": (TQC, {**S, **R, "num_critics": 3, "n_quantiles": 5}, 160),
    # The on-policy paths the on-policy family's restructure moves: each
    # extension phase PPO folds (the reward, on_target in each minibatch),
    # an expert read on its raw observations, the action transforms, the
    # schedules, the exposed rollout and whole-rollout recurrent sequences.
    "PPO-discrete-keyed": (PPO, {**ON, "env_id": "CartPole-v1", **KEYED}, 128),
    "PPO-bc": (
        PPO,
        {**ON, **NORM, "extensions": (ImitationLoss(push, coef=1.0),)},
        128,
    ),
    "PPO-gail": (PPO, {**ON, "extensions": (AdversarialReward(GAIL_DATA),)}, 128),
    "PPO-reward": (PPO, {**ON, "extensions": (RewardKeyed(),)}, 128),
    "PPO-residual": (PPO, {**ON, **RESIDUAL}, 128),
    "PPO-observe": (PPO, {**ON, **OBSERVE}, 128),
    "PPO-gain": (PPO, {**ON, **GAIN}, 128),
    "PPO-time-keyed": (PPO, {**TIME_RESET, **KEYED}, 256),
    "PPO-reward-norm": (PPO, {**ON, **REWARD_NORM}, 128),
    "PPO-squash-schedule": (
        PPO,
        {**ON, "squash": True, "clip_range": linear(0.3, 0.1, 128)},
        128,
    ),
    "PPO-expose": (PPO, {**ON, "expose_recent_rollout": True}, 128),
    "PPO-gru-whole": (PPO, {**ON, "n_steps": 16, "num_minibatches": 2, **PPO_GRU}, 128),
    "APO-keyed": (APO, {**ON, **KEYED}, 128),
    "APO-bc": (APO, {**ON, **NORM, **BC}, 128),
    "APO-discrete": (APO, {**ON, "env_id": "CartPole-v1"}, 128),
    "PQN-huber": (PQN, {**PQN_KW, "td_loss_fn": make_huber_td_loss(1.0)}, 512),
    "PQN-norm-expose": (
        PQN,
        {**PQN_KW, **REWARD_NORM, "expose_recent_rollout": True},
        512,
    ),
    **SAC_FAMILY_CASES,
    **OWN_COLLECTOR_CASES,
    **B5_CASES,
    **PLUMBING_CASES,
    **RESIDUAL_STAGE_CASES,
}


def run(name: str, agent: Any, logging: Any = None) -> tuple[Any, Any]:
    """The case's training: ``train`` for its steps (with a RESUMED case's
    options) or, OFFLINE, ``train_offline`` for as many updates on
    test_offline's dataset (CartPole's observations for DQN)."""
    steps = CASES[name][2]
    if name in OFFLINE:
        discrete = name == "DQN-offline"  # CartPole's observations, else Pendulum's
        data = episodes(4, discrete=True) if discrete else episodes(3, discrete=False)
        if name in DATASETS:
            data = DATASETS[name]
        return agent.train_offline(data, steps, seed=SEEDS, logging_config=logging)
    options = RESUMED.get(name, ({}, 0))[0]
    return agent.train(SEEDS, n_timesteps=steps, logging_config=logging, **options)


def record(state: Any, evals: Any, tag: str = "") -> dict[str, np.ndarray]:
    """Every final-state leaf (keys as their data) and logged evaluation."""
    out = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        if isinstance(leaf, jax.Array) and jnp.issubdtype(
            leaf.dtype, jax.dtypes.prng_key
        ):
            leaf = jax.random.key_data(leaf)
        out[f"{tag}state" + jax.tree_util.keystr(path)] = np.asarray(leaf)
    # APG returns every update's auxiliaries, which TargetFoundation reads
    # (matching_loss, m_rmse), its evaluations inside them.
    if hasattr(evals, "evaluations"):
        updates = evals.replace(evaluations=None)
        for path, leaf in jax.tree_util.tree_flatten_with_path(updates)[0]:
            out[f"{tag}aux" + jax.tree_util.keystr(path)] = np.asarray(leaf)
        evals = evals.evaluations
    out.update({f"{tag}eval/" + k: np.asarray(v) for k, v in (evals or {}).items()})
    return out


def leaves(name: str) -> dict[str, np.ndarray]:
    """The case's :func:`record`, then an OFFLINE case's online one or a
    RESUMED case's second call's."""
    cls, kwargs, _ = CASES[name]
    logging = LoggingConfig(
        config={}, log_frequency=40, use_wandb=False, sweep=name in SWEEP
    )
    agent = cls(**kwargs)
    state, evals = run(name, agent, logging)
    out = record(state, evals)
    if name in OFFLINE:
        state, evals = agent.train(
            SEEDS, n_timesteps=ONLINE, initial_state=state, logging_config=logging
        )
        out.update(record(state, evals, tag="online/"))
    if name in RESUMED:
        options, steps = RESUMED[name]
        # A copy: the resume donates the state it is given.
        state, evals = agent.train(
            SEEDS,
            n_timesteps=steps,
            initial_state=jax.tree.map(jnp.copy, state),
            logging_config=logging,
            **options,
        )
        out.update(record(state, evals, tag="resumed/"))
    return out


class _Built(Exception):
    pass


def program(name: str) -> str:
    """The text of the case's training program: the jaxpr of the function
    ``train`` vmaps over the seeds (logging off: its run ids vary)."""
    cls, kwargs, _ = CASES[name]
    agent, built = cls(**kwargs), []
    make_train = agent.get_make_train

    def spy() -> Any:
        def build(**kw: Any) -> None:
            built.append(make_train()(**kw))
            raise _Built

        return build

    agent.get_make_train = spy  # type: ignore[method-assign]
    try:
        run(name, agent)
    except _Built:
        pass
    keys = jax.vmap(jax.random.PRNGKey)(jnp.array(SEEDS))
    return str(jax.make_jaxpr(jax.vmap(built[0]))(keys, jnp.arange(len(SEEDS))))


def differences(ref: dict[str, np.ndarray], got: dict[str, np.ndarray]) -> list[str]:
    """Each leaf that is not bitwise the same, with its largest relative gap."""
    found = [f"{k}: only in one record" for k in sorted(ref.keys() ^ got.keys())]
    for k in sorted(set(ref).intersection(got)):
        a, b = ref[k], got[k]
        if a.shape != b.shape:
            found.append(f"{k}: shape {a.shape} -> {b.shape}")
        elif not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
            gap = np.abs(a.astype(np.float64) - b) / np.maximum(np.abs(a), 1e-6)
            found.append(f"{k}: largest relative gap {float(np.nanmax(gap)):.2e}")
    return found


def main(argv: list[str]) -> None:
    mode, folder, names = argv[0], Path(argv[1]), argv[2:] or list(CASES)
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        got, text = leaves(name), program(name)
        if mode == "record":
            np.savez(folder / f"{name}.npz", **got)  # type: ignore[arg-type]
            (folder / f"{name}.jaxpr").write_text(text)
            print(f"{name}: recorded {len(got)} arrays")
            continue
        found = differences(dict(np.load(folder / f"{name}.npz")), got)
        same = (folder / f"{name}.jaxpr").read_text() == text
        print(
            f"{name}: {'BITWISE' if not found else f'{len(found)} differ'}; jaxpr "
            + ("same" if same else "differs")
        )
        for line in found:
            print("  " + line)


if __name__ == "__main__":
    main(sys.argv[1:])
