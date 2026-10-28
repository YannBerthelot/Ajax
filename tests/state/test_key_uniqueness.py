"""No random key hands out related numbers twice: over a fresh run's
initialisation and pretraining, then a few training iterations, each key is
drawn on once, and never also split or folded at an index its draw spans
(the conformance kit's key-use tape, tests/conformance/kit.py). The runs
cover the gates that leave a draw out or put one in: the warm-up, an actor
waiting for policy_update_start (before and after it), a policy delay,
several gradient steps per update, forced resets, the extensions'
post_update, pretraining, the action stage, an evaluation chooser and the
world models' row collector.

Each case lists the reuses it still has, by the pair of uses that meet; a
change that removes one removes it here (the comment names the change
planned to remove it)."""

from __future__ import annotations

from typing import Any

import gymnax
import jax
import pytest

import ajax
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.extensions.expert import ExpertWarmup, JSRLCurriculum
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.pretrain import (
    BellmanPretrain,
    ExpertPrefill,
    MCPretrain,
    PriorData,
)
from ajax.extensions.target_mods import IBRL
from ajax.logging.wandb_logging import LoggingConfig
from ajax.wrappers import ObservationNoiseWrapper
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY
from tests.agents.extension_stubs import DrawEveryPhase
from tests.agents.TDMPC2.common import TINY_AGENT as TDMPC2_TINY
from tests.agents.test_offline import episodes
from tests.conformance import kit


def push(obs: jax.Array) -> jax.Array:
    """An expert pushing at half the torque."""
    return 0.5 * jax.numpy.ones(obs.shape[:-1] + (1,))


NET: dict[str, Any] = {
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
}
REPLAY: dict[str, Any] = {"env_id": "Pendulum-v1", "n_envs": 2, **NET}
REPLAY.update({"buffer_size": 64, "batch_size": 4, "learning_starts": 4})
# Updates from tick 2 (4 steps on 2 envs), the actor kept from tick 4.
WAIT: dict[str, Any] = {**REPLAY, "policy_update_start": 8}
# SAC waits for 2000 steps by default; these cases leave that gate out.
SAC: dict[str, Any] = {**REPLAY, "policy_update_start": 0}
INCREMENTAL: dict[str, Any] = {
    "env_id": "Pendulum-v1",
    "n_envs": 2,
    **NET,
    "learning_starts": 4,
}
NOISY_PENDULUM = ObservationNoiseWrapper(gymnax.make("Pendulum-v1")[0], 0.1)
SMALL_MC = MCPretrain(
    push, n_mc_steps=200, n_mc_episodes=4, n_steps=4, online_light_steps=1
)
EVALUATE = {"logging_config": LoggingConfig(log_frequency=4, backends=())}

SAC_EXPERT: dict[str, Any] = {**SAC, "expert_policy": push}
CROSSQ: dict[str, Any] = {"env_id": "Pendulum-v1", "n_envs": 2, "actor_width": 8}
CROSSQ.update({"critic_width": 8, "buffer_size": 64, "batch_size": 4})
DQN: dict[str, Any] = {"env_id": "CartPole-v1", "n_envs": 2, "buffer_size": 64}
DQN.update({"architecture": ("8", "relu"), "batch_size": 4, "n_gradient_steps": 2})
TWO_DRAWING = (DrawEveryPhase(), DrawEveryPhase(name="second"))
PPO: dict[str, Any] = {"env_id": "Pendulum-v1", "n_envs": 2, "n_steps": 8}
PPO.update({"batch_size": 8, "num_resets_per_eval": 1, "num_evals": 3})
PPO.update({"actor_architecture": ("8", "tanh"), "critic_architecture": ("8", "tanh")})
PRETRAINING = (
    PriorData(episodes(3, discrete=False), offline_updates=2),
    BellmanPretrain(n_steps=2),
)
GATES = (JSRLCurriculum(episode_length=8, decay_frac=0.5), ExpertWarmup(fraction=0.5))
DREAMER: dict[str, Any] = {**DREAMER_TINY, "env_id": "CartPole-v1", "n_envs": 2}
DREAMER.update({"train_ratio": 32, "replay_capacity": 80})
TDMPC2_KW: dict[str, Any] = {**TDMPC2_TINY, "env_id": "Pendulum-v1", "n_envs": 2}
TDMPC2_KW.update({"seed_steps": 4})

#: name: (the agent's class, its arguments, the reuses it still has).
CASES: dict[str, tuple[Any, dict[str, Any], set]] = {
    "SAC-warmup-wait": (ajax.SAC, WAIT, set()),
    "REDQ-wait": (ajax.REDQ, {**WAIT, "num_critic_updates": 2}, set()),
    "TQC-wait": (ajax.TQC, WAIT, set()),
    "ASAC-wait": (ajax.ASAC, WAIT, set()),
    "AVG-wait": (ajax.AVG, {**INCREMENTAL, "policy_update_start": 8}, set()),
    "TD3-delay": (ajax.TD3, {**REPLAY, "policy_delay": 2}, set()),
    "CrossQ-delay": (ajax.CrossQ, {**CROSSQ, "learning_starts": 4}, set()),
    "DQN-gradient-steps": (ajax.DQN, {**DQN, "learning_starts": 4}, set()),
    "PPO-forced-reset": (ajax.PPO, {**PPO, "extensions": TWO_DRAWING}, set()),
    "SAC-every-phase": (ajax.SAC, {**SAC, "extensions": TWO_DRAWING}, set()),
    "SAC-prior-data-bellman": (ajax.SAC, {**SAC, "extensions": PRETRAINING}, set()),
    "SAC-mc-pretrain": (
        ajax.SAC,
        {**SAC_EXPERT, "extensions": (SMALL_MC,)},
        set(),
    ),
    "SAC-expert-prefill": (
        ajax.SAC,
        {**SAC_EXPERT, "extensions": (ExpertPrefill(n_steps=8),)},
        set(),
    ),
    "SAC-edge": (
        ajax.SAC,
        {**SAC_EXPERT, "extensions": (EDGEExploration(gate="boltzmann"),)},
        set(),
    ),
    "TD3-jsrl-warmup": (
        ajax.TD3,
        {**REPLAY, "expert_policy": push, "extensions": GATES},
        set(),
    ),
    "SAC-ibrl-evaluation": (
        ajax.SAC,
        {**SAC, "extensions": (IBRL(push, soft_choice=True),)},
        set(),
    ),
    "SAC-observation-noise": (
        ajax.SAC,
        {**SAC, "env_id": NOISY_PENDULUM},
        set(),
    ),
    "DreamerV3": (DreamerV3, DREAMER, set()),
    "TDMPC2": (TDMPC2, TDMPC2_KW, set()),
}
#: The cases whose iterations evaluate (their evaluation chooser's draws).
EVALUATED = {"SAC-ibrl-evaluation"}


def shared_keys(name: str) -> set[tuple[str, ...]]:
    """The pairs of uses of one key in the case's fresh run, then 7
    iterations of a train call for 2 env steps."""
    cls, kwargs, _ = CASES[name]
    options = EVALUATE if name in EVALUATED else {}
    step = kit.capture_step(cls(**kwargs), 2, **options)
    with kit.recording_key_uses() as uses:
        # Seed 7: TD-MPC2 builds its metrics' structure from draws on
        # PRNGKey(0), whose values it discards.
        state = step.fresh(jax.random.PRNGKey(7))
        iterate = jax.jit(lambda state, tick: step.iteration(state, 0, tick)[0])
        for tick in range(7):
            state = iterate(state, tick)
        jax.block_until_ready(state)
        jax.effects_barrier()
    return uses.shared()


@pytest.mark.parametrize("name", CASES)
def test_no_key_hands_out_related_numbers_twice(name: str) -> None:
    assert shared_keys(name) == CASES[name][2]
