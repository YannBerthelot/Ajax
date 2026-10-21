"""The expert's actions an agent stores (``a_expert`` at s, ``next_a_expert``
at s') come from the expert's state tracked along each trajectory: a PID,
re-run step by step on each env's stored observations from its initial
state, reset at each episode start, takes exactly the stored actions. Its
actor-loss extensions read them (ImitationLoss with the agent's expert)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.TD3.TD3 import TD3
from ajax.extensions.base import Extension
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.expert import ImitationLoss
from ajax.extensions.target_mods import IBRL
from tests.extensions.test_expert_observations import (
    CounterEnv,
    IntegratingExpert,
    integrated_action,
)

NET = ("8", "relu")
SETPOINT = 3.0  # the counter's observation is its step in the episode


class PID:
    """A PID on the counter's observation, its state a pytree (integral of
    the error, previous error), as TargetGym's: ``u = kp e + ki I + kd (e -
    e_prev)`` with ``I`` the integral after the call, clipped to [-1, 1]."""

    kp, ki, kd = 0.2, 0.05, 0.1

    def init_state(self, n_envs: int) -> dict:
        zeros = jnp.zeros((n_envs, 1))
        return {"integral": zeros, "prev_error": zeros}

    def __call__(self, state: dict, obs: jax.Array) -> tuple:
        error = SETPOINT - obs[..., :1]
        integral = state["integral"] + error
        derivative = error - state["prev_error"]
        u = self.kp * error + self.ki * integral + self.kd * derivative
        return jnp.clip(u, -1.0, 1.0), {"integral": integral, "prev_error": error}


def rerun(
    expert: PID, obs: np.ndarray, next_obs: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The PID's actions at each of one env's rows, in time order, at s and
    at s' (from its state after the call at s), its state reset after
    every row that ended an episode."""
    state, at_s, at_next = expert.init_state(1), [], []
    for t in range(len(obs)):
        action, state = expert(state, jnp.asarray(obs[t : t + 1]))
        at_s.append(np.asarray(action)[0])
        at_next.append(
            np.asarray(expert(state, jnp.asarray(next_obs[t : t + 1]))[0])[0]
        )
        if ends[t]:
            state = expert.init_state(1)
    return np.stack(at_s), np.stack(at_next)


#: The replay agents' updates from step 16, on batches of 8.
REPLAY = {"learning_starts": 16, "batch_size": 8}
#: Each agent storing its expert's actions, and its options for a short run
#: (SAC: no expert prefill or warm-up mixing; its actor steps from its first
#: update. APO: two rollouts of two episodes per env, the last one kept).
AGENTS: dict[str, tuple[type, dict]] = {
    "SAC": (
        SAC,
        {
            **REPLAY,
            "expert_buffer_n_steps": 0,
            "expert_fraction": 0.0,
            "policy_update_start": 8,
        },
    ),
    "TD3": (TD3, REPLAY),
    "REDQ": (REDQ, REPLAY | {"num_critic_updates": 2}),  # not its 20 a step
    "ASAC": (ASAC, REPLAY),
    "AVG": (AVG, {"learning_starts": 16}),
    "APO": (
        APO,
        {"n_steps": 16, "batch_size": 8, "n_epochs": 1, "expose_recent_rollout": True},
    ),
}
#: The agents keeping a trajectory to re-run: AVG keeps only the step it
#: updates on.
TRAJECTORY_AGENTS = sorted(set(AGENTS) - {"AVG"})
FIELDS = ("raw_obs", "next_obs", "terminated", "truncated", "a_expert", "next_a_expert")


def train(agent: str, expert: Any, extensions: tuple = ()) -> Any:
    """A short run on the counter (64 steps, 2 envs)."""
    env = CounterEnv()
    agent_cls, kw = AGENTS[agent]
    model = agent_cls(
        env,
        env_params=env.default_params,
        n_envs=2,
        actor_architecture=NET,
        critic_architecture=NET,
        expert_policy=expert,
        extensions=extensions,
        **kw,
    )
    state, _ = model.train(seed=0, n_timesteps=64, num_episode_test=1)
    jax.effects_barrier()
    return state


def stored_rows(agent: str, state: Any) -> dict:
    """Each env's stored rows in time order, ``(n_envs, rows, ...)``, of the
    only seed: the replay's 32 rows, or APO's last rollout of 16, which
    starts an episode (the rerun starts from the initial state)."""
    if agent == "APO":
        rollout = state.last_rollout  # (seeds, T, n_envs, ...)
        rows = {
            k: np.swapaxes(np.asarray(getattr(rollout, k))[0], 0, 1) for k in FIELDS
        }
        assert (rows["raw_obs"][:, 0] == 0).all()
        return rows
    experience = state.collector_state.buffer_state.experience
    return {k: np.asarray(experience[k])[0][:, : 64 // 2] for k in FIELDS}


@pytest.mark.parametrize("agent", TRAJECTORY_AGENTS)
def test_stored_expert_actions_are_a_sequential_pid_rerun(agent: str) -> None:
    rows = stored_rows(agent, train(agent, PID()))
    n_written = rows["raw_obs"].shape[1]
    for env in range(2):
        obs, next_obs = (
            rows["raw_obs"][env, :n_written],
            rows["next_obs"][env, :n_written],
        )
        ends = np.logical_or(rows["terminated"], rows["truncated"])[env, :n_written, 0]
        assert ends.sum() >= 2  # several episodes, so the resets are checked
        at_s, at_next = rerun(PID(), obs, next_obs, ends)
        np.testing.assert_allclose(rows["a_expert"][env, :n_written], at_s, atol=1e-6)
        np.testing.assert_allclose(
            rows["next_a_expert"][env, :n_written], at_next, atol=1e-6
        )


class ActorBatchRecorder(Extension):
    """Records the raw observations and the expert's actions of every
    actor-loss batch it is given; adds nothing to the loss."""

    name = "actor_batch_recorder"

    def __init__(self) -> None:
        self.seen: list[tuple[np.ndarray, np.ndarray]] = []

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        keep = lambda o, a: self.seen.append((np.asarray(o), np.asarray(a)))  # noqa: E731
        jax.debug.callback(keep, batch["raw_observations"], batch["a_expert"])
        return 0.0


@pytest.mark.parametrize("agent", sorted(AGENTS))
def test_the_actor_loss_reads_the_stored_expert_actions(agent: str) -> None:
    """Each actor-loss batch holds, at every row, the integrating expert's
    action from its state there, A(s); ImitationLoss holding the same
    expert, callable only with its state, reads them instead of calling
    it (a call without its state would fail when the update is traced)."""
    expert, recorder = IntegratingExpert(), ActorBatchRecorder()
    train(agent, expert, (recorder, ImitationLoss(expert, coef=1.0)))
    raw, a_expert = (np.concatenate(x) for x in zip(*recorder.seen, strict=True))
    assert len(raw) >= 8 and len(np.unique(raw[..., 0])) > 2
    np.testing.assert_allclose(
        a_expert[..., 0], integrated_action(raw[..., 0]), atol=1e-6
    )


class TargetBatchRecorder(Extension):
    """Records the raw next observations and the expert's actions there of
    every target batch it is given; leaves the target as it is."""

    name = "target_batch_recorder"

    def __init__(self) -> None:
        self.seen: list[tuple[np.ndarray, np.ndarray]] = []

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        keep = lambda o, a: self.seen.append((np.asarray(o), np.asarray(a)))  # noqa: E731
        jax.debug.callback(keep, batch["raw_next_observations"], batch["next_a_expert"])
        return target


@pytest.mark.parametrize("agent", ["SAC", "TD3", "REDQ", "AVG"])
def test_the_target_reads_the_stored_expert_actions_at_the_next_state(
    agent: str,
) -> None:
    """Each target batch (a replay batch; AVG: its step) holds the raw s' and, there, the integrating
    expert's action from its state after the call at s, A(s'); IBRL holding
    the same expert, callable only with its state, reads them instead of
    calling it."""
    expert, recorder = IntegratingExpert(), TargetBatchRecorder()
    train(agent, expert, (recorder, IBRL(expert_policy=expert)))
    raw_next, a_next = (np.concatenate(x) for x in zip(*recorder.seen, strict=True))
    assert len(raw_next) >= 8 and len(np.unique(raw_next[..., 0])) > 2
    np.testing.assert_allclose(
        a_next[..., 0], integrated_action(raw_next[..., 0]), atol=1e-6
    )


def test_ibrl_is_refused_on_an_average_reward_target() -> None:
    """IBRL's bonus is discounted; ASAC's target is average-reward."""
    expert = IntegratingExpert()
    with pytest.raises(MissingCapability, match="needs discounted"):
        train("ASAC", expert, (IBRL(expert_policy=expert),))
