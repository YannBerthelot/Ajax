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
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.TD3.TD3 import TD3
from ajax.extensions.base import Extension
from ajax.extensions.expert import ImitationLoss
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


#: Each agent storing its expert's actions, and its options for a short run
#: (SAC: no expert prefill or warm-up mixing; its actor steps from its first
#: update).
AGENTS: dict[str, tuple[type, dict]] = {
    "SAC": (
        SAC,
        {"expert_buffer_n_steps": 0, "expert_fraction": 0.0, "policy_update_start": 8},
    ),
    "TD3": (TD3, {}),
    "REDQ": (REDQ, {}),
    "ASAC": (ASAC, {}),
}


def train(agent: str, expert: Any, extensions: tuple = ()) -> Any:
    """A short run on the counter (64 steps, 2 envs, updates from step 16)."""
    env = CounterEnv()
    agent_cls, kw = AGENTS[agent]
    model = agent_cls(
        env,
        env_params=env.default_params,
        n_envs=2,
        actor_architecture=NET,
        critic_architecture=NET,
        learning_starts=16,
        batch_size=8,
        expert_policy=expert,
        extensions=extensions,
        **kw,
    )
    state, _ = model.train(seed=0, n_timesteps=64, num_episode_test=1)
    jax.effects_barrier()
    return state


@pytest.mark.parametrize("agent", sorted(AGENTS))
def test_stored_expert_actions_are_a_sequential_pid_rerun(agent: str) -> None:
    experience = train(agent, PID()).collector_state.buffer_state.experience
    # The replay's rows, (n_envs, rows, ...) in time order, of the only seed.
    rows = {k: np.asarray(v)[0] for k, v in experience.items()}
    n_written = 64 // 2
    for env in range(2):
        obs, next_obs = (
            rows["raw_obs"][env, :n_written],
            rows["next_obs"][env, :n_written],
        )
        ends = np.logical_or(rows["terminated"], rows["truncated"])[env, :n_written, 0]
        assert ends.sum() >= 3  # several episodes, so the resets are checked
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
