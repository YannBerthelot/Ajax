"""The expert's actions an agent stores (``a_expert`` at s, ``next_a_expert``
at s') come from the expert's state tracked along each trajectory: a PID,
re-run step by step on each env's stored observations from its initial
state, reset at each episode start, takes exactly the stored actions."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.agents.TD3.TD3 import TD3
from tests.extensions.test_expert_observations import CounterEnv

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


def replay_rows(agent_cls: type, **kw: Any) -> dict:
    """A short run's replay, ``(n_envs, rows, ...)`` in time order."""
    env = CounterEnv()
    expert = PID()
    agent = agent_cls(
        env,
        env_params=env.default_params,
        n_envs=2,
        actor_architecture=NET,
        critic_architecture=NET,
        learning_starts=16,
        batch_size=8,
        expert_policy=expert,
        **kw,
    )
    state, _ = agent.train(seed=0, n_timesteps=64, num_episode_test=1)
    rows = state.collector_state.buffer_state.experience
    return {k: np.asarray(v)[0] for k, v in rows.items()}  # the only seed


REPLAY_AGENTS = {
    "SAC": lambda: replay_rows(SAC, expert_buffer_n_steps=0, expert_fraction=0.0),
    "TD3": lambda: replay_rows(TD3),
}


@pytest.mark.parametrize("agent", sorted(REPLAY_AGENTS))
def test_stored_expert_actions_are_a_sequential_pid_rerun(agent: str) -> None:
    rows = REPLAY_AGENTS[agent]()
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
