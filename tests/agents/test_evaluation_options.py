"""The evaluation options every agent takes from its family: the expert
whose return each evaluation logs (ExpertGuided's eval_expert_policy) and a
sweep's evaluations in the last fifth of training (LoggingConfig.sweep, read
by the shared loop)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ajax import PPO, TD3
from ajax.logging.wandb_logging import LoggingConfig

TINY: dict[str, Any] = {
    "env_id": "Pendulum-v1",
    "n_envs": 2,
    "buffer_size": 64,
    "batch_size": 8,
}
TINY.update({"learning_starts": 8})
TINY.update({"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")})


def push(obs: jnp.ndarray) -> jnp.ndarray:
    return jnp.full(obs.shape[:-1] + (1,), 2.0)


def pull(obs: jnp.ndarray) -> jnp.ndarray:
    return jnp.full(obs.shape[:-1] + (1,), -2.0)


def _logs(agent: Any, sweep: bool = False, n_timesteps: int = 16) -> dict:
    logging = LoggingConfig(log_frequency=8, sweep=sweep, backends=())
    return agent.train(seed=0, n_timesteps=n_timesteps, logging_config=logging)[1]


def test_each_evaluation_logs_the_eval_experts_return_instead() -> None:
    """With eval_expert_policy, the expert's logged return is that expert's,
    on the same episodes the training expert's would be."""
    key = "Eval/episodic mean expert reward"
    instead = _logs(TD3(**TINY, expert_policy=push, eval_expert_policy=pull))[key]
    np.testing.assert_array_equal(instead, _logs(TD3(**TINY, expert_policy=pull))[key])
    assert not np.array_equal(instead, _logs(TD3(**TINY, expert_policy=push))[key])


def test_a_sweep_evaluates_in_the_last_fifth_of_training_on_every_agent() -> None:
    """LoggingConfig.sweep: no evaluation before 80% of the run (the row's
    timestep stays -1), for an off-policy and an on-policy agent alike."""
    on_policy: dict[str, Any] = {
        "env_id": "Pendulum-v1",
        "n_envs": 2,
        "n_steps": 4,
        "batch_size": 8,
    }
    for agent in (TD3(**TINY), PPO(**on_policy)):
        swept = np.asarray(_logs(agent, sweep=True, n_timesteps=40)["timestep"])
        every = np.asarray(_logs(agent, n_timesteps=40)["timestep"])
        assert (every[every >= 0] < 32).any(), every
        assert (swept[swept >= 0] >= 32).all() and (swept >= 32).any(), swept
