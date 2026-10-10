"""APO's differential GAE stops at episode ends, as APO's official code does.

``xtma/apo`` (``apo/algos/utils.py``, ``generalized_advantage_estimation``,
run with ``discount = 1``) computes, backwards over the rollout,
``delta_t = r_t - eta + V(s_{t+1}) (1 - d_t) - V(s_t)`` and
``A_t = delta_t + lambda (1 - d_t) A_{t+1}``. One update runs on a
synthetic rollout where env 0 terminates and env 1 is truncated (its final
observation differs from the reset one); the advantages and value targets
it computes are checked against that recursion in NumPy, a truncated step
bootstrapping on its final observation.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.APO import train_APO
from ajax.agents.APO.APO import APO
from ajax.agents.APO.state import APOConfig
from ajax.extensions.base import ExtensionStack
from ajax.networks.networks import predict_value
from ajax.state import Transition

T, N_ENVS = 6, 2
TERMINATED, TRUNCATED = (2, 0), (3, 1)  # (step, env)


def _rollout() -> Transition:
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    states = jax.random.normal(keys[0], (T + 1, N_ENVS, 3))
    obs, next_obs = states[:-1], states[1:]
    # The rows after an episode end start from a reset observation.
    for t, e in (TERMINATED, TRUNCATED):
        obs = obs.at[t + 1, e].set(jax.random.normal(keys[1], (3,)))
    flag = jnp.zeros((T, N_ENVS, 1))
    return Transition(
        obs=obs,
        action=jnp.zeros((T, N_ENVS, 1)),
        reward=jax.random.uniform(keys[2], (T, N_ENVS, 1)),
        terminated=flag.at[TERMINATED].set(1.0),
        truncated=flag.at[TRUNCATED].set(1.0),
        next_obs=next_obs,
        raw_obs=obs,
        log_prob=jnp.zeros((T, N_ENVS, 1)),
        raw_action=jnp.zeros((T, N_ENVS, 1)),
    )


def test_differential_gae_stops_at_episode_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = APO(
        "Pendulum-v1",
        n_envs=N_ENVS,
        n_steps=T,
        num_minibatches=2,
        actor_architecture=("8", "tanh"),
        critic_architecture=("8", "tanh"),
    )
    config = cast(APOConfig, agent.agent_config)
    state = train_APO.init_APO(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    seen: dict[str, Any] = {}
    compute_gae = train_APO._compute_gae

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["gae"], seen["targets"] = compute_gae(*args, **kwargs)
        return seen["gae"], seen["targets"]

    monkeypatch.setattr(train_APO, "_compute_gae", spy)
    rollout = _rollout()
    new_state, _ = train_APO.update_agent(
        state, rollout, config, ExtensionStack(()), total_timesteps=1_000
    )

    critic = state.critic_state
    v = np.asarray(predict_value(critic, critic.params, rollout.obs))[0, ..., 0]
    v_next = np.asarray(predict_value(critic, critic.params, rollout.next_obs))
    v_next = v_next[0, ..., 0]
    rho = float(new_state.average_reward)
    r = np.asarray(rollout.reward)[..., 0]
    term = np.asarray(rollout.terminated)[..., 0]
    end = np.maximum(term, np.asarray(rollout.truncated)[..., 0])
    want = np.zeros((T + 1, N_ENVS))
    for t in reversed(range(T)):
        delta = r[t] - rho + (1 - term[t]) * v_next[t] - v[t]
        want[t] = delta + config.gae_lambda * (1 - end[t]) * want[t + 1]
    np.testing.assert_allclose(seen["gae"][..., 0], want[:T], atol=1e-5)
    np.testing.assert_allclose(seen["targets"][..., 0], want[:T] + v, atol=1e-5)
    # The rate is the plain mean over every step, episode ends included.
    np.testing.assert_allclose(rho, config.alpha * r.mean(), rtol=1e-6)
