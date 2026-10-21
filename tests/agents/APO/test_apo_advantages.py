"""APO's differential GAE stops at episode ends, as APO's official code does.

``xtma/apo`` (``apo/algos/utils.py``, ``generalized_advantage_estimation``,
run with ``discount = 1``) computes, backwards over the rollout,
``delta_t = r_t - eta + V(s_{t+1}) (1 - d_t) - V(s_t)`` and
``A_t = delta_t + lambda (1 - d_t) A_{t+1}``, ``d_t`` gym's done flag, a
time limit included, and drops a time-limited row from the loss
(``apo/algos/apg/base.py``, ``bootstrap_timelimit``). One update runs on a
synthetic rollout where env 0 terminates and env 1 is truncated (its final
observation differs from the reset one); the advantages and value targets
it computes are checked against that recursion in NumPy.
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
        delta = r[t] - rho + (1 - end[t]) * v_next[t] - v[t]
        want[t] = delta + config.gae_lambda * (1 - end[t]) * want[t + 1]
    np.testing.assert_allclose(seen["gae"][..., 0], want[:T], atol=1e-5)
    np.testing.assert_allclose(seen["targets"][..., 0], want[:T] + v, atol=1e-5)
    # The rate is the plain mean over every step, episode ends included; the
    # first update sets it to that mean (xtma/apo seeds eta with it).
    np.testing.assert_allclose(rho, r.mean(), rtol=1e-6)


def test_the_rate_and_the_bias_start_at_the_first_rollouts_means() -> None:
    """As xtma/apo (apo/algos/apg/base.py:43-50): the first update sets
    rho and b to the rollout's mean reward and mean value, and the EMA at
    rate alpha runs from the second update on."""
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
    rollout = _rollout()
    means = []
    for batch in (rollout, rollout.replace(reward=rollout.reward + 1.0)):
        critic = state.critic_state
        v = predict_value(critic, critic.params, batch.obs)
        means.append((float(jnp.mean(batch.reward)), float(jnp.mean(v))))
        state, _ = train_APO.update_agent(
            state, batch, config, ExtensionStack(()), total_timesteps=1_000
        )
        if len(means) == 1:
            np.testing.assert_allclose(state.average_reward, means[0][0], rtol=1e-6)
            np.testing.assert_allclose(state.b, means[0][1], rtol=1e-5)
    a = config.alpha
    rho = (1 - a) * means[0][0] + a * means[1][0]
    b = (1 - a) * means[0][1] + a * means[1][1]
    np.testing.assert_allclose(state.average_reward, rho, rtol=1e-6)
    np.testing.assert_allclose(state.b, b, rtol=1e-5)


def test_a_time_limited_step_leaves_the_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """``bootstrap_timelimit`` (xtma/apo ``base.py:73-77``): the losses average
    over the rows that did not hit a time limit (rlpyt's ``valid_mean``), so a
    truncated row's error moves neither; every minibatch of the update gets
    ``1 - truncated``."""
    agent = APO(
        "Pendulum-v1",
        n_envs=N_ENVS,
        n_steps=T,
        num_minibatches=1,
        n_epochs=2,
        critic_architecture=("8", "tanh"),
        actor_architecture=("8", "tanh"),
    )
    state = train_APO.init_APO(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    critic, obs = state.critic_state, jax.random.normal(jax.random.PRNGKey(2), (3, 3))
    valid = jnp.array([[1.0], [1.0], [0.0]])

    def loss(targets: jax.Array) -> float:
        args = (critic.params, critic, obs, targets, 0.1, 0.5, 1.0, valid)
        return float(train_APO.value_loss_function(*args)[0])

    assert loss(jnp.array([[1.0], [2.0], [3.0]])) == loss(
        jnp.array([[1.0], [2.0], [99.0]])
    )
    seen: list = []
    value_loss = train_APO.value_loss_function

    def spy(*args: Any) -> Any:
        jax.debug.callback(lambda v: seen.append(np.asarray(v)), args[-1])
        return value_loss(*args)

    monkeypatch.setattr(train_APO, "value_loss_function", spy)
    config = cast(APOConfig, agent.agent_config)
    rollout = _rollout()
    train_APO.update_agent(state, rollout, config, ExtensionStack(()), 1_000)
    assert len(seen) == 2  # one minibatch, two epochs
    want = sorted(1.0 - np.asarray(rollout.truncated).reshape(-1))
    for v in seen:
        np.testing.assert_array_equal(sorted(v.reshape(-1)), want)
