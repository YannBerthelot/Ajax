"""agent.policy and agent.predict (ajax.policy).

Contract (owner's design, 2026-10-27; the user-facing fixes plan, section 3):

- policy = agent.policy(state, seed_index); carry = policy.reset(n_envs);
  action, carry = policy.act(obs, carry, done, reward, key) acts on the
  environment's own observations, in the environment's units, exactly as the
  agent's evaluation acts. Oracle: the evaluation itself. A run's logged
  evaluation of its final state, on the keys of that timestep
  (ajax.log.evaluation_keys), is reproduced exactly by a scan of policy.act
  over the same reset envs and the same step keys, its return counted to each
  env's first done as the evaluation counts it.
- predict(state, obs, seed_index) is policy.act's action from a reset, for a
  single observation or a batch; a policy carrying memory between steps, and
  deterministic=False, refuse.
- act compiles once: acting again, on new observations, compiles nothing.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import AVG, DQN, PPO, SAC
from ajax.environments.interaction import reset, step
from ajax.environments.utils import env_keys
from ajax.evaluate import setup_environment
from ajax.log import evaluation_keys
from ajax.logging import LoggingConfig
from ajax.networks.memory import MemoryConfig

SMALL = ("16", "relu")
NETS: dict[str, Any] = {"actor_architecture": SMALL, "critic_architecture": SMALL}
EPISODES = 3  # the evaluation's episodes, num_episode_test

#: (agent, its arguments, n_timesteps), each at 128 gradient steps or fewer.
#: PPO and SAC normalise their observations: PPO in its controller, from its
#: collector's statistics, SAC in its actor.
FAST = {
    "PPO": (
        PPO,
        {"env_id": "Pendulum-v1", "n_envs": 4, "n_steps": 16, "batch_size": 32},
        {"n_epochs": 1, "normalize_observations": True},
        128,
    ),
    "SAC": (
        SAC,
        {"env_id": "Pendulum-v1", "learning_starts": 64, "batch_size": 8},
        {"buffer_size": 128, "normalize_observations": True},
        128,
    ),
    "DQN": (
        DQN,
        {"env_id": "CartPole-v1", "architecture": SMALL, "learning_starts": 64},
        {"batch_size": 8, "buffer_size": 128, "target_update_interval": 10},
        128,
    ),
}


def build(name: str) -> tuple[Any, int]:
    cls, kwargs, more, steps = FAST[name]
    kwargs = {**kwargs, **more}
    if cls is not DQN:
        kwargs.update(NETS)
    return cls(**kwargs), steps


def evaluated_run(agent: Any, steps: int) -> tuple[Any, float]:
    """The agent's final state and its logged evaluation return there. A
    first run finds the final timestep (an on-policy run collects one
    rollout past its budget); the second, the same run, evaluates there
    alone, its horizon past it."""
    horizon = 4 * steps
    state, _ = agent.train(seed=0, n_timesteps=steps, total_timesteps=horizon)
    final = int(state.collector_state.timestep[0])
    logging_config = LoggingConfig(log_frequency=final, backends=())
    state, logs = agent.train(
        seed=0,
        n_timesteps=steps,
        total_timesteps=horizon,
        num_episode_test=EPISODES,
        logging_config=logging_config,
    )
    timesteps = list(np.asarray(logs["timestep"][0]))
    assert final in timesteps, f"no evaluation at the final timestep {final}"
    column = timesteps.index(final)
    return state, float(logs["Eval/episodic mean reward"][0, column])


def shared_evaluation_return(agent: Any, policy: Any) -> jax.Array:
    """The shared evaluation's return (ajax.evaluate.evaluate), acting with
    policy.act: the training env rebuilt for EPISODES envs, reset on one half
    of the evaluation key, each step's env keys split off the other, the
    policy's key the carry kept (the choosers'), and each env's return
    counted to its first done."""
    env_args, state = agent.env_args, policy.agent_state
    eval_key, _ = evaluation_keys(state, state.collector_state.timestep)
    env, mode, _ = setup_environment(
        env_args.env, env_args.env_params, EPISODES, None, 0.99
    )
    params = env_args.env_params
    rng, reset_key = jax.random.split(eval_key)
    obs, env_state = reset(env_keys(reset_key, mode, EPISODES), env, mode, params)

    def body(carry: tuple, _: Any) -> tuple:
        obs, env_state, policy_carry, rng, done, reward, ended, total = carry
        rng, step_key = jax.random.split(rng)
        action, policy_carry = policy.act(obs, policy_carry, done, reward, key=rng)
        stepped = step(
            env_keys(step_key, mode, EPISODES), env_state, action, env, mode, params
        )
        obs, env_state, reward, terminated, truncated, _ = stepped
        total = total + reward * (1 - ended)
        done = jnp.logical_or(terminated, truncated)
        return (
            obs,
            env_state,
            policy_carry,
            rng,
            done,
            reward,
            ended | done,
            total,
        ), None

    zeros, no = jnp.zeros(EPISODES), jnp.zeros(EPISODES, bool)
    first = (obs, env_state, policy.reset(EPISODES), rng, no, zeros, no, zeros)
    length = int(params.max_steps_in_episode)
    final, _ = jax.lax.scan(body, first, None, length=length)
    return final[-1].mean()


@pytest.mark.parametrize("name", sorted(FAST))
def test_acting_with_the_policy_reproduces_the_evaluations_return(name: str) -> None:
    agent, steps = build(name)
    state, logged = evaluated_run(agent, steps)

    acted = shared_evaluation_return(agent, agent.policy(state))

    assert float(acted) == logged


#: The exact pins' tiny runs (tests/exact_pins.py) of every agent on the
#: shared evaluation, and the ways it acts: residual and gain transforms,
#: features, a stateful expert, the choosers, memory, the normalisers, a PID
#: policy.
SLOW = (
    "APO-discrete APO-norm ASAC AVG CrossQ-norm DAgger-discrete DAgger-stateful"
    " DQN-expert IQL-residual IQL-residual-jsrl PPO-gain PPO-gru PPO-observe"
    " PPO-pid-norm PQN-jsrl PQN-norm REDQ SAC-augment-state SAC-edge SAC-gru"
    " SAC-ibrl SAC-jsrl SAC-pid-norm TD3-gain TD3-gru TD3-observe TD3-residual TQC"
).split()


@pytest.mark.slow
@pytest.mark.parametrize("name", SLOW)
def test_every_agent_acts_as_its_evaluation(name: str) -> None:
    from tests.exact_pins import CASES

    cls, kwargs, steps = CASES[name]
    agent = cls(**kwargs)
    state, logged = evaluated_run(agent, steps)

    acted = shared_evaluation_return(agent, agent.policy(state))

    assert float(acted) == logged


@pytest.mark.parametrize("name", ["PPO", "SAC"])
def test_predict_is_the_policys_action_from_a_reset(name: str) -> None:
    """For a feed-forward agent of the on-policy and replay families, on
    seed 1 of two: one observation and a batch."""
    agent, _ = build(name)
    state, _ = agent.train(seed=[0, 1], n_timesteps=64)
    observations = jnp.array([[1.0, 0.0, 0.5], [0.0, 1.0, -3.0]])
    policy = agent.policy(state, seed_index=1)
    # Each on its own batch: XLA may round a batch of one apart from two.
    expected_one, _ = policy.act(observations[:1], policy.reset(1))
    expected, _ = policy.act(observations, policy.reset(2))

    one = agent.predict(state, observations[0], seed_index=1)
    batch = agent.predict(state, observations, seed_index=1)

    np.testing.assert_array_equal(one, expected_one[0])
    np.testing.assert_array_equal(batch, expected)


def test_predict_of_the_incremental_family_is_the_policys_action() -> None:
    agent = AVG("Pendulum-v1", **NETS)
    state, _ = agent.train(seed=0, n_timesteps=8)
    observation = jnp.array([1.0, 0.0, 0.5])
    policy = agent.policy(state)

    expected, _ = policy.act(observation[None], policy.reset(1))

    np.testing.assert_array_equal(agent.predict(state, observation), expected[0])


def test_predict_refuses_a_policy_with_memory_and_sampling() -> None:
    memory = MemoryConfig(kind="gru", hidden_size=4)
    agent = PPO("Pendulum-v1", n_envs=2, n_steps=8, memory=memory, **NETS)
    state, _ = agent.train(seed=0, n_timesteps=0)
    observation = jnp.zeros(3)

    with pytest.raises(ValueError, match="agent.policy"):
        agent.predict(state, observation)
    with pytest.raises(ValueError, match="deterministic"):
        build("PPO")[0].predict(state, observation, deterministic=False)


def test_acting_again_compiles_nothing(caplog: pytest.LogCaptureFixture) -> None:
    """Once act has run, with and without the optional inputs, acting on new
    observations, ends and rewards compiles nothing."""
    agent, _ = build("PPO")
    state, _ = agent.train(seed=0, n_timesteps=64)
    policy = agent.policy(state)
    carry = policy.reset(2)
    observations = [jnp.full((2, 3), float(i)) for i in range(4)]
    keys = [jax.random.PRNGKey(i) for i in range(4)]
    done, reward = jnp.array([False, True]), jnp.array([-1.0, -2.0])
    other_done, other_reward = jnp.array([True, False]), jnp.array([0.5, 3.0])
    action, carry = policy.act(observations[0], carry)
    action, carry = policy.act(observations[1], carry, done, reward, key=keys[1])

    with jax.log_compiles(True), caplog.at_level(logging.WARNING):
        action, carry = policy.act(observations[2], carry, other_done, other_reward)
        action, carry = policy.act(observations[3], carry, done, reward, key=keys[3])

    assert not [r for r in caplog.records if "Compiling" in r.getMessage()]
