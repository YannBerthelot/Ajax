"""agent.policy and agent.predict (ajax.policy).

Contract (owner's design, 2026-10-27; the user-facing fixes plan, section 3):

- policy = agent.policy(state, seed_index); carry = policy.reset(n_envs);
  action, carry = policy.act(obs, carry, done, reward, key) acts on the
  environment's own observations, in the environment's units, exactly as the
  agent's evaluation acts. Oracle: the evaluation itself. A run's logged
  evaluation of its final state, on the keys of that timestep
  (ajax.log.evaluation_keys), is reproduced, to rounding (RTOL), by a scan
  of policy.act over the same reset envs and the same step keys, its return
  counted to each env's first done as the evaluation counts it. The evaluations: the shared
  one (ajax.evaluate.evaluate, UDRL's included), the world models'
  (evaluate_policy: one episode an env) and APG's closed-loop rollout.
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

from ajax import APG, AVG, DQN, PPO, SAC, UDRL, DreamerV3
from ajax.agents.APG.APG import draw_systems
from ajax.environments.differentiable import closed_loop_rollout
from ajax.environments.interaction import reset, step
from ajax.environments.utils import (
    agent_episode_length,
    env_keys,
    get_state_action_shapes,
)
from ajax.evaluate import setup_environment
from ajax.log import evaluation_keys
from ajax.logging import LoggingConfig
from ajax.networks.memory import MemoryConfig
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY

#: The policy's program and the evaluation's are compiled apart, and XLA may
#: round each apart in the last bits (fused otherwise; CI runs Linux x86, not
#: macOS ARM): on macOS ARM their actions differ by at most 4 float32 ulps
#: in two of 47 runs (UDRL-gain, APG-pid), their returns by 2e-7 relative.
#: Skipping the normaliser's statistics in act moves PPO's return by 6e-2.
RTOL = 1e-5

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
    "UDRL": (
        UDRL,
        {"env_id": "Pendulum-v1", "n_steps": 8, "batch_size": 8},
        {"n_updates_per_iter": 4, "buffer_capacity": 8, "normalize_rewards": True},
        128,
    ),
    "DreamerV3": (
        DreamerV3,
        {**DREAMER_TINY, "env_id": "CartPole-v1", "n_envs": 2},
        {"train_ratio": 32, "replay_capacity": 80},
        80,
    ),
}


def build(name: str) -> tuple[Any, int]:
    cls, kwargs, more, steps = FAST[name]
    kwargs = {**kwargs, **more}
    if cls in (PPO, SAC):
        kwargs.update(NETS)
    if cls is UDRL:
        kwargs.update({"actor_architecture": SMALL})
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
    logs = getattr(logs, "evaluations", logs)  # APG's, in its every update's
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


def world_model_evaluation_return(agent: Any, policy: Any) -> jax.Array:
    """The world models' evaluation's return (ajax.evaluate.evaluate_policy),
    acting with policy.act: the training env rebuilt for EPISODES envs with
    its action repeat, reset as the shared evaluation resets, each step's
    key split in three (the carry, the policy's, the env's), one episode an
    env and its return counted to its first done."""
    env_args, state = agent.env_args, policy.agent_state
    eval_key, _ = evaluation_keys(state, state.collector_state.timestep)
    params, repeat = env_args.env_params, env_args.action_repeat
    env, mode, _ = setup_environment(
        env_args.env, params, EPISODES, None, 0.99, action_repeat=repeat
    )
    key, reset_key = jax.random.split(eval_key)
    obs, env_state = reset(env_keys(reset_key, mode, EPISODES), env, mode, params)

    def body(carry: tuple, _: Any) -> tuple:
        obs, env_state, policy_carry, key, ended, total = carry
        key, policy_key, step_key = jax.random.split(key, 3)
        action, policy_carry = policy.act(obs, policy_carry, key=policy_key)
        stepped = step(
            env_keys(step_key, mode, EPISODES), env_state, action, env, mode, params
        )
        obs, env_state, reward, terminated, truncated, _ = stepped
        total = total + jnp.where(ended, 0.0, reward)
        ended = ended | (terminated > 0) | (truncated > 0)
        return (obs, env_state, policy_carry, key, ended, total), None

    no, zeros = jnp.zeros(EPISODES, bool), jnp.zeros(EPISODES)
    first = (obs, env_state, policy.reset(EPISODES), key, no, zeros)
    length = agent_episode_length(env, params, repeat)
    final, _ = jax.lax.scan(body, first, None, length=length)
    return final[-1].mean()


def apg_evaluation_return(agent: Any, policy: Any) -> jax.Array:
    """APG's evaluation's return (its closed-loop rollout over its horizon,
    on EPISODES systems drawn from the evaluation key), acting with
    policy.act, an episode starting where the rollout resets."""
    state = policy.agent_state
    eval_key, _ = evaluation_keys(state, state.collector_state.timestep)
    env_args = agent.env_args
    params, key = draw_systems(eval_key, env_args, agent.system_class, EPISODES)

    def act(carry: Any, obs: jax.Array, resets: jax.Array) -> tuple:
        return policy.act(obs, carry, done=resets)

    rollout, _ = closed_loop_rollout(
        act, policy.reset(EPISODES), key, env_args.env, params, agent.horizon
    )
    return rollout.reward.sum(0).mean()


def evaluation_return(agent: Any, policy: Any) -> jax.Array:
    if isinstance(agent, APG):
        return apg_evaluation_return(agent, policy)
    if agent.name in ("DreamerV3", "TDMPC2"):
        return world_model_evaluation_return(agent, policy)
    return shared_evaluation_return(agent, policy)


@pytest.mark.parametrize("name", sorted(FAST))
def test_acting_with_the_policy_reproduces_the_evaluations_return(name: str) -> None:
    agent, steps = build(name)
    state, logged = evaluated_run(agent, steps)

    acted = evaluation_return(agent, agent.policy(state))

    np.testing.assert_allclose(float(acted), logged, rtol=RTOL)


#: The exact pins' tiny runs (tests/exact_pins.py) of every agent, and the
#: ways it acts: residual and gain transforms, features, a stateful expert,
#: the choosers, memory, the normalisers, a PID policy.
SLOW = (
    "APG",
    "APG-gain",
    "APG-gru",
    "APG-observe",
    "APG-pid",
    "APG-residual",
    "APO-discrete",
    "APO-norm",
    "ASAC",
    "AVG",
    "CrossQ-norm",
    "DAgger-discrete",
    "DAgger-stateful",
    "DQN-expert",
    "DreamerV3-continuous",
    "DreamerV3-features",
    "IQL-residual",
    "IQL-residual-jsrl",
    "PPO-gain",
    "PPO-gru",
    "PPO-observe",
    "PPO-pid-norm",
    "PQN-jsrl",
    "PQN-norm",
    "REDQ",
    "SAC-augment-state",
    "SAC-edge",
    "SAC-gru",
    "SAC-ibrl",
    "SAC-jsrl",
    "SAC-pid-norm",
    "TD3-gain",
    "TD3-gru",
    "TD3-observe",
    "TD3-residual",
    "TDMPC2",
    "TDMPC2-features",
    "TQC",
    "UDRL-gain",
    "UDRL-norm",
    "UDRL-observe",
    "UDRL-residual",
)


@pytest.mark.slow
@pytest.mark.parametrize("name", SLOW)
def test_every_agent_acts_as_its_evaluation(name: str) -> None:
    from tests.exact_pins import CASES

    cls, kwargs, steps = CASES[name]
    agent = cls(**kwargs)
    state, logged = evaluated_run(agent, steps)

    acted = evaluation_return(agent, agent.policy(state))

    np.testing.assert_allclose(float(acted), logged, rtol=RTOL)


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


@pytest.mark.parametrize("name", ["PPO-gru", "UDRL", "DreamerV3"])
def test_an_episode_start_restarts_what_the_policy_carries(name: str) -> None:
    """Episodes are independent: where done says obs starts an episode, the
    policy acts, and carries on, as from a reset, whatever it carried (a
    recurrent actor's memory, UDRL's command, DreamerV3's latent state)."""
    if name == "PPO-gru":
        memory = MemoryConfig(kind="gru", hidden_size=4)
        agent = PPO("Pendulum-v1", n_envs=2, n_steps=8, memory=memory, **NETS)
    else:
        agent = build(name)[0]
    state, _ = agent.train(seed=0, n_timesteps=0)
    policy = agent.policy(state)
    width = get_state_action_shapes(agent.env_args.env)[0][-1]
    observations = jax.random.normal(jax.random.PRNGKey(0), (4, 2, width))
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    reward, carry = jnp.array([1.0, -2.0]), policy.reset(2)
    for obs, key in zip(observations[:3], keys[:3], strict=True):
        _, carry = policy.act(obs, carry, jnp.zeros(2, bool), reward, key=key)
    fresh = policy.act(observations[3], policy.reset(2), key=keys[3])

    started = policy.act(observations[3], carry, jnp.ones(2, bool), reward, key=keys[3])

    # The action and what it carries on: an action may saturate at a bound.
    leaves = zip(jax.tree.leaves(started), jax.tree.leaves(fresh), strict=True)
    for got, expected in leaves:
        np.testing.assert_array_equal(got, expected)


def test_predict_of_the_own_iteration_family_is_the_policys_action() -> None:
    agent = APG("Pendulum-v1", n_envs=2, horizon=8, actor_architecture=SMALL)
    state, _ = agent.train(seed=0, n_timesteps=32)
    observation = jnp.array([1.0, 0.0, 0.5])
    policy = agent.policy(state)

    expected, _ = policy.act(observation[None], policy.reset(1))

    np.testing.assert_array_equal(agent.predict(state, observation), expected[0])


def test_a_commanded_policy_asks_for_the_reward_with_done() -> None:
    agent, _ = build("UDRL")
    state, _ = agent.train(seed=0, n_timesteps=0)
    policy = agent.policy(state)
    action, carry = policy.act(jnp.zeros((1, 3)), policy.reset(1))

    with pytest.raises(ValueError, match="reward"):
        policy.act(jnp.zeros((1, 3)), carry, done=jnp.array([False]))
    with pytest.raises(ValueError, match="agent.policy"):
        agent.predict(state, jnp.zeros(3))


def test_the_multi_task_agent_refuses_a_policy() -> None:
    from tests.agents.test_agent_pins import _multitask

    agent = _multitask()

    with pytest.raises(NotImplementedError, match="task"):
        agent.policy(None)
    with pytest.raises(NotImplementedError, match="task"):
        agent.predict(None, jnp.zeros(3))


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
