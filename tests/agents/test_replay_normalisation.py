"""A replay agent's normalisation reads each replayed row with the current
statistics, as SB3's replay buffer does under ``VecNormalize``: the replay
keeps the env's own observations, which the agent normalises itself (the
collector's running statistics, applied by its networks). A row normalised
when collected would keep the statistics of then.

On the identity task (tests/environments/identity_env.py) the observation
``[t, c]`` reads its own units back: ``t`` counts the episode's steps.
"""

import inspect
from typing import Any

import jax
import numpy as np
import pytest

import ajax
from ajax.environments.create import strip_ajax_wrappers
from tests.agents.test_agent_contract import CASES
from tests.environments.identity_env import LIMIT, IdentityEnv

#: The agents that replay stored transitions.
REPLAYING = sorted(
    n
    for n in ajax.__all__
    if inspect.isclass(getattr(ajax, n))
    and getattr(getattr(ajax, n), "replays_transitions", False)
)


def test_the_replaying_agents_are_the_replay_buffer_ones() -> None:
    assert REPLAYING == ["ASAC", "DQN", "REDQ", "SAC", "TD3"]


@pytest.mark.parametrize("name", REPLAYING)
def test_a_replay_agent_keeps_the_envs_own_observations(name: str) -> None:
    """The env passes its observations through unnormalised, the replay
    stores them so, and the networks read the collector's current
    statistics."""
    case = CASES[name]
    env = IdentityEnv(discrete=name == "DQN")
    agent = case.cls(env_id=env, n_envs=2, normalize_observations=True, **case.kwargs)
    assert agent.env_args.normalize_obs_in_agent
    assert not strip_ajax_wrappers(agent.env_args.env)[1].get("normalize_obs", False)
    state, _ = agent.train(seed=0, n_timesteps=24)
    state = jax.tree.map(lambda x: x[0], state)  # the one seed
    assert state.collector_state.buffer_state is not None
    rows = state.collector_state.buffer_state.experience
    written = int(state.collector_state.timestep) // 2
    t = np.asarray(rows["obs"])[:, :written, 0]
    np.testing.assert_array_equal(t, np.round(t))
    assert t.min() == 0 and t.max() <= LIMIT - 1
    stats = state.collector_state.obs_norm_info
    assert stats is not None and int(np.sum(stats.count)) > 0
    for network in (state.actor_state, state.critic_state):
        for mine, theirs in zip(
            jax.tree.leaves(network.obs_norm_info), jax.tree.leaves(stats), strict=True
        ):
            np.testing.assert_array_equal(mine, theirs)


def test_an_on_policy_agent_keeps_the_envs_normaliser() -> None:
    """An agent learning from its current rollout reads observations
    normalised when collected, by the env, as before."""
    agent = ajax.PPO(env_id=IdentityEnv(), normalize_observations=True)
    assert not agent.env_args.normalize_obs_in_agent
    assert strip_ajax_wrappers(agent.env_args.env)[1]["normalize_obs"]


@pytest.mark.parametrize("name", REPLAYING)
def test_a_replay_agent_stores_the_envs_own_rewards(name: str) -> None:
    """The replay stores the env's own reward (on the identity task the
    executed action: twice the agent's on Box(-2, 2), itself on
    Discrete(2)); a replayed batch divides it by the reward normaliser's
    current statistics."""
    from ajax.buffers.utils import get_batch_from_buffer
    from ajax.environments.interaction import reward_statistics

    case = CASES[name]
    discrete = name == "DQN"
    env = IdentityEnv(discrete=discrete)
    # Any: the replay agents hold a buffer, which ActorCritic does not declare.
    agent: Any = case.cls(env_id=env, n_envs=2, normalize_rewards=True, **case.kwargs)
    state, _ = agent.train(seed=0, n_timesteps=24)
    state = jax.tree.map(lambda x: x[0], state)  # the one seed
    buffer_state = state.collector_state.buffer_state
    assert buffer_state is not None
    rows = buffer_state.experience
    written = int(state.collector_state.timestep) // 2
    paid = 1.0 if discrete else 2.0
    np.testing.assert_allclose(
        np.asarray(rows["reward"])[:, :written, 0],
        paid * np.asarray(rows["action"])[:, :written, 0],
        rtol=1e-6,
    )
    stats = reward_statistics(state.collector_state.env_state)
    assert stats is not None
    std = np.sqrt(np.mean(np.asarray(stats.mean_2) / np.asarray(stats.count)) + 1e-8)
    batch = get_batch_from_buffer(agent.buffer, buffer_state, jax.random.PRNGKey(1))
    replayed = get_batch_from_buffer(
        agent.buffer, buffer_state, jax.random.PRNGKey(1), reward_stats=stats
    )
    np.testing.assert_allclose(replayed.reward, batch.reward / std, rtol=1e-5)
