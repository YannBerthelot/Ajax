"""TargetGym's PID experts drive an Ajax agent through their batched form.

Contract (docs/guide/expert-guidance.md): an agent's stateful expert answers
``init_state(n_envs)`` and ``expert(state, obs) -> (action, state)``.
TargetGym's ``make_expert`` returns a user-facing object of another form
(``reset``, ``step``); ``REGISTRY[name].make_pid_controller().batched()`` is
the adapter. The realistic bug: a TargetGym release drops or renames that
form, and the benchmark's arms stop running. Runs with the ``plane`` group.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC

target_gym = pytest.importorskip("target_gym")
registry = pytest.importorskip("target_gym.registry")

TASK = "four_tank"
SHORT = {"learning_starts": 4, "batch_size": 4, "buffer_size": 64}


def test_the_batched_pid_is_a_stateful_expert_of_one_action_per_env() -> None:
    env, params = target_gym.make(TASK)
    expert = registry.REGISTRY[TASK].make_pid_controller().batched()
    n_actions = env.action_space(params).shape[0]
    obs = jnp.stack([env.reset(jax.random.PRNGKey(i), params)[0] for i in range(3)])
    action, state = expert(expert.init_state(3), obs)
    assert action.shape == (3, n_actions)
    jax.tree.map(lambda x: np.testing.assert_equal(x.shape[0], 3), state)


def test_an_agent_trains_with_it_and_stores_its_actions() -> None:
    env, params = target_gym.make(TASK)
    expert = registry.REGISTRY[TASK].make_pid_controller().batched()
    agent = SAC(env, env_params=params, n_envs=2, expert_policy=expert, **SHORT)
    state, _ = agent.train(seed=0, n_timesteps=16)
    replay = state.collector_state.buffer_state
    assert replay is not None
    a_expert = np.asarray(replay.experience["a_expert"])[..., :8, :]
    assert np.isfinite(a_expert).all() and np.abs(a_expert).max() <= 1.0 + 1e-6
    assert a_expert.std() > 0  # the PID's own actions, not an empty column
