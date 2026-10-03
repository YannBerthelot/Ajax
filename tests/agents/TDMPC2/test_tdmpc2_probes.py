"""Ajax-local probes for TD-MPC2 (``docs/world_models/DESIGN.md`` §8).

The stock probing envs (``tests/agents/test_probing.py``) end after 1-2 steps;
paper-era TD-MPC2 is for fixed-length, non-terminating tasks and its training
slices need ``T + 1 >= H + 1`` rows, so it is probed on fixed-length
non-terminating toy envs instead (:mod:`.toy_envs`, ``T = 10``):

* constant reward 1: the episode end is a time limit, bootstrapped, so the
  learned value is ``sum_t gamma^t = 1 / (1 - gamma)`` for every action
  (discounting, value loss, target network, two-hot decoding);
* reward = action: the reward model learns ``r = a`` and the planner, in
  ``eval_mode``, plays the best action +1 (reward head, MPPI, policy prior).

Reduced config: a tiny model and planner, ``gamma = 0.8`` (value 5),
``tau = 0.1`` and ``learning_rate = 1e-3`` so that ~1000 updates converge.
The value tolerance is 10 % of the target: across seeds 0-7 the error stays
below 0.31, while the failure modes are far off (no time-limit bootstrap
gives about 3.2 at this constant observation, a discount of 0.9 gives 10).
Backend differences (CI runs Linux x86) act like a seed change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import TDMPC2
from ajax.agents.TDMPC2 import core
from ajax.agents.TDMPC2.train_TDMPC2 import planner_policy

from .toy_envs import ConstantRewardEnv, RewardIsActionEnv

PROBE = {
    "enc_dim": 32,
    "mlp_dim": 32,
    "latent_dim": 16,
    "num_q": 2,
    "batch_size": 32,
    "num_samples": 64,
    "num_elites": 8,
    "num_pi_trajs": 4,
    "iterations": 4,
    "gamma": 0.8,
    "tau": 0.1,
    "learning_rate": 1e-3,
    "seed_steps": 100,
}
SEEDS = [0, 1]
N_TIMESTEPS = 1000  # 2 envs x 500 steps: ~1000 updates


def _train(env):
    agent = TDMPC2(env, n_envs=2, **PROBE)
    state, _ = agent.train(seed=SEEDS, n_timesteps=N_TIMESTEPS, num_episode_test=1)
    return agent, state


def _per_seed(state):
    return [jax.tree.map(lambda x, i=i: x[i], state) for i in range(len(SEEDS))]


def _latents(state, n):
    wm = state.world_model_state
    return wm.apply_fn({"params": wm.params}, jnp.ones((n, 1)), method="encode")


@pytest.fixture(scope="module")
def constant_reward():
    return _train(ConstantRewardEnv(length=10))


@pytest.fixture(scope="module")
def reward_is_action():
    return _train(RewardIsActionEnv(length=10))


def test_value_is_the_discounted_sum_through_time_limits(constant_reward):
    agent, state = constant_reward
    assert state.ext_state == ()  # no extensions: no extension state
    target = 1.0 / (1.0 - agent.gamma)
    actions = jnp.linspace(-1.0, 1.0, 5)[:, None]
    for seed_state in _per_seed(state):
        wm = seed_state.world_model_state
        z = _latents(seed_state, len(actions))
        q = agent.agent_config.two_hot.decode(
            core.q_logits(wm.apply_fn, wm.params, z, actions)
        )  # [num_q, 5]
        np.testing.assert_allclose(q, target, rtol=0.1)
        assert np.ptp(np.asarray(q)) < 0.1  # the same for every action


def test_planner_plays_the_rewarding_action(reward_is_action):
    agent, state = reward_is_action
    actions = jnp.linspace(-1.0, 1.0, 5)[:, None]
    for i, seed_state in enumerate(_per_seed(state)):
        wm = seed_state.world_model_state
        z = _latents(seed_state, len(actions))
        reward = agent.agent_config.two_hot.decode(
            wm.apply_fn({"params": wm.params}, z, actions, method="reward_logits")
        )
        np.testing.assert_allclose(reward[2:], [0.0, 0.5, 1.0], atol=0.1)
        n = 8
        action, _, _ = planner_policy(
            jnp.zeros((n, agent.agent_config.horizon, 1)),
            jnp.ones((n, 1)),
            jnp.ones(n, bool),
            jax.random.PRNGKey(i),
            wm_params=wm.params,
            pi_params=seed_state.actor_state.params,
            config=agent.agent_config,
            gamma=agent.gamma,
            eval_mode=True,
        )
        assert np.all(np.asarray(action) > 0.9)
