"""What TD-MPC2 learns on the terminating chain s0 -0-> s1 -0-> s2 -1-> s3
(``tests.agents.TDMPC2.toy_envs.ChainEnv``, time limit 10): the episodic mode
reads the chain's values and its termination, the paper-era mode the
continuing chain's (deviation T29). Trains about 1,000 updates: slow tier.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import TDMPC2
from ajax.agents.base import ActorCritic
from ajax.agents.TDMPC2 import core
from tests.agents.TDMPC2.test_tdmpc2_episodic import CHAIN, CHAIN_GAMMA
from tests.agents.TDMPC2.toy_envs import ChainEnv

# The terminating chain s0 -0-> s1 -0-> s2 -1-> s3 (terminal), time limit 10:
# the values are gamma^2, gamma, 1 at s0, s1, s2. The paper-era agent does
# not see the termination (gymnax auto-resets to s0 mid-slot), so it learns
# the continuing chain with reward 1 every third step, gamma^(2 - t) / (1 -
# gamma^3). Calibrated on seeds 0-3 (1072 updates): the episodic values
# within 0.008 of the truth and the termination probability below 0.002 or
# above 0.999; the paper-era values within 0.035 of the continuing chain's.
TRUTH = CHAIN_GAMMA ** np.array([2.0, 1.0, 0.0])
CONTINUING = TRUTH / (1 - CHAIN_GAMMA**3)
CHAIN_SEEDS = [0, 1]
ROWS = 1400  # 700 ticks of 2 envs


@pytest.fixture(scope="module")
def episodic_chain():
    agent = TDMPC2(ChainEnv(), episodic=True, buffer_size=4000, **CHAIN)
    state, _ = agent.train(seed=CHAIN_SEEDS, n_timesteps=ROWS, num_episode_test=1)
    return agent, state


def _chain_values(agent, state):
    """Per seed: Q at s0, s1, s2 (every head, action 0) and the termination
    probability after one step from each."""
    out = []
    for i in range(len(CHAIN_SEEDS)):
        wm = jax.tree.map(lambda x, i=i: x[i], state.world_model_state)
        apply = functools.partial(wm.apply_fn, {"params": wm.params})
        z = apply(jax.nn.one_hot(jnp.arange(3), 4), method="encode")
        a = jnp.zeros((3, 1))
        q = agent.model_config.two_hot.decode(
            core.q_logits(wm.apply_fn, wm.params, z, a)
        )
        p = None
        if agent.model_config.episodic:
            p = jax.nn.sigmoid(
                apply(apply(z, a, method="next"), method="termination_logits")
            )
        out.append((np.asarray(q), p))
    return out


def test_the_episodic_agent_learns_the_terminating_chains_values(episodic_chain):
    agent, state = episodic_chain
    for q, p in _chain_values(agent, state):
        np.testing.assert_allclose(q, np.broadcast_to(TRUTH, q.shape), atol=0.05)
        np.testing.assert_array_equal(np.asarray(p) > 0.5, [False, False, True])


def test_the_paper_era_agent_learns_a_continuing_chain_instead():
    """Without the episodic mode the agent refuses the chain after training
    (T10); the base class's ``train`` skips that check, to read what it
    learned: the values of a chain that never ends."""
    agent = TDMPC2(ChainEnv(), **CHAIN)
    budget = ROWS * 3 // 4  # about as many updates as the episodic run
    replay, _ = agent.prepare_replay(0, agent.count_ticks(budget, 0), None)
    state, _ = ActorCritic.train(
        agent,
        seed=CHAIN_SEEDS,
        n_timesteps=budget,
        num_episode_test=1,
        replay=replay,
        start_tick=0,
    )
    assert np.all(np.asarray(state.collector_state.n_offschedule_dones) > 0)
    for q, _ in _chain_values(agent, state):
        np.testing.assert_allclose(q, np.broadcast_to(CONTINUING, q.shape), atol=0.1)
        assert np.all(q > TRUTH + 0.5)
