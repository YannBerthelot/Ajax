"""DQN's two equations, on a real Q-network at its initialisation.

Contract (docs/algorithms/dqn.md, Notes, "Target"; ajax.agents.critics):
  - the target is ``scale r + gamma (1 - terminated) max_a' Q'(s', a')``,
    with the target network's values, and its next action is that maximum's;
  - the loss compares the target with ``Q(s, a)`` at the action taken.
The network's values are read with ``predict_q``; the maximum and the
action's column are taken here with NumPy.
"""

import jax
import numpy as np

from ajax.agents.critics import compute_mse_td_loss, compute_q_loss
from ajax.agents.DQN.DQN import DQN, compute_dqn_td_target
from ajax.environments.utils import get_state_action_shapes
from ajax.networks.q_networks import predict_q

GAMMA, REWARD_SCALE = 0.9, 2.0
REWARDS = np.array([[1.0], [0.0], [2.0], [-1.0]], np.float32)
TERMINATED = np.array([[0.0], [1.0], [0.0], [0.0]], np.float32)
ACTIONS = np.array([[0], [1], [1], [0]])


def q_network_and_observations():
    agent = DQN("CartPole-v1", architecture=("16", "relu"))
    state = agent.init(jax.random.PRNGKey(0), agent.make_context(1)[1])
    obs_shape, _ = get_state_action_shapes(agent.env_args.env)
    observations = jax.random.normal(jax.random.PRNGKey(1), (4, *obs_shape))
    return state.actor_state, observations


def test_the_target_bootstraps_on_the_largest_next_value_unless_terminated():
    q_state, next_observations = q_network_and_observations()
    next_q = np.asarray(predict_q(q_state, q_state.target_params, next_observations))
    # Two actions whose values differ, so the maximum is not the mean.
    assert np.all(np.abs(next_q[:, 0] - next_q[:, 1]) > 1e-3)

    target, next_actions, next_values = compute_dqn_td_target(
        q_state, next_observations, TERMINATED, REWARDS, GAMMA, REWARD_SCALE
    )

    largest = next_q.max(axis=1, keepdims=True)
    expected = REWARD_SCALE * REWARDS + GAMMA * (1.0 - TERMINATED) * largest
    # float32 products and sums of values of order 1: a few ulps.
    np.testing.assert_allclose(target, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(next_values, largest, rtol=1e-6)
    np.testing.assert_array_equal(next_actions, next_q.argmax(axis=1)[:, None])


def test_the_loss_reads_q_at_the_action_taken():
    q_state, observations = q_network_and_observations()
    q_all = np.asarray(predict_q(q_state, q_state.params, observations))
    q_taken = q_all[np.arange(4), ACTIONS[:, 0]][:, None]
    targets = np.zeros((4, 1), np.float32)

    loss, aux = compute_q_loss(
        q_state.params, q_state, observations, ACTIONS, targets, compute_mse_td_loss
    )

    # Against a target of 0 the squared error is the mean of Q(s, a)^2.
    np.testing.assert_allclose(loss, np.mean(q_taken**2), rtol=1e-5)
    np.testing.assert_allclose(aux.q_pred_mean, [np.mean(q_taken)], rtol=1e-5)
