"""REINFORCE's update, against Williams (1992) and Spinning Up's VPG.

Contract, from the method and its reference (spinningup @038665d):

* without a baseline, each step's weight is its reward-to-go
  G_t = sum_k gamma^k r_{t+k}, stopped at its episode's end, termination or
  time limit (2_rtg_pg.py:17-22); the steps of an episode the rollout cuts
  weigh nothing (Ajax's choice for 2_rtg_pg's whole-episode batches);
* with a baseline, the critic's target is the reward-to-go bootstrapped
  with V(s_{t+1}) where an episode is cut (rollout end, time limit) and
  not where it terminates (vpg.py:42-68, :295-298), and the advantage is
  GAE(lambda) of V;
* one policy step moves the policy up the weighted log-likelihood: a
  rewarded action becomes likelier, a punished one less likely;
* the critic is trained only with a baseline.

Every expected value below is worked out by hand from those rules.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.REINFORCE.REINFORCE import REINFORCE, compute_returns, normalise
from ajax.environments.interaction import get_pi
from ajax.state import Transition

GAMMA = 0.5


def _rollout(rewards: list, terminated: list, truncated: list | None = None, **kw):
    """A (T, N, 1) rollout from per-env lists, time along each list."""
    as_column = lambda rows: jnp.asarray(rows, jnp.float32).T[..., None]  # noqa: E731
    reward = as_column(rewards)
    flags = as_column(truncated) if truncated else jnp.zeros_like(reward)
    return Transition(
        obs=kw.get("obs", jnp.zeros(reward.shape[:2] + (4,))),
        action=kw.get("action", jnp.zeros(reward.shape[:2], jnp.int32)),
        reward=reward,
        terminated=as_column(terminated),
        truncated=flags,
        next_obs=jnp.zeros(reward.shape[:2] + (4,)),
        log_prob=jnp.zeros(reward.shape[:2]),
    )


def test_reward_to_go_stops_at_each_episode_end_and_drops_cut_episodes() -> None:
    """Catches a return that leaks across an auto-reset, sums the whole
    episode instead of the rewards from t on, or trains on a cut episode."""
    # env 0: one episode, terminated at t=3; env 1: ends at t=1, then an
    # episode the rollout cuts (its 5s must not reach t=1).
    rollout = _rollout([[1, 1, -1, 0], [2, 1, 5, 5]], [[0, 0, 0, 1], [0, 1, 0, 0]])
    zeros = jnp.zeros_like(rollout.reward)
    returns, advantages, weights = compute_returns(
        rollout, zeros, zeros, GAMMA, 0.97, baseline=False
    )
    # G = r_t + 0.5 G_{t+1}: 0; -1; 1 - 0.5 = 0.5; 1 + 0.25 = 1.25 | 1; 2 + 0.5.
    np.testing.assert_allclose(returns[:, 0, 0], [1.25, 0.5, -1.0, 0.0])
    np.testing.assert_allclose(returns[:2, 1, 0], [2.5, 1.0])
    np.testing.assert_array_equal(advantages, returns)
    np.testing.assert_array_equal(weights[:, :, 0].T, [[1, 1, 1, 1], [1, 1, 0, 0]])


def test_a_time_limit_ends_the_return_without_a_baseline() -> None:
    """Catches a plain return that runs on past a truncation into the
    next episode (2_rtg_pg ends it at every done)."""
    rollout = _rollout([[1, 3, 1]], [[0, 0, 1]], truncated=[[1, 0, 0]])
    zeros = jnp.zeros_like(rollout.reward)
    returns, _, weights = compute_returns(rollout, zeros, zeros, GAMMA, 0.97, False)
    np.testing.assert_allclose(returns[:, 0, 0], [1.0, 3.5, 1.0])
    np.testing.assert_array_equal(weights[:, 0, 0], [1, 1, 1])


@pytest.mark.parametrize(
    ("terminated", "truncated", "first_return"),
    [
        # Cut by the rollout's end: G_1 = 1 + 0.5 V(s_2) = 3, G_0 = 1 + 1.5.
        ([0, 0], [0, 0], 2.5),
        # A time limit at t=0 bootstraps there: G_0 = 1 + 0.5 V(s_1) = 2.
        ([0, 0], [1, 0], 2.0),
        # A termination at t=0 does not: G_0 = 1.
        ([1, 0], [0, 0], 1.0),
    ],
    ids=["rollout-cut", "time-limit", "terminated"],
)
def test_the_critics_target_bootstraps_only_where_an_episode_is_cut(
    terminated: list, truncated: list, first_return: float
) -> None:
    """Catches bootstrapping a terminal state, or dropping V at a cut
    (vpg's finish_path last_val)."""
    rollout = _rollout([[1, 1]], [terminated], truncated=[truncated])
    values = jnp.array([0.0, 2.0])[:, None, None]  # V(s_0), V(s_1)
    next_values = jnp.array([2.0, 4.0])[:, None, None]  # V(s_1), V(s_2)
    returns, _, weights = compute_returns(
        rollout, values, next_values, GAMMA, 0.97, baseline=True
    )
    assert returns[0, 0, 0] == pytest.approx(first_return)
    np.testing.assert_array_equal(weights, 1.0)  # a baseline trains every step


@pytest.mark.parametrize(
    ("gae_lambda", "advantages"),
    # V = (1, 1), V(s_2) = 2, r = (1, 1), no end. delta = r + 0.5 V' - V:
    # (0.5, 1). lambda 0: the deltas; lambda 1: G - V = (2, 2) - 1.
    [(0.0, [0.5, 1.0]), (1.0, [1.0, 1.0])],
)
def test_advantages_are_gae_of_the_baseline(
    gae_lambda: float, advantages: list
) -> None:
    """Catches advantages that ignore lambda or forget to subtract V."""
    rollout = _rollout([[1, 1]], [[0, 0]])
    values = jnp.ones_like(rollout.reward)
    next_values = jnp.array([1.0, 2.0])[:, None, None]
    _, got, _ = compute_returns(rollout, values, next_values, GAMMA, gae_lambda, True)
    np.testing.assert_allclose(got[:, 0, 0], advantages, rtol=1e-6)


def test_normalisation_reads_only_the_steps_that_weigh() -> None:
    """Catches a cut episode's unused returns shifting the others' scale."""
    got = normalise(jnp.array([1.0, 3.0, 100.0]), jnp.array([1.0, 1.0, 0.0]))
    # Over (1, 3): mean 2, population std 1.
    np.testing.assert_allclose(got[:2], [-1.0, 1.0], rtol=1e-6)


def _agent(**overrides: Any) -> REINFORCE:
    settings = {
        "n_envs": 2,
        "n_steps": 4,
        "actor_architecture": ("8", "tanh"),
        "critic_architecture": ("8", "tanh"),
        "normalize_advantage": False,
        "gamma": GAMMA,
        **overrides,
    }
    return REINFORCE("CartPole-v1", **settings)


def _update(agent: REINFORCE, rollout: Transition) -> tuple[Any, Any]:
    _, context = agent.make_context(1_000)
    state = agent.init(jax.random.PRNGKey(0), context)
    return state, agent.update(state, rollout, state, context)[0]


def _mean_log_probability_of_action_0(state: Any, obs: jax.Array) -> float:
    actor = state.actor_state
    return float(get_pi(actor, actor.params, obs)[0].log_prob(0).mean())


@pytest.mark.parametrize("reward", [1.0, -1.0])
def test_one_step_makes_a_rewarded_action_likelier(reward: float) -> None:
    """Catches a sign error in the loss (gradient descent on J), or a
    loss that does not depend on the actions' log-probabilities.

    Every step took action 0 and every return has the reward's sign, so the
    step's objective is that sign times the batch's mean log pi(0 | s): to
    first order in the small learning rate, one Adam step raises it."""
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, 2, 4))
    rollout = _rollout([[reward] * 4] * 2, [[0, 0, 0, 1]] * 2, obs=obs)
    before, after = _update(_agent(baseline=False), rollout)
    change = _mean_log_probability_of_action_0(
        after, obs
    ) - _mean_log_probability_of_action_0(before, obs)
    assert np.sign(change) == np.sign(reward), change


def test_a_cut_episodes_steps_do_not_move_the_policy() -> None:
    """Catches the mask missing from the loss: without a baseline, the
    actions of the episode the rollout cuts must not change the step."""
    obs = jax.random.normal(jax.random.PRNGKey(2), (4, 2, 4))
    ends = [[0, 1, 0, 0], [0, 0, 0, 1]]  # env 0's steps 2-3 are cut
    rewards = [[1, 1, 1, 1]] * 2
    actions = jnp.zeros((4, 2), jnp.int32)
    agent = _agent(baseline=False)
    _, one = _update(agent, _rollout(rewards, ends, obs=obs, action=actions))
    flipped = actions.at[2:, 0].set(1)
    _, other = _update(agent, _rollout(rewards, ends, obs=obs, action=flipped))
    jax.tree.map(
        np.testing.assert_array_equal, one.actor_state.params, other.actor_state.params
    )


@pytest.mark.parametrize("baseline", [True, False])
def test_the_critic_trains_only_with_a_baseline(baseline: bool) -> None:
    """Catches value steps run on a critic nothing reads, or skipped when
    the baseline needs them."""
    rollout = _rollout([[1, 1, 1, 1]] * 2, [[0, 0, 0, 1]] * 2)
    before, after = _update(_agent(baseline=baseline, value_steps=3), rollout)
    leaves = jax.tree.leaves(
        jax.tree.map(
            lambda a, b: bool(np.any(a != b)),
            before.critic_state.params,
            after.critic_state.params,
        )
    )
    assert any(leaves) == baseline


@pytest.mark.parametrize("option", [{"n_epochs": 2}, {"num_minibatches": 2}])
def test_more_than_one_policy_step_per_rollout_is_refused(option: dict) -> None:
    """Catches REINFORCE silently reusing a rollout, which its gradient
    estimate does not allow (that is PPO): neither is a keyword."""
    with pytest.raises(TypeError):
        _agent(**option)
