"""ValueCentering (ajax.extensions.centering): reward centering for any discounted agent with a
one-step target (Naik, Wan, Tomar & Sutton, RLC 2024).

Contract (the extension's docstring, from the papers' rules in ajax.agents.centering):
- on_target returns the target minus the offset the rule gives it, and the offset after the update:
  the batch mean of the next values followed at rate (Hisaki & Ono, Eq. 18), the stored offset
  moved by step_size times the mean TD error of the centred target (Naik et al., Eq. 10), or gain
  times the target critics' value at the reference row (zeros where None).
- It refuses the agents that centre themselves (ASAC, APO: not discounted) and those
  whose targets are lambda-returns (PPO, PQN), and runs on SAC, TD3 and DQN.
- It comes last of the extensions shaping the target.
- In a run, the offset moves from 0 to the reward rate's sign on a task of positive rewards.
Expected values are worked by hand below.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import APO, ASAC, DQN, PPO, PQN, SAC, TD3
from ajax.agents.centering import ReferencePoint, RVIBatchMean, TDErrorCentering
from ajax.extensions.base import ExtensionStack
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.centering import ValueCentering
from tests.agents.extension_stubs import DrawEveryPhase

NET = ("16", "relu", "16", "relu")


def fold(rule, batch: dict, target, offset: float):
    stack = ExtensionStack((ValueCentering(rule),))
    state = SimpleNamespace(ext_state=(jnp.asarray(offset),))
    return stack.apply_on_target(state, batch, target, 0, jax.random.PRNGKey(0), 10)


def test_the_batch_mean_rule_centres_on_the_offset_then_follows_the_next_values():
    # centred: [2 - 0.5, 4 - 0.5]; offset: 0.9 * 0.5 + 0.1 * mean(1, 3) = 0.65
    batch = {"next_values": jnp.array([[1.0], [3.0]])}
    target, metrics, (offset,) = fold(
        RVIBatchMean(rate=0.1), batch, jnp.array([[2.0], [4.0]]), 0.5
    )
    np.testing.assert_allclose(target, [[1.5], [3.5]])
    np.testing.assert_allclose(offset, 0.65, rtol=1e-6)
    assert float(metrics["centering/offset"]) == 0.5


def test_the_td_error_rule_moves_by_the_mean_error_of_the_centred_target():
    # centred: [1.5, 3.5]; predictions 1: errors [0.5, 2.5], mean 1.5; offset 0.5 + 0.01 * 1.5
    batch = {"next_values": jnp.zeros((2, 1)), "predictions": lambda: jnp.ones((2, 1))}
    target, _, (offset,) = fold(
        TDErrorCentering(0.01), batch, jnp.array([[2.0], [4.0]]), 0.5
    )
    np.testing.assert_allclose(target, [[1.5], [3.5]])
    np.testing.assert_allclose(offset, 0.515, rtol=1e-6)


def test_the_reference_rule_reads_the_target_critics_at_its_row_with_zeros_for_none():
    seen = []

    def target_value_at(observation, action):
        seen.append((np.asarray(observation), np.asarray(action)))
        return jnp.array([[4.0]])

    batch = {
        "next_values": jnp.zeros((2, 1)),
        "observations": jnp.ones((2, 3)),
        "actions": jnp.ones((2, 2)),
        "target_value_at": target_value_at,
    }
    rule = ReferencePoint(observation=(1.0, 2.0, 3.0), gain=0.5)
    target, _, (offset,) = fold(rule, batch, jnp.array([[2.0], [4.0]]), 9.0)
    np.testing.assert_allclose(target, [[0.0], [2.0]])  # minus 0.5 * 4
    np.testing.assert_allclose(offset, 2.0)
    np.testing.assert_array_equal(seen[0][0], [[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(seen[0][1], [[0.0, 0.0]])


@pytest.mark.parametrize(
    "build",
    [
        lambda e: ASAC("Pendulum-v1", extensions=e),
        lambda e: APO("Pendulum-v1", extensions=e),
        lambda e: PPO("Pendulum-v1", extensions=e),
        lambda e: PQN("CartPole-v1", extensions=e),
    ],
    ids=["ASAC", "APO", "PPO", "PQN"],
)
def test_it_refuses_agents_that_centre_themselves_or_return_lambda_returns(build):
    with pytest.raises(MissingCapability):
        build((ValueCentering(),))


def test_it_runs_on_discounted_one_step_agents():
    for agent in (
        SAC("Pendulum-v1", extensions=(ValueCentering(),)),
        TD3("Pendulum-v1", extensions=(ValueCentering(),)),
        DQN("CartPole-v1", extensions=(ValueCentering(TDErrorCentering(0.01)),)),
    ):
        agent.make_context(16)


def test_it_comes_last_of_the_extensions_shaping_the_target():
    with pytest.raises(ValueError, match="comes last"):
        SAC(
            "Pendulum-v1", extensions=(ValueCentering(), DrawEveryPhase())
        ).make_context(16)
    SAC("Pendulum-v1", extensions=(DrawEveryPhase(), ValueCentering())).make_context(16)


@pytest.mark.parametrize("rule", [RVIBatchMean(rate=0.05), TDErrorCentering(0.05)])
def test_on_rewards_of_one_the_offset_rises_towards_them(rule):
    """CartPole pays 1 a step: its reward rate is 1, so from 0 the offset rises."""
    agent = DQN(
        "CartPole-v1",
        architecture=NET,
        learning_starts=32,
        buffer_size=512,
        batch_size=32,
        extensions=(ValueCentering(rule),),
    )
    state, _ = agent.train(seed=0, n_timesteps=128)  # 96 updates
    offset = float(np.asarray(state.ext_state[0]).reshape(-1)[0])
    assert 0.0 < offset
