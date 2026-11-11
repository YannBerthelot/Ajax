"""MEW (Adamczyk, Rojas and Kulkarni, 2026, arXiv 2603.12324), against the
paper and the authors' code (JacobHA/MEW@8f180eb, annealed_asac_temporal.py).

Contract:
  * friction (eq. 4, eq. 8): beta times the sum over every lag k of each
    env's centred-reward autocovariance, lag k averaged over its n - k
    pairs, averaged over the envs, floored at 1e-3 (the code's MIN_ZETA);
  * one step (Algorithm 1, the code's form): beta = 1 / (alpha + 1e-8),
    beta <- beta + speed / sqrt(friction), alpha <- 1 / beta;
  * MEW steps after more than ``history`` steps per env and sets the
    temperature in place of the agent's own step, on every agent with a
    maximum-entropy objective; any other agent refuses it.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.DQN.DQN import DQN
from ajax.agents.IQL.IQL import IQL
from ajax.agents.PPO.PPO import PPO
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.extensions.annealing import MEW, anneal, integrated_autocovariance
from ajax.extensions.capabilities import MissingCapability
from tests.agents.test_agent_contract import CASES, OFF_POLICY

MAX_ENTROPY_AGENTS = sorted(name for name, case in CASES.items() if case.alpha)
#: One critic step per update, so 120 steps stay under the fast tier's 128.
FEWER_CRITIC_STEPS = {"REDQ": {"num_critic_updates": 1}}


def test_autocovariance_averages_each_lag_over_its_pairs():
    """Catches dividing every lag by n (the code's comment) or not centring.
    [3, 0, 0] centres to [2, -1, -1]: lag 0 (4 + 1 + 1) / 3 = 2, lag 1
    (-2 + 1) / 2 = -0.5, lag 2 -2 / 1 = -2; sum -0.5 (dividing by n: 1.0)."""
    total = integrated_autocovariance(jnp.array([[3.0, 0.0, 0.0]]))
    np.testing.assert_allclose(total, -0.5, rtol=1e-5)


def test_autocovariance_matches_its_definition_summed_directly():
    """Catches a circular (unpadded) FFT, whose lags wrap around, and a
    wrong mean over the envs. The reference sums eq. 8's products directly;
    float32 FFT over 64 steps: rtol 1e-4."""
    rewards = np.random.default_rng(0).normal(size=(2, 64))
    centred = rewards - rewards.mean(axis=1, keepdims=True)
    n = rewards.shape[1]
    per_env = [
        sum(np.dot(row[: n - k], row[k:]) / (n - k) for k in range(n))
        for row in centred
    ]
    total = integrated_autocovariance(jnp.asarray(rewards, jnp.float32))
    np.testing.assert_allclose(total, np.mean(per_env), rtol=1e-4, atol=1e-5)


def test_a_step_moves_beta_by_speed_over_the_root_of_the_friction():
    """Catches Algorithm 1's beta sqrt(zeta) denominator, or zeta without
    beta, in place of the code's speed / sqrt(beta sum C). [1, 0, 1] centres
    to [1/3, -2/3, 1/3]: sum C = 2/9 - 2/9 + 1/9 = 1/9; alpha 0.5, beta
    1 / (0.5 + 1e-8), speed 0.1: alpha = 1 / (beta + 0.1 / sqrt(beta / 9))
    = 0.45205259."""
    alpha, zeta = anneal(jnp.asarray(0.5), jnp.array([[1.0, 0.0, 1.0]]), 0.1)
    np.testing.assert_allclose(alpha, 0.45205259, rtol=1e-5)
    np.testing.assert_allclose(zeta, 2.0 / 9.0, rtol=1e-5)


def test_constant_rewards_step_at_the_floor_of_the_friction():
    """Catches a missing floor (a division by zero on rewards that do not
    vary): friction 1e-3, alpha = 1 / (2 + 1e-3 / sqrt(1e-3)) = 0.49221737."""
    alpha, zeta = anneal(jnp.asarray(0.5), jnp.full((1, 8), 3.0), 1e-3)
    np.testing.assert_allclose(alpha, 0.49221737, rtol=1e-5)
    np.testing.assert_allclose(zeta, 1e-3, rtol=1e-6)


@pytest.mark.parametrize("name", MAX_ENTROPY_AGENTS)
def test_it_anneals_the_temperature_of_every_max_entropy_agent(name):
    """Catches an agent MEW does not reach (no capability, no step written,
    AVG's missing history). 120 steps past a 101-reward history: MEW steps
    from the 102nd, so the temperature ends below where it started."""
    case = CASES[name]
    extensions = (MEW(speed=1.0, history=101),)
    kwargs = {**case.kwargs, **FEWER_CRITIC_STEPS.get(name, {})}
    agent = case.cls("Pendulum-v1", **kwargs, extensions=extensions)
    state, _ = agent.train(seed=0, n_timesteps=120)
    alpha = np.exp(np.asarray(state.alpha.params["log_alpha"]))
    assert np.all((alpha > 0.0) & (alpha < agent.alpha_args.alpha_init))
    assert np.all(np.isfinite(np.asarray(state.ext_state[0].friction)))


def test_before_its_history_fills_the_temperature_stays_where_it_started():
    """Catches the agent's own temperature step running beside MEW, and MEW
    stepping before its history fills: a target entropy of +10 per dimension
    would raise SAC's temperature at every step."""
    agent = SAC(
        "Pendulum-v1",
        **OFF_POLICY,
        alpha_init=0.3,
        alpha_update_start=0,
        target_entropy_per_dim=10.0,
        extensions=(MEW(history=1_000),),
    )
    state, _ = agent.train(seed=0, n_timesteps=64)
    np.testing.assert_allclose(jnp.exp(state.alpha.params["log_alpha"]), 0.3, rtol=1e-6)


@pytest.mark.parametrize("cls", [TD3, IQL, PPO], ids=["TD3", "IQL", "PPO"])
def test_an_agent_without_a_temperature_refuses_it(cls):
    """Catches MEW folding into an agent with no maximum-entropy objective,
    where it would write a temperature nothing reads (PPO's entropy bonus is
    not one)."""
    with pytest.raises(MissingCapability, match="entropy_temperature"):
        cls("Pendulum-v1", extensions=(MEW(),)).make_context(16)
    with pytest.raises(MissingCapability, match="entropy_temperature"):
        DQN("CartPole-v1", extensions=(MEW(),)).make_context(16)


def test_it_refuses_a_history_the_authors_update_never_reads():
    """Catches a silent no-op: the authors' code updates only on more than
    100 rewards."""
    with pytest.raises(ValueError, match="above 100"):
        MEW(history=100)
