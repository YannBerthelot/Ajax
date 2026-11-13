"""Reward normalisation's exact contract, for every agent that accepts
``normalize_rewards`` (found from the signatures: a new one joins):

(i) the normaliser divides by the running std of the discounted return, as
SB3's ``VecNormalize`` does and ``ActorCritic.__init__`` documents: with the
agent's own discount by default (0.99 for an agent without one, VecNormalize's
default), with ``reward_normalization_gamma`` when given (0: the single-step
reward's std);
(ii) the Train and Eval returns are in the env's units: on a task paying 1 a
step for ``L`` steps both are ``L``, not the normalised rewards' sum;
(iii) a constant reward is not divided by its spread of 0: SB3's clip to
+-10 bounds the step where every env's return is the same.

Dividing, never centring, is the normaliser's own contract
(tests/environments/test_wrappers.py::
test_reward_normaliser_divides_by_the_chosen_statistics), and restoring its
statistics on resume is the resume contract's
(tests/agents/test_resume_contract.py runs every agent here with
``normalize_rewards`` and ``reward_normalization_gamma``).
"""

import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajax
from ajax.buffers.utils import normalized_rewards
from ajax.environments.create import strip_ajax_wrappers
from ajax.environments.interaction import reset, step
from ajax.logging.wandb_logging import LoggingConfig
from ajax.policy import normalised_reward
from ajax.types import EnvNormalizationInfo, NormalizationInfo
from tests.agents.test_agent_contract import CASES, DISCRETE
from tests.probing import envs

#: Every agent Ajax ships that takes normalize_rewards (test_exports:
#: ajax.__all__ names them all).
NORMALISING = sorted(
    n
    for n in ajax.__all__
    if "normalize_rewards" in inspect.signature(getattr(ajax, n)).parameters
)
#: The agents with a discount of their own, which their reward normaliser takes.
DISCOUNTED = [
    n for n in NORMALISING if "gamma" in inspect.signature(getattr(ajax, n)).parameters
]
L = 4


def _pays_one(st: envs.State, a: jnp.ndarray, key: object) -> tuple:
    return st.replace(s=st.s + 1), 1.0, st.s + 1 >= L


def _step_count(st: envs.State) -> jnp.ndarray:
    return envs.one_hot(jnp.minimum(st.s, L), L + 1)


def _counter(actions: int) -> envs.Spec:
    """1 a step, every episode ``L`` steps (the time limit, ``L + 1``, never
    fires); Discrete(``actions``), or Box(-1, 1) for 0."""
    return envs.Spec(
        "pays_one", _pays_one, _step_count, obs_dim=L + 1, actions=actions, limit=L + 1
    )


def test_every_normalising_agent_has_a_row() -> None:
    """A new agent taking normalize_rewards joins test_agent_contract.CASES."""
    assert set(NORMALISING) <= set(CASES), set(NORMALISING).difference(CASES)


def _normaliser_gamma(agent: object) -> object:
    layers = strip_ajax_wrappers(agent.env_args.env)[1]  # type: ignore[attr-defined]
    assert layers["normalize_reward"]
    return layers["gamma"]


@pytest.mark.parametrize("name", DISCOUNTED)
def test_the_normaliser_takes_the_agents_discount_by_default(name: str) -> None:
    """A normaliser on the single-step reward, or on a fixed discount, would
    read another return than the one the agent's value estimates."""
    discounted: Any = CASES[name].cls  # ActorCritic itself takes no gamma
    agent = discounted(env_id=CASES[name].envs[0], normalize_rewards=True, gamma=0.95)
    assert _normaliser_gamma(agent) == 0.95


@pytest.mark.parametrize("name", sorted(set(NORMALISING).difference(DISCOUNTED)))
def test_an_agent_without_a_discount_normalises_with_sb3s(name: str) -> None:
    """The average-reward agents (APO, ASAC, RVI-SAC) have no discount: the
    normaliser takes VecNormalize's default, 0.99, not the single-step std."""
    case = CASES[name]
    agent = case.cls(env_id=case.envs[0], normalize_rewards=True)
    assert _normaliser_gamma(agent) == 0.99


@pytest.mark.parametrize("name", NORMALISING)
def test_a_given_normalisation_discount_wins(name: str) -> None:
    """0 keeps the single-step reward's std, the default before 0.6."""
    case = CASES[name]
    cls, env_id = case.cls, case.envs[0]
    for gamma in (0.9, 0.0):
        given = cls(
            env_id=env_id, normalize_rewards=True, reward_normalization_gamma=gamma
        )
        assert _normaliser_gamma(given) == gamma
    off = strip_ajax_wrappers(cls(env_id=env_id).env_args.env)[1]
    assert not off.get("normalize_reward", False)
    with pytest.raises(ValueError, match="normalize_rewards=True"):
        cls(env_id=env_id, reward_normalization_gamma=0.9)


def test_a_constant_reward_is_not_divided_by_a_zero_spread() -> None:
    """CartPole pays 1 a step. Dividing by the single-step reward's std, 0,
    paid r / sqrt(1e-8) = 10 000 every step, and recurrent PPO's value
    targets near 1e6 collapsed it on masked CartPole. SB3's VecNormalize,
    worked by hand for 8 envs at gamma 0.9: at step 1 every env's return is
    1, a spread of 0, so 10 000 clips to 10; at step 2 the returns counted
    are eight 1s and eight 1 + 0.9 = 1.9s, std 0.45, so the reward is
    1 / 0.45."""
    from ajax import PPO

    agent = PPO("CartPole-v1", n_envs=8, gamma=0.9, normalize_rewards=True)
    env, params = agent.env_args.env, agent.env_args.env_params
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    _, state = reset(keys, env, "gymnax", params)
    action = jnp.zeros(8, dtype=jnp.int32)
    _, state, first, *_ = step(keys, state, action, env, "gymnax", params)
    _, state, second, *_ = step(keys, state, action, env, "gymnax", params)
    np.testing.assert_allclose(first, 10.0, rtol=1e-6)
    np.testing.assert_allclose(second, 1 / 0.45, rtol=1e-5)


@pytest.mark.parametrize("name", NORMALISING)
def test_the_returns_are_reported_in_the_envs_units(name: str) -> None:
    case = CASES[name]
    env, params = _counter(2 if case.envs == DISCRETE else 0).make()
    agent = case.cls(
        env_id=env, env_params=params, n_envs=2, normalize_rewards=True, **case.kwargs
    )
    budget = case.n_timesteps
    logging = LoggingConfig(log_frequency=budget, backends=())
    _, out = agent.train(
        seed=0, n_timesteps=budget, num_episode_test=2, logging_config=logging
    )
    evaluated = np.asarray(out["timestep"]) >= 0
    assert evaluated.any(), "no evaluation"
    keys = ("Train/episodic mean reward", "Eval/episodic mean reward")
    got = {k: np.asarray(out[k])[evaluated] for k in keys}
    wrong = {k: v for k, v in got.items() if not np.allclose(v, L, rtol=1e-5, atol=0)}
    assert not wrong, wrong


def _replayed(reward: jnp.ndarray, stats: NormalizationInfo) -> jnp.ndarray:
    return normalized_rewards(reward, stats)


def _evaluated(reward: jnp.ndarray, stats: NormalizationInfo) -> jnp.ndarray:
    return normalised_reward(reward, EnvNormalizationInfo(reward=stats, obs=None))


@pytest.mark.parametrize("divide", [_replayed, _evaluated])
def test_rewards_divided_after_collection_are_clipped_as_collected(divide) -> None:
    """A replay agent divides its raw rows at sampling, and evaluation the
    rewards it scores the critic and counts a command down by: unclipped, a
    constant reward collected as 10 would come back as 10 000. Statistics
    of eight returns of 1, a spread of 0."""
    one = jnp.ones((1, 1))
    stats = NormalizationInfo(var=0 * one, count=8 * one, mean=one, mean_2=0 * one)
    np.testing.assert_allclose(divide(jnp.ones(3), stats), 10.0)
