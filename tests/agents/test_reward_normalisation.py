"""Reward normalisation's exact contract, for every agent that accepts
``normalize_rewards`` (found from the signatures: a new one joins):

(i) the normaliser divides by the running std of the single-step reward by
default and of the discounted return with ``reward_normalization_gamma``, as
``ActorCritic.__init__`` documents it; the agent's own discount never stands
in for it;
(ii) the Train and Eval returns are in the env's units: on a task paying 1 a
step for ``L`` steps both are ``L``, not ``L / sqrt(1e-8)``, the normalised
return of a constant reward.

Dividing, never centring, is the normaliser's own contract
(tests/environments/test_wrappers.py::
test_reward_normaliser_divides_by_the_chosen_statistics), and restoring its
statistics on resume is the resume contract's
(tests/agents/test_resume_contract.py runs every agent here with
``normalize_rewards`` and ``reward_normalization_gamma``).
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import ajax
from ajax.environments.create import strip_ajax_wrappers
from ajax.logging.wandb_logging import LoggingConfig
from tests.agents.test_agent_contract import CASES, DISCRETE
from tests.probing import envs

#: Every agent Ajax ships that takes normalize_rewards (test_exports:
#: ajax.__all__ names them all).
NORMALISING = sorted(
    n
    for n in ajax.__all__
    if "normalize_rewards" in inspect.signature(getattr(ajax, n)).parameters
)
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
    assert set(NORMALISING) <= set(CASES), set(NORMALISING) - set(CASES)


@pytest.mark.parametrize("name", NORMALISING)
def test_the_normaliser_divides_by_the_documented_statistics(name: str) -> None:
    case = CASES[name]
    cls, env_id = case.cls, case.envs[0]
    default = cls(env_id=env_id, normalize_rewards=True)
    layers = strip_ajax_wrappers(default.env_args.env)[1]
    assert layers["normalize_reward"] and layers["gamma"] is None
    given = cls(env_id=env_id, normalize_rewards=True, reward_normalization_gamma=0.9)
    assert strip_ajax_wrappers(given.env_args.env)[1]["gamma"] == 0.9
    off = strip_ajax_wrappers(cls(env_id=env_id).env_args.env)[1]
    assert not off.get("normalize_reward", False)
    with pytest.raises(ValueError, match="normalize_rewards=True"):
        cls(env_id=env_id, reward_normalization_gamma=0.9)


#: Rows that fail on a known bug, each failing until it is fixed (strict).
KNOWN_BUGS = {
    "UDRL": "UDRL's collection sums the normalised reward into its Train return"
    " (train_UDRL._udrl_collect_step), not the env's own (RAW_REWARD_KEY), as"
    " collect_experience does",
}


ROWS = [
    pytest.param(
        n,
        marks=pytest.mark.xfail(
            reason=KNOWN_BUGS[n], raises=AssertionError, strict=True
        ),
    )
    if n in KNOWN_BUGS
    else n
    for n in NORMALISING
]


@pytest.mark.parametrize("name", ROWS)
def test_the_returns_are_reported_in_the_envs_units(name: str) -> None:
    case = CASES[name]
    env, params = _counter(2 if case.envs == DISCRETE else 0).make()
    agent = case.cls(
        env_id=env, env_params=params, n_envs=2, normalize_rewards=True, **case.kwargs
    )
    budget = case.n_timesteps
    logging = LoggingConfig(config={}, log_frequency=budget, use_wandb=False)
    _, out = agent.train(
        seed=0, n_timesteps=budget, num_episode_test=2, logging_config=logging
    )
    evaluated = np.asarray(out["timestep"]) >= 0
    assert evaluated.any(), "no evaluation"
    keys = ("Train/episodic mean reward", "Eval/episodic mean reward")
    got = {k: np.asarray(out[k])[evaluated] for k in keys}
    wrong = {k: v for k, v in got.items() if not np.allclose(v, L, rtol=1e-5, atol=0)}
    assert not wrong, wrong
