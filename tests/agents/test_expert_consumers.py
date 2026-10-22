"""What every agent taking an ``expert_policy`` (found from the signatures)
gives its expert and does with its answers, on the identity task
(tests/environments/identity_env.py) with the observations normalised
(and SAC's training fraction appended), its expert stateful and callable
only as ``expert(state, obs)``:

* every call gets the raw observation [t, c] (t an integer step, c in
  [-1, 1], no extra column); collection and the cloning collector also
  give it its state, its t calls since the episode's start (the
  evaluation's expert rollout does not reset it after the first episode,
  the only one whose return counts);
* the cloning dataset holds its actions in agent units;
* the evaluation's expert return is its closed form, TERMINAL.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajax
import ajax.extensions.pretrain
from ajax.environments.utils import env_action_to_agent
from ajax.extensions.pretrain import CloningPretrain
from ajax.logging.wandb_logging import LoggingConfig
from tests.agents.test_agent_contract import CASES
from tests.environments.identity_env import LIMIT, TERMINAL, IdentityEnv

N_ENVS, N_EVAL = 2, 3  # the calls' batch sizes tell training's from evaluation's
#: Every agent Ajax ships (test_exports: ajax.__all__ names them all) that
#: takes an expert; one without a row in CASES fails.
EXPERT_AGENTS = sorted(
    n
    for n in ajax.__all__
    if "expert_policy" in inspect.signature(getattr(ajax, n)).parameters
)
#: The agents binding CloningPretrain's context (their train_*.py).
CLONING_AGENTS = {"APO", "ASAC", "AVG", "CrossQ", "DAgger", "REDQ", "SAC", "TD3"}


def expert_action(t):
    """1 (env units) before step TERMINAL, then 0: every episode returns TERMINAL."""
    return (t < TERMINAL).astype(np.float32)


class StrictExpert:
    """Callable only as ``expert(state, obs)``; its state counts its calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array):
        keep = lambda o, s: self.calls.append((np.asarray(o), np.asarray(s)))  # noqa: E731
        jax.debug.callback(keep, obs, state)
        return expert_action(obs[..., :1]), state + 1


@pytest.mark.parametrize("name", EXPERT_AGENTS)
def test_an_agent_gives_its_expert_raw_observations_and_its_state(name, monkeypatch):
    assert name in CASES, f"{name} takes an expert: give it a row in CASES"
    case, expert, env = CASES[name], StrictExpert(), IdentityEnv()
    takes = inspect.signature(case.cls).parameters
    clones = name in CLONING_AGENTS
    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    pre_train = ajax.extensions.pretrain.pre_train

    def recording(*, dataset, **kwargs):
        keep = lambda o, a: datasets.append((np.asarray(o), np.asarray(a)))  # noqa: E731
        jax.debug.callback(keep, dataset.raw_obs, dataset.action)
        return pre_train(dataset=dataset, **kwargs)

    monkeypatch.setattr(ajax.extensions.pretrain, "pre_train", recording)
    options = {
        k: True for k in ("normalize_observations", "use_train_frac") if k in takes
    }
    cloning = (CloningPretrain(expert, n_steps=8, epochs=1, batch_size=8),)
    agent = case.cls(
        env_id=env,
        env_params=env.default_params,
        n_envs=N_ENVS,
        extensions=cloning if clones else (),
        **options,
        **(case.kwargs | {"expert_policy": expert}),
    )
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    _, out = agent.train(
        seed=0,
        n_timesteps=case.n_timesteps,
        num_episode_test=N_EVAL,
        logging_config=logging,
    )
    jax.effects_barrier()
    assert {len(obs) for obs, _ in expert.calls} == (
        {N_ENVS, N_EVAL} if clones else {N_EVAL}
    )
    for obs, state in expert.calls:
        t, c = obs[..., 0], obs[..., 1]
        assert obs.shape[-1] == 2 and (np.abs(c) <= 1).all(), obs
        assert (t == np.round(t)).all() and (0 <= t).all() and (t <= LIMIT).all(), obs
        if len(obs) == N_ENVS:
            np.testing.assert_array_equal(state[..., 0], t)
    assert len(datasets) == clones
    for raw_obs, actions in datasets:
        env_units = expert_action(raw_obs[..., :1])
        expected = env_action_to_agent(env_units, env, env.default_params)
        np.testing.assert_allclose(actions, expected, rtol=1e-6)
    evaluated = np.asarray(out["timestep"]) >= 0
    assert evaluated.any()
    returns = np.asarray(out["Eval/episodic mean expert reward"])[evaluated]
    np.testing.assert_allclose(returns, TERMINAL, rtol=1e-6)
