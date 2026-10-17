"""PriorData: a transition dataset in a replay agent's replay and
normalisers, then the agent's own updates on it, before it acts; an agent
without a replay refuses it.

The rows are the identity task's (tests/environments/identity_env.py),
recorded from an expert acting ``c`` (its sign, on Discrete(2)).
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.buffers import dataset as ds
from ajax.environments.interaction import reward_statistics
from ajax.environments.utils import wrapper_chain
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.pretrain import PriorData
from tests.agents.test_agent_contract import CASES
from tests.agents.test_replay_normalisation import REPLAYING
from tests.environments.identity_env import IdentityEnv


@pytest.fixture(scope="module")
def data() -> dict[bool, ds.TransitionDataset]:
    """Whole episodes of 24 steps, by whether the actions are discrete."""
    out = {}
    sign = lambda o: (o[1:] > 0).astype(jnp.float32)  # noqa: E731
    for discrete, expert in ((False, lambda o: o[1:]), (True, sign)):
        agent = CASES["DQN" if discrete else "TD3"].cls(IdentityEnv(discrete), n_envs=1)
        out[discrete] = ds.record(expert, agent.env_args, jax.random.PRNGKey(0), 24)
    return out


def train(name: str, extensions: tuple = (), **kw: Any) -> Any:
    """``name`` on the identity task, trained 0 steps: one seed's state."""
    case, env = CASES[name], IdentityEnv(discrete=name == "DQN")
    agent = case.cls(env, n_envs=1, extensions=extensions, **case.kwargs | kw)
    return jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])


def params(state: Any) -> list:
    nets = (state.actor_state, getattr(state, "critic_state", None))
    return jax.tree.leaves([n.params for n in nets if n is not None])


@pytest.mark.parametrize("name", REPLAYING)
def test_every_replay_agent_learns_from_the_rows_before_acting(name, data) -> None:
    rows = data[name == "DQN"]
    untouched = params(train(name))
    for updates in (0, 3):
        state = train(name, (PriorData(rows, updates),))
        replay, n = state.collector_state.buffer_state, len(rows.reward)
        assert int(replay.current_index) == n
        np.testing.assert_array_equal(replay.experience["obs"][0, :n], rows.obs)
        np.testing.assert_array_equal(replay.experience["action"][0, :n], rows.action)
        # Its own update ran offline_updates times: no network moves at 0.
        moved = [
            not np.array_equal(a, b)
            for a, b in zip(params(state), untouched, strict=False)
        ]
        assert any(moved) == (updates > 0), moved


def test_the_rows_join_the_normalisers_once_each(data) -> None:
    """The observation statistics hold the rows' moments, mirrored on the
    networks; the reward statistics are the env normaliser's after stepping
    through the rows, its running discounted return left at zero."""
    rows = data[False]
    norm: dict[str, Any] = {"normalize_observations": True}
    norm |= {"normalize_rewards": True, "reward_normalization_gamma": 0.9}
    state = train("TD3", (PriorData(rows),), **norm)
    obs = state.collector_state.obs_norm_info
    assert float(obs.count.sum()) == len(rows.reward)
    np.testing.assert_allclose(obs.mean[0], rows.obs.mean(0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(obs.var[0], rows.obs.var(0), rtol=1e-4, atol=1e-6)
    for net in (state.actor_state, state.critic_state):
        for a, b in zip(
            jax.tree.leaves(net.obs_norm_info), jax.tree.leaves(obs), strict=False
        ):
            np.testing.assert_array_equal(a, b)

    env = CASES["TD3"].cls(IdentityEnv(), n_envs=1, **norm).env_args.env
    layer = next(w for w in wrapper_chain(env) if hasattr(w, "_normalize_reward"))
    stepped = layer._initial_stats().reward
    for r, end in zip(rows.reward, rows.terminated | rows.truncated, strict=False):
        _, stepped = layer._normalize_reward(jnp.float32(r), jnp.bool_(end), stepped)
    got = reward_statistics(state.collector_state.env_state)
    assert got is not None
    for field in ("count", "mean", "mean_2", "var"):
        want = np.ravel(getattr(stepped, field))
        np.testing.assert_allclose(np.ravel(getattr(got, field)), want, rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(got.returns), 0.0)


@pytest.mark.parametrize("name", ["PPO", "AVG"])
def test_an_agent_without_a_replay_refuses_it(name, data) -> None:
    with pytest.raises(MissingCapability, match=r"replay \(its updates sample"):
        train(name, (PriorData(data[False]),))
