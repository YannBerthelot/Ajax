"""The transitions every agent folding ``on_target`` hands its extensions,
on two envs of the identity task (tests/environments/identity_env.py),
whose observation [t, c] says which step of which episode a row is:

(i) a row that did not terminate continues its own episode: its s' is
[t + 1, c] for its s = [t, c], a time-limited row's the final observation
(t = LIMIT), not the next episode's first, nor another env's;
(ii) the flags are those the step raised: ``dones`` (where the target cuts
the bootstrap) the termination, ``terminated`` and ``truncated`` apart;
(iii) under reward normalisation the rewards are the normalised channel the
env produced, not the raw one it keeps in its info (on the rows that did
not terminate: ASAC penalises a termination's reward).
"""

import inspect
from dataclasses import dataclass

import jax
import numpy as np
import pytest

import ajax
from ajax.extensions.base import Extension
from ajax.wrappers import _OnlineNormalization
from tests.agents.test_agent_contract import CASES, DISCRETE
from tests.environments.identity_env import IdentityEnv, ends

FLAGS = ("dones", "terminated", "truncated")
KEYS = ("observations", "next_observations", "rewards", *FLAGS, "reward_scale")
#: Every agent Ajax ships (test_exports: ajax.__all__ names them all) that
#: folds on_target; one without a row in CASES fails.
FOLDING = sorted(
    n
    for n in ajax.__all__
    if "on_target" in getattr(getattr(ajax, n), "supported_extension_phases", ())
)
BATCHES: list[dict] = []


def _keep(names: tuple, *values: np.ndarray) -> None:
    BATCHES.append({k: np.asarray(v) for k, v in zip(names, values, strict=True)})


@dataclass(frozen=True)
class _RecordTargets(Extension):
    """Records the transition keys of every ``on_target`` batch."""

    name: str = "record-targets"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        names = tuple(k for k in KEYS if batch.get(k) is not None)
        jax.debug.callback(lambda *v: _keep(names, *v), *(batch[k] for k in names))
        return target


def _close_to_any(x: np.ndarray, pool: np.ndarray) -> np.ndarray:
    return np.isclose(x[:, None], pool[None, :], rtol=1e-5, atol=1e-6).any(axis=1)


def _column(key: str) -> np.ndarray:
    return np.concatenate([b[key].reshape(-1) for b in BATCHES])


@pytest.mark.parametrize("name", FOLDING)
def test_on_target_rows_are_their_episodes_transitions(name, monkeypatch):
    assert name in CASES, f"{name} folds on_target: give it a row in CASES"
    case = CASES[name]
    normalising = "normalize_rewards" in inspect.signature(case.cls).parameters
    produced: list[tuple] = []  # the env's (raw, normalised) rewards
    normalize = _OnlineNormalization._normalize_reward

    def recording(self, reward, done, stats):
        normed, stats = normalize(self, reward, done, stats)
        jax.debug.callback(lambda *v: produced.append(v), reward, normed)
        return normed, stats

    monkeypatch.setattr(_OnlineNormalization, "_normalize_reward", recording)
    BATCHES.clear()
    env = IdentityEnv(discrete=case.envs == DISCRETE, keep_raw=True)
    agent = case.cls(
        env_id=env,
        env_params=env.default_params,
        n_envs=2,
        extensions=[_RecordTargets()],
        **({"normalize_rewards": True} if normalising else {}),
        **case.kwargs,
    )
    agent.train(seed=0, n_timesteps=case.n_timesteps)
    jax.effects_barrier()
    obs = np.concatenate([b["observations"].reshape(-1, 2) for b in BATCHES])
    nxt = np.concatenate([b["next_observations"].reshape(-1, 2) for b in BATCHES])
    terminated, truncated = ends(obs)
    assert terminated.any() and truncated.any(), "no episode end was learned from"
    # (i)
    np.testing.assert_array_equal(nxt[~terminated], obs[~terminated] + [1.0, 0.0])
    # (ii)
    flags = [k for k in FLAGS if k in BATCHES[0]]
    assert set(flags) - {"truncated"}, f"no termination flag in {sorted(BATCHES[0])}"
    for key in flags:
        expected = truncated if key == "truncated" else terminated
        np.testing.assert_array_equal(_column(key) > 0, expected, err_msg=key)
    if not normalising:
        return
    # (iii)
    raw, normed = (np.concatenate([np.ravel(p[i]) for p in produced]) for i in (0, 1))
    assert np.abs(normed - raw).max() > 0.1, "the normalisation was the identity"
    scale = np.ravel(BATCHES[0].get("reward_scale", 1.0))[0]
    rewards = _column("rewards")[~terminated] / scale
    assert _close_to_any(rewards, normed).all()
    assert not _close_to_any(rewards, raw).all()
