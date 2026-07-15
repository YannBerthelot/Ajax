"""Gap A (Phase 4a): on-policy agents expose ``last_rollout`` on
``BaseAgentState``.

When the opt-in config flag ``expose_recent_rollout`` is set to
``True`` on PPO / PQN / APO / AVG, the most recent ``(T, n_envs, ...)``
rollout transition produced by ``collect_experience`` is written to
``agent_state.last_rollout`` after each iteration. When the flag is
``False`` (the default), ``last_rollout`` stays ``None`` and the JIT
trace / pytree shape are identical to the legacy path.

These tests pin both sides of that contract for each on-policy agent
so Phase 4b extensions (BiasVoreDecomposition / ConditioningMetrics /
…) can rely on it without re-running the rollout per eval.
"""

from __future__ import annotations

import jax.numpy as jnp

from ajax.agents.APO.APO import APO
from ajax.agents.AVG.AVG import AVG
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN


def _unwrap(state):
    """Strip the ``(state, metrics)`` tuple that some ``agent.train``
    paths return so the tests can read ``state.last_rollout`` uniformly.
    """
    if isinstance(state, tuple):
        state = state[0]
    return state


# Tiny configs — mirror the ``_TINY_*`` style used in
# tests/extensions/test_ppo_dqn_pqn_extension_smoke.py so the JIT cost
# stays in the multi-second range.
_TINY_PPO = {
    "env_id": "CartPole-v1",
    "n_envs": 2,
    "actor_architecture": ("16", "relu"),
    "critic_architecture": ("16", "relu"),
    "n_steps": 32,
    "batch_size": 32,
    "n_epochs": 1,
}

_TINY_PQN = {
    "env_id": "CartPole-v1",
    "n_envs": 2,
    "architecture": ("16", "relu"),
    "n_steps": 8,
    "n_epochs": 2,
    "num_minibatches": 2,
}

_TINY_APO = {
    "env_id": "CartPole-v1",
    "n_envs": 2,
    "actor_architecture": ("16", "relu"),
    "critic_architecture": ("16", "relu"),
    "n_steps": 32,
    "batch_size": 32,
    "n_epochs": 1,
}

_TINY_AVG = {
    "env_id": "Pendulum-v1",
    "n_envs": 1,
    "actor_architecture": ("16", "relu"),
    "critic_architecture": ("16", "relu"),
}

_N_TIMESTEPS = 128
_AVG_N_TIMESTEPS = 64  # AVG steps timestep-per-iteration; smaller is enough.


# --------------------------------------------------------------------------
# Default (flag off) — last_rollout stays None on every agent.
# --------------------------------------------------------------------------
def test_ppo_default_last_rollout_is_none():
    agent = PPO(**_TINY_PPO)
    state = _unwrap(agent.train(seed=42, n_timesteps=_N_TIMESTEPS))
    assert state.last_rollout is None, (
        "PPO with expose_recent_rollout=False (default) must leave "
        "agent_state.last_rollout untouched (None)."
    )


def test_pqn_default_last_rollout_is_none():
    agent = PQN(**_TINY_PQN)
    state = _unwrap(agent.train(seed=42, n_timesteps=_N_TIMESTEPS))
    assert state.last_rollout is None


def test_apo_default_last_rollout_is_none():
    agent = APO(**_TINY_APO)
    state = _unwrap(agent.train(seed=42, n_timesteps=_N_TIMESTEPS))
    assert state.last_rollout is None


def test_avg_default_last_rollout_is_none():
    agent = AVG(**_TINY_AVG)
    state = _unwrap(agent.train(seed=42, n_timesteps=_AVG_N_TIMESTEPS))
    assert state.last_rollout is None


# --------------------------------------------------------------------------
# Opt-in (flag on) — last_rollout is populated with the full
# (T, n_envs, ...) Transition pytree and obs values are finite.
# --------------------------------------------------------------------------
def _check_exposed_rollout(state, *, expected_n_steps, expected_n_envs):
    """Shared shape + finiteness asserts on a populated ``last_rollout``."""
    rollout = state.last_rollout
    assert (
        rollout is not None
    ), "expose_recent_rollout=True must populate agent_state.last_rollout"
    obs = rollout.obs
    # vmap over seeds adds a single leading axis; the rollout's own (T,
    # n_envs) axes come right after. So obs.shape[-3:-1] == (T, n_envs).
    assert obs.shape[-3] == expected_n_steps, (
        f"expected T={expected_n_steps} on axis -3 of last_rollout.obs; "
        f"got shape {obs.shape}"
    )
    assert obs.shape[-2] == expected_n_envs, (
        f"expected n_envs={expected_n_envs} on axis -2 of last_rollout.obs; "
        f"got shape {obs.shape}"
    )
    assert jnp.all(jnp.isfinite(obs)), "last_rollout.obs contains NaN or Inf values"


def test_ppo_exposed_last_rollout_has_correct_shape():
    agent = PPO(expose_recent_rollout=True, **_TINY_PPO)
    state = _unwrap(agent.train(seed=42, n_timesteps=_N_TIMESTEPS))
    _check_exposed_rollout(
        state,
        expected_n_steps=_TINY_PPO["n_steps"],
        expected_n_envs=_TINY_PPO["n_envs"],
    )


def test_pqn_exposed_last_rollout_has_correct_shape():
    agent = PQN(expose_recent_rollout=True, **_TINY_PQN)
    state = _unwrap(agent.train(seed=42, n_timesteps=_N_TIMESTEPS))
    _check_exposed_rollout(
        state,
        expected_n_steps=_TINY_PQN["n_steps"],
        expected_n_envs=_TINY_PQN["n_envs"],
    )


def test_apo_exposed_last_rollout_has_correct_shape():
    agent = APO(expose_recent_rollout=True, **_TINY_APO)
    state = _unwrap(agent.train(seed=42, n_timesteps=_N_TIMESTEPS))
    _check_exposed_rollout(
        state,
        expected_n_steps=_TINY_APO["n_steps"],
        expected_n_envs=_TINY_APO["n_envs"],
    )


def test_avg_exposed_last_rollout_has_correct_shape():
    # AVG collects ``length=1`` per iteration; the stored rollout
    # carries that leading T=1 axis unchanged.
    agent = AVG(expose_recent_rollout=True, **_TINY_AVG)
    state = _unwrap(agent.train(seed=42, n_timesteps=_AVG_N_TIMESTEPS))
    _check_exposed_rollout(
        state,
        expected_n_steps=1,
        expected_n_envs=_TINY_AVG["n_envs"],
    )
