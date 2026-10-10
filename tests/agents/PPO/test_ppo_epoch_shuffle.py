"""PPO and APO reshuffle the rollout into fresh minibatches every epoch.

OpenAI baselines' ppo2 and CleanRL's ``ppo.py`` shuffle inside their epoch
loop; APO's official code (``xtma/apo``, ``APPO.optimize_agent``) draws
``iterate_mb_idxs(..., shuffle=True)`` once per epoch. One update runs on a
synthetic rollout whose observation ``(t, e, 0)`` names its row; the critic
step records the observations of every minibatch it trains on.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.APO import train_APO
from ajax.agents.APO.APO import APO
from ajax.agents.APO.state import APOConfig
from ajax.agents.PPO import train_PPO
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PPO.state import PPOConfig
from ajax.extensions.base import ExtensionStack
from ajax.networks.memory import MemoryConfig
from ajax.state import Transition

N_STEPS, N_EPOCHS = 16, 3
SMALL: dict[str, Any] = {"env_id": "Pendulum-v1", "n_steps": N_STEPS}
SMALL |= {"n_envs": 2, "n_epochs": N_EPOCHS, "num_minibatches": 4}
SMALL |= {"actor_architecture": ("8", "tanh"), "critic_architecture": ("8", "tanh")}


def _rollout(n_envs: int) -> Transition:
    t, e = jnp.meshgrid(jnp.arange(N_STEPS), jnp.arange(n_envs), indexing="ij")
    obs = jnp.stack([t, e, jnp.zeros_like(t)], -1).astype(jnp.float32)
    column = jnp.zeros((N_STEPS, n_envs, 1))
    return Transition(
        obs=obs,
        action=column,
        reward=column,
        terminated=column,
        truncated=column,
        next_obs=obs + 0.5,
        raw_obs=obs,
        log_prob=column,
        raw_action=column,
    )


def _record(seen: list[np.ndarray], obs: jax.Array) -> None:
    jax.debug.callback(lambda o: seen.append(np.asarray(o)), obs, ordered=True)


def _assert_fresh_partitions(seen: list[np.ndarray], n_envs: int, k: int) -> None:
    """Every epoch's ``k`` minibatches cover each ``(t, e)`` row once, and
    no two epochs split the rollout the same way."""
    assert len(seen) == N_EPOCHS * k, len(seen)
    every_row = {(t, e) for t in range(N_STEPS) for e in range(n_envs)}
    partitions = []
    for i in range(N_EPOCHS):
        minibatches = [
            frozenset(map(tuple, obs.reshape(-1, 3)[:, :2].astype(int).tolist()))
            for obs in seen[i * k : (i + 1) * k]
        ]
        assert set().union(*minibatches) == every_row
        assert sum(map(len, minibatches)) == len(every_row)
        partitions.append(set(minibatches))
    for i in range(N_EPOCHS):
        for j in range(i):
            assert partitions[i] != partitions[j], (i, j)


@pytest.mark.parametrize(
    "geometry",
    [
        {},  # flat: 2 envs do not split into 4 minibatches
        {"n_envs": 8},  # time: the env axis split
        {"unroll_length": 2},  # time: fragments of 2 steps
        {"bptt_length": 2, "memory": MemoryConfig(kind="gru", hidden_size=4)},
    ],
    ids=["flat", "env-split", "unroll", "recurrent"],
)
def test_ppo_reshuffles_every_epoch(
    geometry: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = PPO(**{**SMALL, **geometry})
    config = cast(PPOConfig, agent.agent_config)
    n_envs = agent.env_args.n_envs
    seen: list[np.ndarray] = []
    value_loss = train_PPO.value_loss_function

    def spy(params: Any, critic: Any, obs: jax.Array, *args: Any, **kw: Any) -> Any:
        _record(seen, obs)
        return value_loss(params, critic, obs, *args, **kw)

    monkeypatch.setattr(train_PPO, "value_loss_function", spy)
    state = train_PPO.init_PPO(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    train_PPO.update_agent(
        state,
        _rollout(n_envs),
        state,
        config,
        agent.env_args,
        "gymnax",
        agent.network_args.memory is not None,
        ExtensionStack(()),
        total_timesteps=1_000,
        total_n_updates=10,
    )
    jax.effects_barrier()
    _assert_fresh_partitions(seen, n_envs, config.num_minibatches)


def test_apo_reshuffles_every_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = APO(**SMALL)
    seen: list[np.ndarray] = []
    update_values = train_APO.update_value_functions

    def spy(agent_state: Any, obs: jax.Array, *args: Any) -> Any:
        _record(seen, obs)
        return update_values(agent_state, obs, *args)

    monkeypatch.setattr(train_APO, "update_value_functions", spy)
    state = train_APO.init_APO(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    train_APO.update_agent(
        state,
        _rollout(SMALL["n_envs"]),
        cast(APOConfig, agent.agent_config),
        ExtensionStack(()),
        total_timesteps=1_000,
    )
    jax.effects_barrier()
    _assert_fresh_partitions(seen, SMALL["n_envs"], SMALL["num_minibatches"])
