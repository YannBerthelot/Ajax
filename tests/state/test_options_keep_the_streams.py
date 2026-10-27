"""Extensions and options that change which draws an update makes leave
both streams of keys as they were (ajax.random_keys): a key that exists only
under an extension, an option or a count is derived, never drawn from a
stream, so two runs that differ by one of them share every other key. The
updates' stream and the collection's end where they end without it: with
extensions or without; an actor step that counts on every update or on some
(a policy delay, policy_update_start); one critic step per update or several
(SAC's and REDQ's num_critic_updates, DQN's n_gradient_steps); one PPO
minibatch or several. With memory or without, a network's encoder starts
from the same weights."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import ajax
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.extensions.instrumentation import ConditioningMetrics
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import Actor, init_network_state
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY
from tests.agents.extension_stubs import CounterExt

NET = {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")}
PENDULUM: dict[str, Any] = {"env_id": "Pendulum-v1", "n_envs": 2, **NET}
REPLAY: dict[str, Any] = {**PENDULUM, "buffer_size": 64, "batch_size": 4}
REPLAY.update({"learning_starts": 8})
SAC: dict[str, Any] = {**REPLAY, "policy_update_start": 0}
REDQ: dict[str, Any] = {**REPLAY, "num_critic_updates": 2}
ON_POLICY: dict[str, Any] = {"env_id": "Pendulum-v1", "n_envs": 2, "n_steps": 8}
ON_POLICY.update({"batch_size": 8, "n_epochs": 1})
ON_POLICY.update(
    {"actor_architecture": ("8", "tanh"), "critic_architecture": ("8", "tanh")}
)
DQN: dict[str, Any] = {"env_id": "CartPole-v1", "n_envs": 2, "buffer_size": 64}
DQN.update({"architecture": ("8", "relu"), "batch_size": 4, "learning_starts": 8})
DREAMER: dict[str, Any] = {**DREAMER_TINY, "env_id": "CartPole-v1", "n_envs": 2}
DREAMER.update({"train_ratio": 32, "replay_capacity": 80})


def streams(cls: Any, kwargs: dict[str, Any], steps: int) -> tuple[Any, Any]:
    """The updates' key and the collection's at the end of a tiny run."""
    state, _ = cls(**kwargs).train(seed=0, n_timesteps=steps)
    return np.asarray(state.rng), np.asarray(state.collector_state.rng)


def assert_same_streams(
    cls: Any, kwargs: dict[str, Any], options: dict[str, Any], steps: int
) -> None:
    with_options = dict(kwargs)
    with_options.update(options)
    for plain, changed in zip(
        streams(cls, kwargs, steps), streams(cls, with_options, steps), strict=True
    ):
        np.testing.assert_array_equal(plain, changed)


#: name: (the agent, its arguments, the env steps).
EXTENDED: dict[str, tuple[Any, dict[str, Any], int]] = {
    "SAC": (ajax.SAC, SAC, 16),
    "TD3": (ajax.TD3, REPLAY, 16),
    "PPO": (ajax.PPO, ON_POLICY, 32),
    "DQN": (ajax.DQN, DQN, 16),
    "DreamerV3": (DreamerV3, DREAMER, 32),
}


@pytest.mark.parametrize("name", EXTENDED)
@pytest.mark.parametrize("extension", [CounterExt(), ConditioningMetrics()])
def test_an_extension_moves_neither_stream(name: str, extension: Any) -> None:
    """An inert post_update (CounterExt) and an evaluation metric
    (ConditioningMetrics) leave both streams where they end without them."""
    cls, kwargs, steps = EXTENDED[name]
    assert_same_streams(cls, kwargs, {"extensions": (extension,)}, steps)


#: name: (the agent, its arguments, the option changed, the env steps).
OPTIONS: dict[str, tuple[Any, dict[str, Any], dict[str, Any], int]] = {
    "TD3-policy_delay": (ajax.TD3, REPLAY, {"policy_delay": 3}, 24),
    "SAC-policy_update_start": (ajax.SAC, SAC, {"policy_update_start": 16}, 24),
    "REDQ-policy_update_start": (ajax.REDQ, REDQ, {"policy_update_start": 16}, 24),
    "TQC-policy_update_start": (ajax.TQC, REPLAY, {"policy_update_start": 16}, 24),
    "ASAC-policy_update_start": (ajax.ASAC, REPLAY, {"policy_update_start": 16}, 24),
    "AVG-policy_update_start": (
        ajax.AVG,
        {**PENDULUM, "learning_starts": 8},
        {"policy_update_start": 16},
        24,
    ),
    "SAC-num_critic_updates": (ajax.SAC, SAC, {"num_critic_updates": 2}, 16),
    "REDQ-num_critic_updates": (ajax.REDQ, REDQ, {"num_critic_updates": 3}, 16),
    "DQN-n_gradient_steps": (ajax.DQN, DQN, {"n_gradient_steps": 2}, 16),
    "PPO-num_minibatches": (ajax.PPO, ON_POLICY, {"batch_size": 4}, 32),
}


@pytest.mark.parametrize("name", OPTIONS)
def test_an_option_moves_neither_stream(name: str) -> None:
    cls, kwargs, options, steps = OPTIONS[name]
    assert_same_streams(cls, kwargs, options, steps)


def test_memory_leaves_the_encoders_initial_weights() -> None:
    """A recurrent network initialises on the key a feed-forward one does,
    its carry on a key derived from it: the encoder both have starts from
    the same weights."""
    key, observation = jax.random.PRNGKey(4), jnp.zeros((1, 3))
    encoders = []
    for memory in (None, MemoryConfig(kind="gru", hidden_size=8)):
        actor = Actor(("8", "relu"), action_dim=1, continuous=True, memory=memory)
        state = init_network_state(observation, actor, key, optax.adam(1e-3), 2, memory)
        encoders.append(state.params["params"]["encoder"])
    jax.tree.map(np.testing.assert_array_equal, *encoders)
