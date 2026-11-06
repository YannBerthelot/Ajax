"""IQL's rewards, normalised as the plan's every other arm has them.

Contract (ajax.agents.families.RewardNormalization): normalize_rewards divides
the rewards learned from by their running standard deviation, so a task's
reward units leave what the agent learns as it was. The realistic bug: the
option does not reach IQL's replay, its target and its expectile regression
stay in the env's units, and knobs such as the temperature, which reads
advantages, mean something else on each task.
"""

import dataclasses
from typing import Any

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments.classic_control.pendulum import Pendulum

from ajax import IQL
from ajax.buffers import dataset as ds
from ajax.networks.networks import predict_value

GAIN = 1000.0
KWARGS: dict[str, Any] = {
    "normalize_rewards": True,
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
    "learning_starts": 4,
    "batch_size": 8,
    "buffer_size": 64,
}
#: Float32 keeps 7 digits of each reward, and 8 Adam steps carry the rounding
#: into values of order 1. Unnormalised, V moves GAIN times further: this
#: tolerance parts the two by orders of magnitude.
RTOL = 1e-3


class Louder(Pendulum):
    """Pendulum, its reward in other units."""

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
        obs, state, reward, done, info = super().step_env(key, state, action, params)
        return obs, state, GAIN * reward, done, info


def values(state: Any) -> np.ndarray:
    state = jax.tree.map(lambda x: x[0], state)
    rows = state.collector_state.buffer_state.experience["obs"].reshape(-1, 3)
    return np.asarray(predict_value(state.value_state, state.value_state.params, rows))


def test_reward_units_leave_iqls_online_values_as_they_were() -> None:
    plain = IQL(Pendulum(), **KWARGS).train(seed=0, n_timesteps=24)[0]
    louder = IQL(Louder(), **KWARGS).train(seed=0, n_timesteps=24)[0]

    np.testing.assert_allclose(values(louder), values(plain), rtol=RTOL, atol=RTOL)


def test_reward_units_leave_iqls_offline_values_as_they_were() -> None:
    """Offline, the statistics come from the dataset's rows (seed_replay)."""
    params = gymnax.make("Pendulum-v1")[1].replace(max_steps_in_episode=8)
    plain = IQL(Pendulum(), env_params=params, **KWARGS)
    zero = lambda obs: jnp.zeros((*obs.shape[:-1], 1))  # noqa: E731
    rows = ds.record(zero, plain.env_args, jax.random.PRNGKey(0), 32)
    louder = dataclasses.replace(rows, reward=GAIN * rows.reward)

    plain_state = plain.train_offline(rows, n_updates=8)[0]
    louder_state = IQL(Pendulum(), env_params=params, **KWARGS).train_offline(
        louder, n_updates=8
    )[0]

    np.testing.assert_allclose(
        values(louder_state), values(plain_state), rtol=RTOL, atol=RTOL
    )


def test_normalising_is_off_by_default() -> None:
    """The reference's rewards are its own: reward_scale alone moves them."""
    assert IQL("Pendulum-v1").normalize_rewards is False


def test_a_discount_for_the_normaliser_needs_the_normaliser() -> None:
    with pytest.raises(ValueError, match="normalize_rewards"):
        IQL("Pendulum-v1", reward_normalization_gamma=0.99)
