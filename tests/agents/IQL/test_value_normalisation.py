"""IQL's state value reads the observations its actor and critics read.

Contract (ajax.environments.obs_norm): under normalize_observations every
network applies the collector's running statistics, so a rescaled sensor
leaves what each network reads, and what it predicts, as it was. The
realistic bug: V keeps reading the env's own units while Q and the actor
read normalised ones, so its expectile regression and the critics' target
live on another input scale.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import IQL
from ajax.buffers import dataset as ds
from ajax.networks.networks import predict_value
from tests.environments.identity_env import IdentityEnv

GAIN, OFFSET = 1000.0, 7.0
KWARGS: dict[str, Any] = {
    "normalize_observations": True,
    "actor_architecture": ("8", "relu"),
    "critic_architecture": ("8", "relu"),
    "learning_starts": 4,
    "batch_size": 4,
    "buffer_size": 64,
}
#: Float32 rounds the rescaled sensor (values up to 5007) to about 5e-4 in
#: its own units, 1e-6 once normalised; 8 Adam steps carry that into values
#: of order 1. The unnormalised V is about GAIN times off, so 1e-3 parts it
#: from the fixed one by orders of magnitude.
RTOL = 1e-3


class RescaledSensor(IdentityEnv):
    """The identity task behind a sensor reading gain * x + offset."""

    def get_obs(self, state: Any, params: Any = None, key: Any = None) -> jax.Array:
        return GAIN * super().get_obs(state, params, key) + OFFSET


def online(agent: IQL) -> Any:
    return agent.train(seed=0, n_timesteps=24)[0]


def offline(agent: IQL) -> Any:
    expert = lambda obs: jnp.zeros(obs.shape[:-1] + (1,))  # noqa: E731
    rows = ds.record(expert, agent.env_args, jax.random.PRNGKey(0), 24)
    return agent.train_offline(rows, n_updates=8)[0]


def values_on_held_rows(env: IdentityEnv, train: Callable[[IQL], Any]) -> np.ndarray:
    """V on the rows the replay holds, in that env's own units."""
    state = jax.tree.map(lambda x: x[0], train(IQL(env, n_envs=2, **KWARGS)))
    replay = state.collector_state.buffer_state
    rows = replay.experience["obs"].reshape(-1, 2)
    held = rows[rows.sum(-1) != 0]
    return np.asarray(predict_value(state.value_state, state.value_state.params, held))


@pytest.mark.parametrize("train", [online, offline])
def test_a_rescaled_sensor_leaves_the_state_value_as_it_was(train) -> None:
    plain = values_on_held_rows(IdentityEnv(), train)
    rescaled = values_on_held_rows(RescaledSensor(), train)
    assert plain.size == rescaled.size > 0
    np.testing.assert_allclose(rescaled, plain, rtol=RTOL, atol=RTOL)
