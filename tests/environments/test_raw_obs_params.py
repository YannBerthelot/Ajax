"""An expert reads the run's sensors: get_raw_obs under the params in use.

Contract (Fix: an expert must get the agent's readings, not the default
params'): the raw observation an expert is given is ``env.get_obs(state,
params)`` for the params the run trains on, one system per env where they are
batched. The realistic bug: reading ``env.default_params`` gives the expert
another sensor than the agent as soon as a run sets a sensor through params.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from gymnax.environments import environment, spaces

from ajax import SAC
from ajax.environments.interaction import get_raw_obs
from ajax.environments.system_class import broadcast_env_params
from ajax.extensions.pretrain import ExpertPrefill

SHORT: dict[str, Any] = {"learning_starts": 4, "batch_size": 2, "buffer_size": 64}
SHORT.update({"actor_architecture": ("4", "relu")})
SHORT.update({"critic_architecture": ("4", "relu")})


@struct.dataclass
class SensorParams(environment.EnvParams):
    gain: float = 1.0
    max_steps_in_episode: int = 4


class GainedSensor(environment.Environment):
    """The observation is the sensor's gain, a parameter of the run."""

    @property
    def default_params(self) -> SensorParams:
        return SensorParams()

    def reset_env(self, key: Any, params: SensorParams) -> tuple:
        state = environment.EnvState(time=0)
        return self.get_obs(state, params), state

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> tuple:
        state = state.replace(time=state.time + 1)
        return self.get_obs(state, params), state, 0.0, jnp.array(False), {}

    def get_obs(self, state: Any, params: Any = None, key: Any = None) -> jax.Array:
        return jnp.full((1,), params.gain, jnp.float32)

    def observation_space(self, params: Any) -> spaces.Box:
        return spaces.Box(-jnp.inf, jnp.inf, (1,))

    def action_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (1,))


def _zero(obs: jax.Array) -> jax.Array:
    return jnp.zeros(obs.shape[:-1] + (1,))


def test_raw_observations_read_the_params_in_use() -> None:
    state = environment.EnvState(time=jnp.array(0))
    raw = get_raw_obs(state, GainedSensor(), "gymnax", SensorParams(gain=3.0))
    np.testing.assert_array_equal(raw, [3.0])


def test_each_env_reads_its_own_systems_params() -> None:
    """Batched params hold one system per env: env i reads gain i + 1."""
    params = broadcast_env_params(SensorParams(), 2).replace(gain=jnp.array([1.0, 2.0]))
    states = jax.tree.map(
        lambda x: jnp.stack([x, x]), environment.EnvState(time=jnp.array(0))
    )
    raw = get_raw_obs(states, GainedSensor(), "gymnax", params)
    np.testing.assert_array_equal(raw, [[1.0], [2.0]])


@pytest.mark.parametrize("extensions", [(), (ExpertPrefill(n_steps=4),)])
def test_a_trained_agents_expert_reads_what_the_agent_reads(extensions) -> None:
    """Collection and the prefill's rollout store the raw observation the
    expert read: under gain 3 it equals the agent's own, not the default's."""
    agent = SAC(
        GainedSensor(),
        env_params=SensorParams(gain=3.0),
        n_envs=1,
        expert_policy=_zero,
        extensions=extensions,
        **SHORT,
    )
    state, _ = agent.train(seed=0, n_timesteps=8)
    replay = state.collector_state.buffer_state
    assert replay is not None
    written = int(np.asarray(replay.current_index)[0])
    rows = replay.experience
    np.testing.assert_array_equal(np.asarray(rows["obs"])[..., :written, 0], 3.0)
    np.testing.assert_array_equal(np.asarray(rows["raw_obs"])[..., :written, 0], 3.0)
