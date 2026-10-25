"""The collector steps the environment's own state, also when the agent
conditions on the training fraction."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from ajax import SAC
from ajax.extensions.observe import TrainingProgressInput


@struct.dataclass
class _Params(environment.EnvParams):
    max_steps_in_episode: int = 8


class _ReplacingCounter(environment.Environment):
    """Observes its step count; steps by ``state.replace``, as many gymnax
    environments do, so it returns the class of the state it is given."""

    @property
    def default_params(self) -> _Params:
        return _Params()

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
        state = state.replace(time=state.time + 1)
        return self.get_obs(state), state, jnp.float32(0.0), jnp.bool_(False), {}

    def reset_env(self, key: Any, params: Any) -> Any:
        state = environment.EnvState(time=0)
        return self.get_obs(state), state

    def get_obs(self, state: Any, params: Any = None, key: Any = None) -> Any:
        return jnp.asarray(state.time, jnp.float32).reshape(1)

    def action_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (1,), dtype=jnp.float32)

    def observation_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(0.0, 8.0, (1,), dtype=jnp.float32)


def test_an_agent_on_the_training_fraction_steps_the_envs_own_state():
    """SAC with TrainingProgressInput trains on an env stepping by
    state.replace (gymnax's auto-reset selects between the reset state and
    the stepped one: they must be one class), and its collector holds the
    env's own state class, the fraction being the collector's, not the
    state's."""
    env = _ReplacingCounter()
    agent = SAC(
        env,
        env_params=env.default_params,
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        extensions=(TrainingProgressInput(),),
    )
    state, _ = agent.train(seed=0, n_timesteps=32)
    collector = state.collector_state
    assert type(collector.env_state) is environment.EnvState
    assert float(jnp.ravel(collector.train_time_fraction)[0]) == 1.0
    jax.block_until_ready(state)
