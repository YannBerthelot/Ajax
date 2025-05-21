from typing import Any, Callable, Optional, Tuple

import flashbax as fbx
import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from gymnax import EnvParams

from ajax.types import EnvStateType, EnvType


@struct.dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    next_obs: jnp.ndarray
    log_prob: Optional[jnp.ndarray] = None


@struct.dataclass
class EnvironmentConfig:
    env: EnvType
    env_params: EnvParams
    num_envs: int
    continuous: bool


@struct.dataclass
class RollingMeanState:
    buffer: jnp.ndarray  # shape (window_size,n_envs)
    index: jnp.ndarray  # shape (1,n_envs)
    count: jnp.ndarray  # shape (1,n_envs)
    sum: jnp.ndarray  # shape (1,n_envs)


@struct.dataclass
class RollinEpisodicMeanRewardState(RollingMeanState):
    last_return: jnp.ndarray
    cumulative_reward: jnp.ndarray


@struct.dataclass
class CollectorState:
    """The variables necessary to interact with the environment and collect the transitions"""

    rng: jax.Array
    env_state: EnvStateType
    last_obs: jnp.ndarray
    last_terminated: jnp.ndarray
    last_truncated: jnp.ndarray
    episodic_return_state: RollinEpisodicMeanRewardState
    episodic_mean_return: float = jnp.nan
    num_update: int = 0
    timestep: int = 0
    average_reward: float = 0.0
    buffer_state: Optional[fbx.flat_buffer.TrajectoryBufferState] = None
    rollout: Optional[Transition] = None


@struct.dataclass
class LoadedTrainState(TrainState):
    hidden_state: Optional[Any] = None
    recurrent: bool = False
    target_params: Optional[flax.core.FrozenDict] = None

    def soft_update(self, tau):
        new_target_params = optax.incremental_update(
            self.params,
            self.target_params,
            tau,
        )
        return self.replace(target_params=new_target_params)

    @classmethod
    def create(cls, *, hidden_state=None, apply_fn: Callable, **kwargs):
        # Ensure apply_fn is passed to the parent TrainState
        instance = super().create(apply_fn=apply_fn, **kwargs)
        # Determine if the state is recurrent
        recurrent = hidden_state is not None
        # Return a new instance with hidden_state and recurrent attributes
        return instance.replace(hidden_state=hidden_state, recurrent=recurrent)

    def apply(self, params, *args, **kwargs):
        """Call the apply_fn with the given parameters and arguments."""
        return self.apply_fn(params, *args, **kwargs)


@struct.dataclass
class BaseAgentState:
    rng: jax.Array
    actor_state: LoadedTrainState
    critic_state: LoadedTrainState
    collector_state: CollectorState
    eval_rng: jax.Array


@struct.dataclass
class AgentConfig:
    seed: int
    gamma: float


@struct.dataclass
class NetworkConfig:
    actor_architecture: Tuple[str]
    critic_architecture: Tuple[str]
    lstm_hidden_size: Optional[int] = None
    squash: bool = False
    penultimate_normalization: bool = False


@struct.dataclass
class OptimizerConfig:
    learning_rate: float | Callable[[int], float]
    max_grad_norm: Optional[float] = 0.5
    clipped: bool = True
    beta_1: float = 0.9
    beta_2: float = 0.999


@struct.dataclass
class AlphaConfig:
    alpha_init: float
    learning_rate: float


@struct.dataclass
class BufferConfig:
    buffer_size: int
    batch_size: int
    num_envs: int


@struct.dataclass
class BaseAgentConfig:
    """The agent properties to be carried over iterations of environment interaction and updates"""

    gamma: float
