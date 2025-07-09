from typing import Any, Callable, Optional, Tuple, Union

import flashbax as fbx
import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from gymnax import EnvParams
from jax.tree_util import Partial as partial

from ajax.types import EnvStateType, EnvType, InitializationFunction
from ajax.wrappers import NormalizationInfo


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
    n_envs: int
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


@partial(struct.dataclass, kw_only=True)
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


def normalize_observation(obs: jax.Array, norm_info: NormalizationInfo) -> jax.Array:
    """Normalize the observation using the normalization info."""
    if norm_info is None or norm_info.var is None:
        return obs
    return (obs - norm_info.mean) / jnp.sqrt(norm_info.var + 1e-8)


def unnormalize_observation(obs: jax.Array, norm_info: NormalizationInfo) -> jax.Array:
    """Unnormalize the observation using the normalization info."""
    if norm_info is None or norm_info.var is None:
        return obs
    return obs * jnp.sqrt(norm_info.var + 1e-8) + norm_info.mean


def simplex(a, b, dyna_factor):
    # Use a fused multiply-add to minimize rounding errors:
    return jnp.add(a, dyna_factor * (b - a))


def get_double_train_state(second_state_type: str, dyna_factor: float = 0.5):
    assert second_state_type in [
        "avg",
        "sac",
    ], f"Invalid second_state_type: {second_state_type}. Expected 'avg' or 'sac'."

    @struct.dataclass
    class DoubleTrainState(LoadedTrainState):
        second_state: Optional[LoadedTrainState] = None
        norm_info: Optional[NormalizationInfo] = None
        hidden_state: Optional[Any] = None

        @classmethod
        def from_LoadedTrainState(
            cls,
            lts: LoadedTrainState,
            second_state: LoadedTrainState,
            norm_info: Optional[NormalizationInfo] = None,
        ):
            return cls(
                step=lts.step,
                apply_fn=lts.apply_fn,
                params=lts.params,
                tx=lts.tx,
                opt_state=lts.opt_state,
                target_params=lts.target_params,
                hidden_state=lts.hidden_state,
                recurrent=lts.recurrent,
                second_state=second_state,
                norm_info=norm_info,
            )

        def apply(self, params, obs, *args, **kwargs):
            """Call the apply_fn with the given parameters and arguments."""
            if (
                second_state_type == "avg"
            ):  # This means first state is SAC, so we need to normalize the raw obs
                if self.norm_info is None:
                    processed_obs = obs
                    print(
                        "Warning: norm_info or env is None, not normalizing"
                        " observations."
                    )
                else:
                    processed_obs = normalize_observation(
                        obs,
                        norm_info=jax.tree.map(
                            lambda x: x[0].reshape(1, -1), self.norm_info.obs
                        ),
                    )
            elif (
                second_state_type == "sac"
            ):  # This means first state is AVG, so we need to unnormalize the obs
                if self.norm_info is None:
                    processed_obs = obs
                    print(
                        "Warning: norm_info or env is None, not unnormalizing"
                        " observations."
                    )
                else:
                    processed_obs = unnormalize_observation(
                        obs,
                        norm_info=jax.tree.map(
                            lambda x: x[0].reshape(1, -1), self.norm_info.obs
                        ),  # as the dimensions are only repeats, keep only the first one to prevent messy broadcasting
                    )
            else:
                raise ValueError(
                    f"Invalid second_state_type: {second_state_type}. Expected"
                    " 0:'avg' or 1: 'sac'."
                )
            assert (
                obs.shape == processed_obs.shape
            ), f"{obs.shape} != {processed_obs.shape}"

            raw_output = self.apply_fn(params, obs, *args, **kwargs)
            second_output = self.second_state.apply_fn(
                self.second_state.target_params,
                obs,
                *args,
                **kwargs,
            )
            if isinstance(second_output, jnp.ndarray):
                assert jnp.all(
                    jnp.isfinite(second_output)
                ), "second_output has NaN or Inf!"

            _dyna_factor = dyna_factor(self.step).astype(jnp.float32)

            if not isinstance(raw_output, jnp.ndarray):
                # assume raw_output is a SquashedNormal distribution TODO : Make this for any distrax distributon?
                complete_output = raw_output.mix_distributions(
                    jax.lax.stop_gradient(second_output),
                    dyna_factor=jax.lax.stop_gradient(_dyna_factor),
                )

            else:
                complete_output = simplex(
                    raw_output,
                    jax.last.stop_gradient(second_output),
                    jax.lax_stop_gradient(_dyna_factor),
                )

            return complete_output

    return DoubleTrainState


@struct.dataclass
class BaseAgentState:
    rng: jax.Array
    actor_state: LoadedTrainState
    critic_state: LoadedTrainState
    collector_state: CollectorState
    eval_rng: jax.Array
    n_updates: int = 0
    n_logs: int = 0
    index: Optional[int] = None

    def replace(self, *args, **kwargs):  # To make mypy happy
        """Replace fields in the dataclass with new values."""
        return struct.replace(self, *args, **kwargs)


@struct.dataclass
class BaseAgentConfig:
    gamma: float


@struct.dataclass
class NetworkConfig:
    actor_architecture: Tuple[str]
    critic_architecture: Tuple[str]
    lstm_hidden_size: Optional[int] = None
    squash: bool = False
    penultimate_normalization: bool = False
    actor_kernel_init: Optional[Union[str, InitializationFunction]] = None
    actor_bias_init: Optional[Union[str, InitializationFunction]] = None
    critic_kernel_init: Optional[Union[str, InitializationFunction]] = None
    critic_bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None


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
    n_envs: int
