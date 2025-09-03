from typing import Any, Callable, Optional, TypeAlias, Union

import flashbax as fbx
import jax.numpy as jnp
from brax.envs import Env as BraxEnv
from brax.envs.base import State as BraxEnvState
from flax import struct
from flax.core.frozen_dict import FrozenDict
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvState as GymnaxEnvState

HiddenState: TypeAlias = Union[jnp.ndarray, FrozenDict]
EnvType: TypeAlias = Union[GymnaxEnv, BraxEnv]
EnvStateType: TypeAlias = Union[BraxEnvState, GymnaxEnvState]
ActivationFunction: TypeAlias = Any  # Union[
#     jax._src.custom_derivatives.custom_jvp,
#     jaxlib.xla_extension.PjitFunction,
# ]
BufferType: TypeAlias = Union[
    fbx.flat_buffer.TrajectoryBuffer,
    fbx.trajectory_buffer.TrajectoryBuffer,
]

BufferTypeState: TypeAlias = Union[
    fbx.flat_buffer.TrajectoryBufferState,
    fbx.trajectory_buffer.TrajectoryBufferState,
]
InitializationFunction: TypeAlias = Callable


FloatOrCallable: TypeAlias = Union[float, Callable[[int], float]]


@struct.dataclass
class NormalizationInfo:
    var: jnp.array
    count: jnp.array
    mean: jnp.array
    mean_2: jnp.array
    returns: Optional[jnp.array] = None  # For reward normalization, returns are needed


@struct.dataclass
class EnvNormalizationInfo:
    reward: Optional[NormalizationInfo] = None
    obs: Optional[NormalizationInfo] = None
