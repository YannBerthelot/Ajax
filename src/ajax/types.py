from typing import TypeAlias, Union

import flashbax as fbx
import jax
import jax.numpy as jnp
import jaxlib
from brax.envs import Env as BraxEnv
from brax.envs.base import State as BraxEnvState
from flax.core.frozen_dict import FrozenDict
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvState as GymnaxEnvState

HiddenState = Union[jnp.ndarray, FrozenDict]
EnvType = Union[GymnaxEnv, BraxEnv]
EnvStateType = Union[BraxEnvState, GymnaxEnvState]
ActivationFunction: TypeAlias = Union[
    jax._src.custom_derivatives.custom_jvp, jaxlib.xla_extension.PjitFunction
]
BufferType: TypeAlias = Union[
    fbx.flat_buffer.TrajectoryBuffer, fbx.trajectory_buffer.TrajectoryBuffer
]

BufferTypeState: TypeAlias = Union[
    fbx.flat_buffer.TrajectoryBufferState, fbx.trajectory_buffer.TrajectoryBufferState
]
