from typing import TYPE_CHECKING, Any, Callable, Optional, TypeAlias, Union

import flashbax as fbx
import jax
from flax import struct
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvState as GymnaxEnvState

if TYPE_CHECKING:  # brax comes with the brax extra
    from brax.envs import Env as BraxEnv
    from brax.envs.base import State as BraxEnvState

    from ajax.modules.pid_actor import PIDPolicy  # which imports ajax.state

EnvType: TypeAlias = Union[GymnaxEnv, "BraxEnv"]
EnvStateType: TypeAlias = Union["BraxEnvState", GymnaxEnvState]
ActivationFunction: TypeAlias = Any
BufferType: TypeAlias = Union[
    fbx.flat_buffer.TrajectoryBuffer,
    fbx.trajectory_buffer.TrajectoryBuffer,
]
InitializationFunction: TypeAlias = Callable


FloatOrCallable: TypeAlias = Union[float, Callable[[int], float]]
#: A network as an agent takes it: its hidden layers and their activations
#: (``("256", "relu", "256", "relu")``), or a module in its place (an actor's
#: :class:`~ajax.modules.pid_actor.PIDPolicy`).
Architecture: TypeAlias = Union[tuple[str, ...], "PIDPolicy"]

# A deliberate workaround for a typing gap: flax.struct.dataclass adds a
# ``replace`` method at runtime but does not declare it, so a type checker
# rejects ``state.replace(...)``. A struct dataclass whose ``replace`` typed
# code calls derives from Replaceable, which declares it for the type checker
# only; at runtime it is ``object``, so the class is unchanged.
if TYPE_CHECKING:
    from typing import Self

    class Replaceable:
        def replace(self, **updates: Any) -> Self: ...

else:
    Replaceable = object


@struct.dataclass
class NormalizationInfo:
    var: jax.Array
    count: jax.Array
    mean: jax.Array
    mean_2: jax.Array
    returns: Optional[jax.Array] = None  # For reward normalization, returns are needed


@struct.dataclass
class EnvNormalizationInfo:
    reward: Optional[NormalizationInfo] = None
    obs: Optional[NormalizationInfo] = None
