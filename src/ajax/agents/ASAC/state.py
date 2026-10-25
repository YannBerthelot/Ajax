from typing import Union

import jax
from flax import struct

from ajax.agents.recurrent import RecurrentReplayConfig
from ajax.agents.temperature import SoftActorCriticState


@struct.dataclass(kw_only=True)
class ASACState(SoftActorCriticState):
    """ASAC's state: a soft actor-critic's plus the reward rate and the
    termination penalty."""

    episode_termination_penalty: jax.Array
    theta: Union[float, jax.Array]


@struct.dataclass(kw_only=True)
class ASACConfig(RecurrentReplayConfig):
    """ASAC's hyperparameters (the wrapper sets every one); no discount,
    the criterion is the average reward."""

    target_entropy: float
    tau: float
    tau_theta: float
    learning_starts: int
    reward_scale: float
    p_0: float
