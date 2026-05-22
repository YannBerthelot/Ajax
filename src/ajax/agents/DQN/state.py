from flax import struct
from jax.tree_util import Partial as partial

from ajax.state import BaseAgentConfig, BaseAgentState


@partial(struct.dataclass, kw_only=True)
class DQNState(BaseAgentState):
    """DQN carries a single Q-network -- no actor, no temperature.

    The Q-network train state is stored in ``actor_state`` so the shared
    evaluation loop (hard-wired to ``agent_state.actor_state``) works
    unchanged. ``critic_state`` is set to the same object at init and
    left inert; DQN has no separate state-action critic.
    """


@struct.dataclass
class DQNConfig(BaseAgentConfig):
    """Static DQN hyperparameters carried through the compiled train loop."""

    gamma: float
    # Target-network refresh. A hard update is ``tau=1.0`` applied every
    # ``target_update_interval`` gradient steps (classic DQN); a Polyak
    # update is ``tau<1`` with ``target_update_interval=1``.
    tau: float = 1.0
    target_update_interval: int = 500
    learning_starts: int = 1000
    reward_scale: float = 1.0
    n_gradient_steps: int = 1
