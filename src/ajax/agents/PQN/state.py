from flax import struct
from jax.tree_util import Partial as partial

from ajax.state import BaseAgentConfig, BaseAgentState


@partial(struct.dataclass, kw_only=True)
class PQNState(BaseAgentState):
    """PQN carries a single Q-network -- no replay buffer, no target network.

    The Q-network train state is stored in ``actor_state`` so the shared
    evaluation loop (hard-wired to ``agent_state.actor_state``) works
    unchanged; ``critic_state`` mirrors it at init and is left inert.
    """


@struct.dataclass
class PQNConfig(BaseAgentConfig):
    """Static PQN hyperparameters carried through the compiled train loop."""

    gamma: float = 0.99
    # Q(lambda) trace-decay coefficient (0 -> 1-step TD, 1 -> Monte-Carlo).
    q_lambda: float = 0.65
    n_steps: int = 128
    n_epochs: int = 4
    # Minibatches per epoch; must divide n_steps.
    num_minibatches: int = 4
    reward_scale: float = 1.0
