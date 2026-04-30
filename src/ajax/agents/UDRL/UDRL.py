"""Public UDRL agent class. Mirrors the Ajax convention used by PPO/SAC: a
thin wrapper over the shared ActorCritic base that builds env / network /
optimizer configs and forwards them to a per-agent ``make_train``.
"""

from functools import partial
from typing import Optional

from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.UDRL.state import UDRLConfig
from ajax.agents.UDRL.train_UDRL import make_train
from ajax.types import EnvType


class UDRL(ActorCritic):
    """Upside-Down RL (Schmidhuber, 2019). Command-conditioned supervised
    policy learning. No critic, no replay buffer."""

    name: str = "UDRL"

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 3e-4,
        actor_architecture=("128", "relu", "128", "relu"),
        gamma: float = 1.0,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 0.5,
        n_steps: int = 64,
        batch_size: int = 64,
        n_epochs: int = 4,
        command_return_init: float = 1.0,
        command_horizon_init: float = 100.0,
        bc_loss_type: str = "nll",
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
    ) -> None:
        self.config = {**locals()}
        self.config.update({"algo_name": "UDRL"})

        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=actor_learning_rate,  # critic is unused
            actor_architecture=actor_architecture,
            critic_architecture=actor_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
        )

        self.agent_config = UDRLConfig(
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            command_return_init=command_return_init,
            command_horizon_init=command_horizon_init,
            bc_loss_type=bc_loss_type,
        )

    def get_make_train(self):
        return partial(make_train)
