"""Public UDRL agent class. Mirrors the Ajax convention used by PPO/SAC: a
thin wrapper over the shared ActorCritic base that builds env / network /
optimizer configs and forwards them to a per-agent ``make_train``.
"""

from collections.abc import Sequence
from functools import partial
from typing import Optional, Tuple

from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.UDRL.state import UDRLConfig
from ajax.agents.UDRL.train_UDRL import make_train
from ajax.extensions.base import Extension
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
        command_topk: int = 32,
        command_return_boost: float = 1.0,
        command_target_tau: float = 1.0,
        command_scale_r: float = 0.02,
        command_scale_h: float = 0.01,
        buffer_capacity: int = 64,
        n_updates_per_iter: int = 64,
        bc_loss_type: str = "nll",
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        # If set, use a CNN encoder. Image is assumed to be packed as
        # ``(*batch, H*W*C + 2)`` flat — the last 2 dims are the UDRL
        # command, which is concatenated to the CNN embedding after the
        # convolutions. None keeps the legacy MLP encoder.
        cnn_image_shape: Optional[Tuple[int, int, int]] = None,
        # --- New surface: composable research features as Extensions ---
        extensions: Sequence[Extension] = (),
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
            extensions=extensions,
        )

        self.agent_config = UDRLConfig(
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            command_return_init=command_return_init,
            command_horizon_init=command_horizon_init,
            command_topk=command_topk,
            command_return_boost=command_return_boost,
            command_target_tau=command_target_tau,
            command_scale_r=command_scale_r,
            command_scale_h=command_scale_h,
            buffer_capacity=buffer_capacity,
            n_updates_per_iter=n_updates_per_iter,
            bc_loss_type=bc_loss_type,
        )
        self.cnn_image_shape = cnn_image_shape

    def get_make_train(self):
        return partial(
            make_train,
            cnn_image_shape=self.cnn_image_shape,
            extensions=tuple(self.extension_stack.extensions),
        )
