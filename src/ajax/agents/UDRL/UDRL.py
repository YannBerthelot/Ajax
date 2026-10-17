from collections.abc import Sequence
from typing import Optional, Tuple

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.UDRL.state import UDRLConfig
from ajax.agents.UDRL.train_UDRL import make_train
from ajax.extensions.base import Extension
from ajax.types import EnvType


class UDRL(ActorCritic):
    """Upside-Down RL (Schmidhuber, 2019). Command-conditioned supervised
    policy learning on a replay buffer of rollout segments; no critic is
    trained."""

    name: str = "UDRL"
    supported_extension_phases: frozenset = LOOP_PHASES | {"actor_loss"}
    # It learns from the segments of up to buffer_capacity past iterations.
    replays_transitions: bool = True
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 3e-4,
        actor_architecture=("128", "relu", "128", "relu"),
        gamma: float = 1.0,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        n_steps: int = 64,
        batch_size: int = 64,
        command_return_init: float = 1.0,
        command_horizon_init: float = 100.0,
        command_topk: int = 32,
        command_return_boost: float = 1.0,
        command_target_tau: float = 1.0,
        command_scale_r: float = 0.02,
        command_scale_h: float = 0.01,
        buffer_capacity: int = 64,
        n_updates_per_iter: int = 64,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        # If set, use a CNN encoder (NetworkConfig.cnn_image_shape) on the
        # env's flat (H*W*C,) image; the (d_r, d_h) command UDRL appends
        # joins the CNN embedding after the convolutions. None keeps the
        # MLP encoder.
        cnn_image_shape: Optional[Tuple[int, int, int]] = None,
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        self.config = run_config(locals(), self.name)

        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=actor_learning_rate,  # critic is unused
            actor_architecture=actor_architecture,
            critic_architecture=actor_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            reward_normalization_gamma=reward_normalization_gamma,
            # Kept from Ajax's earlier networks: a LayerNorm on the policy
            # encoder's output (the shared default is now a plain MLP).
            encoder_layer_norm=True,
            cnn_image_shape=cnn_image_shape,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )

        self.agent_config: UDRLConfig = UDRLConfig(
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            command_return_init=command_return_init,
            command_horizon_init=command_horizon_init,
            command_topk=command_topk,
            command_return_boost=command_return_boost,
            command_target_tau=command_target_tau,
            command_scale_r=command_scale_r,
            command_scale_h=command_scale_h,
            buffer_capacity=buffer_capacity,
            n_updates_per_iter=n_updates_per_iter,
        )
