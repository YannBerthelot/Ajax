from collections.abc import Sequence
from typing import Callable, Optional, Tuple, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.recurrent import make_replay_buffer
from ajax.agents.TD3.state import TD3Config
from ajax.agents.TD3.train_TD3 import make_train
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import CNNSpec
from ajax.types import EnvType


class TD3(ActorCritic):
    """Twin Delayed DDPG (Fujimoto et al., 2018) for continuous action spaces.

    Defaults match the original paper (Table 1 / Algorithm 1):
      - Actor / critic: 400-300 ReLU, plain MLPs (no normalisation layer)
      - Adam, lr=1e-3
      - tau=0.005, gamma=0.99
      - Exploration noise: N(0, 0.1)
      - Target policy noise: N(0, 0.2), clipped to +/-0.5
      - Policy delay: 2
      - Replay buffer: 1e6, batch=100, learning_starts=1e4
    """

    name: str = "TD3"
    capabilities: frozenset = frozenset(
        "actor critic_ensemble discounted "
        "obs_networks off_policy q_critic replay".split()
    )
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
    }
    pretrain_context: frozenset = frozenset("buffer env_args update".split())
    extension_context: frozenset = frozenset(
        "actor_optimizer_args env_args expert_policy".split()
    )
    supports_memory: bool = True
    replays_transitions: bool = True
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 1e-3,
        critic_learning_rate: float = 1e-3,
        actor_architecture=("400", "relu", "300", "relu"),
        critic_architecture=("400", "relu", "300", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        buffer_size: int = int(1e6),
        batch_size: int = 100,
        learning_starts: int = int(1e4),
        tau: float = 0.005,
        reward_scale: float = 1.0,
        num_critics: int = 2,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: float = 0.1,
        # Pluggable memory block (see ajax.networks.memory); trains on
        # replayed sequences with R2D2-style burn-in (see ajax.agents.recurrent).
        memory: Optional[Union[MemoryConfig, dict]] = None,
        burn_in: int = 8,
        sequence_length: int = 16,
        # R2D2 stored-state replay: initialize replayed sequences from the
        # actor carries recorded at collection time instead of zero+burn-in
        # (Kapturowski et al. 2019 show this mitigates state staleness best).
        # Stores flatten_carry(actor carry) per step in the buffer.
        stored_state: bool = False,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        # Expert for the CloningPretrain extension and the eval expert-bias
        # metric; online BC is the ImitationLoss extension.
        expert_policy: Optional[Callable] = None,
        # Collection-time action selection override (used downstream).
        action_pipeline: Optional[Callable] = None,
        # CNN encoder for image observations -- see NetworkConfig.cnn_image_shape.
        cnn_image_shape: Optional[Tuple[int, int, int]] = None,
        cnn_extra_obs_dim: int = 0,
        cnn_spec: Optional[CNNSpec] = None,
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        self.config = run_config(locals(), self.name)

        # Read by the extension check in the base initialiser.
        self.expert_policy = expert_policy
        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            memory=memory,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            reward_normalization_gamma=reward_normalization_gamma,
            squash=True,
            episode_length=episode_length,
            action_repeat=action_repeat,
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=cnn_extra_obs_dim,
            cnn_spec=cnn_spec,
            extensions=extensions,
        )

        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("TD3 only supports continuous action spaces.")

        self.agent_config: TD3Config = TD3Config(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            reward_scale=reward_scale,
            num_critics=num_critics,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            exploration_noise=exploration_noise,
            burn_in=burn_in,
            sequence_length=sequence_length,
            stored_state=stored_state,
        )
        self.buffer = make_replay_buffer(
            self.agent_config, n_envs, self.network_args.memory, buffer_size, batch_size
        )
        self.train_kwargs = {
            "buffer": self.buffer,
            "expert_policy": self.expert_policy,
            "action_pipeline": action_pipeline,
        }
