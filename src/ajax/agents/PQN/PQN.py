from collections.abc import Sequence
from typing import Callable, Optional, Tuple

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.PQN.state import PQNConfig
from ajax.agents.PQN.train_PQN import make_train
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import Extension
from ajax.networks.networks import CNNSpec, reference_cnn
from ajax.types import EnvType


class PQN(ActorCritic):
    """Parallelised Q-Network (Gallici et al., 2024) for discrete action spaces.

    On-policy value-based RL: vectorised rollouts, Q(lambda) regression
    targets, no replay buffer and no target network -- training stability
    instead comes from LayerNorm in the Q-network. The policy is greedy
    over Q, with epsilon-greedy exploration during collection.
    """

    name: str = "PQN"
    capabilities: frozenset = frozenset(
        "discounted obs_networks off_policy q_critic".split()
    )
    supported_extension_phases: frozenset = LOOP_PHASES | {"on_target", "critic_loss"}
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 8,
        learning_rate: float = 2.5e-4,
        architecture=("128", "relu", "128", "relu"),
        gamma: float = 0.99,
        q_lambda: float = 0.65,
        n_steps: int = 128,
        n_epochs: int = 4,
        num_minibatches: int = 4,
        reward_scale: float = 1.0,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        # Linear epsilon-greedy schedule: epsilon goes from epsilon_start
        # to epsilon_end over the first epsilon_decay_frac of training.
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_frac: float = 0.5,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        # TD loss: None -> MSE. Pass make_huber_td_loss(delta) for Huber.
        td_loss_fn: Optional[Callable] = None,
        # CNN encoder for image observations -- see NetworkConfig.cnn_image_shape.
        cnn_image_shape: Optional[Tuple[int, int, int]] = None,
        cnn_extra_obs_dim: int = 0,
        cnn_spec: Optional[CNNSpec] = None,
        extensions: Sequence[Extension] = (),
        # Expose the most recent ``(T, n_envs, ...)``
        # rollout transition on ``agent_state.last_rollout``. Off by
        # default — see :attr:`BaseAgentState.last_rollout`.
        expose_recent_rollout: bool = False,
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        self.config = run_config(locals(), self.name)
        # The image network of PQN's reference, purejaxql: Mnih et al.'s, or
        # MinAtar's for images too small for it (reference_cnn), each layer
        # he_normal and followed by a LayerNorm.
        if cnn_image_shape is not None and cnn_spec is None:
            cnn_spec = reference_cnn(cnn_image_shape)._replace(
                kernel_init="he_normal", layer_norm=True
            )

        # PQN has a single Q-network; forward `architecture`/`learning_rate`
        # to both actor/critic slots of the base class.
        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=learning_rate,
            critic_learning_rate=learning_rate,
            actor_architecture=architecture,
            critic_architecture=architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            reward_normalization_gamma=reward_normalization_gamma,
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=cnn_extra_obs_dim,
            cnn_spec=cnn_spec,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )

        if check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("PQN only supports discrete action spaces.")
        if n_steps * self.env_args.n_envs % num_minibatches != 0:
            raise ValueError(
                f"the rollout's n_steps * n_envs ({n_steps} * {self.env_args.n_envs})"
                f" rows must be divisible by num_minibatches ({num_minibatches})."
            )

        self.agent_config: PQNConfig = PQNConfig(
            gamma=gamma,
            q_lambda=q_lambda,
            n_steps=n_steps,
            n_epochs=n_epochs,
            num_minibatches=num_minibatches,
            reward_scale=reward_scale,
            expose_recent_rollout=expose_recent_rollout,
        )
        self.train_kwargs = {
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_frac": epsilon_decay_frac,
            "td_loss_fn": td_loss_fn,
        }
