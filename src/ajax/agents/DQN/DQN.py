from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Tuple

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.DQN.state import DQNConfig
from ajax.agents.DQN.train_DQN import make_train
from ajax.agents.loop import LOOP_PHASES
from ajax.buffers.utils import get_buffer
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import Extension
from ajax.networks.networks import CNNSpec, reference_cnn
from ajax.networks.utils import TORCH_LINEAR
from ajax.types import EnvType


class DQN(ActorCritic):
    """Deep Q-Network (Mnih et al., 2015) for discrete action spaces.

    Value-based, off-policy. A single Q-network is trained by TD against a
    periodically refreshed target network; the policy is greedy over Q,
    with epsilon-greedy exploration during collection.
    """

    name: str = "DQN"
    capabilities: frozenset = frozenset(
        {
            "discounted",
            "obs_networks",
            "off_policy",
            "q_critic",
            "replay",
            "target_critic",
            "reward_learning",
        }
    )
    supported_extension_phases: frozenset = LOOP_PHASES.union(
        {"on_target", "critic_loss"}
    )
    phase_batch_keys: Mapping[str, frozenset] = {
        "on_target": frozenset({"actions", "dones", "reward_scale"}),
        "critic_loss": frozenset({"actions", "q_state"}),
    }
    pretrain_context: frozenset = frozenset({"buffer", "env_args", "update"})
    replays_transitions: bool = True
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        learning_rate: float = 2.5e-4,
        architecture=("128", "relu", "128", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        buffer_size: int = int(1e5),
        batch_size: int = 64,
        learning_starts: int = 1000,
        tau: float = 1.0,
        target_update_interval: int = 500,
        reward_scale: float = 1.0,
        n_gradient_steps: int = 1,
        # Linear epsilon-greedy schedule: epsilon goes from epsilon_start
        # to epsilon_end over the first epsilon_decay_frac of training.
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_frac: float = 0.5,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        # TD target: None -> vanilla DQN. Pass compute_double_dqn_td_target
        # (from train_DQN) for Double DQN.
        td_target_fn: Optional[Callable] = None,
        # TD loss: None -> huber_1_td_loss (the reference's TD-error clip).
        # Pass mse_td_loss (from ajax.agents.critics) for the squared error.
        td_loss_fn: Optional[Callable] = None,
        # Q-network module: None -> QNetwork. Pass DuelingQNetwork
        # (from ajax.networks.q_networks) for the dueling architecture.
        q_network_cls: Optional[type] = None,
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
        # The image network of DQN's reference: Mnih et al.'s, or MinAtar's
        # for images too small for it (reference_cnn), at Torch7's and
        # torch's default initialisation, as the Q head and MLP encoder.
        if cnn_image_shape is not None and cnn_spec is None:
            cnn_spec = reference_cnn(cnn_image_shape)._replace(kernel_init=TORCH_LINEAR)

        # DQN has a single Q-network; forward `architecture`/`learning_rate`
        # to both actor/critic slots of the base class. The Q-network is
        # built from the critic slots and stored in `actor_state`.
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
            # The initialisation of DQN's reference code: google-deepmind/dqn
            # @9d9b1d1 builds its network from Torch7 nn layers it never
            # re-initialises (dqn/convnet.lua:23-61), and Torch7's
            # nn.Linear draws kernel and bias uniform in +-1/sqrt(fan_in)
            # (torch/nn @8726825, Linear.lua:21-41): "torch_linear", for the
            # MLP encoder and the Q head.
            encoder_kernel_init=TORCH_LINEAR,
            value_kernel_init=TORCH_LINEAR,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )

        if check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("DQN only supports discrete action spaces.")
        # The optimiser of DQN's reference (google-deepmind/dqn @9d9b1d1,
        # NeuralQLearner.lua:265-277, lr 2.5e-4 in run_gpu:25): centered
        # RMSProp, both averages decaying at 0.95, eps 0.01 inside the
        # square root of a gradient summed over the batch (:254). Ajax's TD
        # loss averages over the batch, so the same step takes eps
        # 0.01 / batch_size**2.
        rmsprop = {"optimizer": "centered_rmsprop", "beta_2": 0.95}
        rmsprop["eps"] = 0.01 / batch_size**2
        self.actor_optimizer_args = self.actor_optimizer_args.replace(**rmsprop)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(**rmsprop)

        self.agent_config: DQNConfig = DQNConfig(
            gamma=gamma,
            tau=tau,
            target_update_interval=target_update_interval,
            learning_starts=learning_starts,
            reward_scale=reward_scale,
            n_gradient_steps=n_gradient_steps,
        )
        self.buffer = get_buffer(
            buffer_size=buffer_size, batch_size=batch_size, n_envs=n_envs
        )
        self.train_kwargs = {
            "buffer": self.buffer,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_frac": epsilon_decay_frac,
            "td_target_fn": td_target_fn,
            "td_loss_fn": td_loss_fn,
            "q_network_cls": q_network_cls,
        }
