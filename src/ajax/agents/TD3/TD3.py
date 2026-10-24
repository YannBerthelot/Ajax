from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES, REPLAY_BATCH_KEYS, TARGET_KEYS
from ajax.agents.recurrent import make_replay_buffer
from ajax.agents.TD3.state import TD3Config
from ajax.agents.TD3.train_TD3 import make_train
from ajax.environments.interaction import with_acting_state
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig
from ajax.networks.utils import TORCH_LINEAR
from ajax.schedule import Schedule
from ajax.types import EnvType


class TD3(ActorCritic):
    """Twin Delayed DDPG (Fujimoto et al., 2018) for continuous action spaces.

    Defaults are the authors' code as updated in 2019 (sfujim/TD3 a8d53f7,
    ade6260; its README: "minor adjustments to hyperparameters ... to
    improve performance"), not the paper's 400-300, lr 1e-3, batch 100 and
    1e4 warm-up:
      - Actor / critic: 256-256 ReLU, plain MLPs (no normalisation layer)
      - Adam, lr=3e-4
      - tau=0.005, gamma=0.99
      - Exploration noise: N(0, 0.1)
      - Target policy noise: N(0, 0.2), clipped to +/-0.5
      - Policy delay: 2
      - Replay buffer: 1e6, batch=256, learning_starts=25e3
    """

    name: str = "TD3"
    capabilities: frozenset = frozenset(
        "actor critic_ensemble discounted obs_networks "
        "off_policy q_critic replay target_critic reward_learning".split()
    )
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
        "transform_action",
        "actuate",
    }
    # Its smoothed next actions join the shared step's on_target keys.
    phase_batch_keys: Mapping[str, frozenset] = REPLAY_BATCH_KEYS | {
        "on_target": TARGET_KEYS | {"next_actions"}
    }
    pretrain_context: frozenset = frozenset("buffer env_args update".split())
    extension_context: frozenset = frozenset(
        "actor_optimizer_args buffer env_args expert_policy".split()
    )
    supports_memory: bool = True
    replays_transitions: bool = True
    stores_expert_actions: bool = True
    supported_buffer_fields: frozenset = frozenset(
        {"next_action", "raw_next_obs", "return_to_go"}
    )
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_architecture=("256", "relu", "256", "relu"),
        critic_architecture=("256", "relu", "256", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = int(25e3),
        tau: float = 0.005,
        reward_scale: float = 1.0,
        num_critics: int = 2,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise: Schedule = 0.1,  # or a function of the timestep
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
        # metric; online BC is the ImitationLoss extension. Its actions,
        # from its state along each trajectory, are stored for the
        # extensions that read them.
        expert_policy: Optional[Callable] = None,
        # Collection-time action selection override (used downstream).
        action_pipeline: Optional[Callable] = None,
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        self.config = run_config(locals(), self.name)

        # Read by the extension check in the base initialiser.
        self.expert_policy = expert_policy
        # The initialisation of TD3's reference code: sfujim/TD3 builds
        # its actor and critics from torch nn.Linear layers it never
        # re-initialises, at the camera-ready commit bc98e62 (TD3.py:24-26,
        # :43-50) as at 34770cc (TD3.py:18-20, :36-43). torch's default
        # draws kernel and bias uniform in +-1/sqrt(fan_in) (v0.4.0's
        # torch/nn/modules/linear.py:48-52): "torch_linear". The actor's
        # last layer included: the uniform(+-3e-3) of DDPG's paper is in
        # neither.
        # Recurrent TD3 (memory set) keeps Ajax's previous initialisation
        # until Ni et al.'s, its recurrent reference (pomdp-baselines), is
        # audited, as SAC, REDQ and ASAC do.
        reference_init: dict[str, Any] = (
            {}
            if memory is not None
            else {
                "encoder_kernel_init": TORCH_LINEAR,
                "mean_kernel_init": TORCH_LINEAR,
                "value_kernel_init": TORCH_LINEAR,
            }
        )

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
            **reference_init,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        # Adam's eps is torch's default, 1e-8, which the reference's Adams
        # keep (sfujim/TD3 @34770cc TD3.py:83, :87, PyTorch 1.2); recurrent
        # TD3's reference (Ni et al., pomdp-baselines e7c19c3
        # policy_rnn.py:75, :93) keeps it too. Ajax's shared default is 1e-5.
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=1e-8)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=1e-8)

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
            "action_pipeline": with_acting_state(action_pipeline),
        }
