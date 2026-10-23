from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES, REPLAY_BATCH_KEYS
from ajax.agents.recurrent import make_replay_buffer
from ajax.agents.TQC.state import TQCConfig
from ajax.agents.TQC.train_TQC import make_train
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig
from ajax.networks.utils import TORCH_LINEAR
from ajax.state import AlphaConfig
from ajax.types import EnvType


class TQC(ActorCritic):
    """Truncated Quantile Critics (Kuznetsov, Shvechikov, Grishin and Vetrov,
    ICML 2020) for continuous action spaces: SAC with distributional
    critics whose largest pooled target quantiles are dropped. The defaults
    are the authors' code's (bayesgroup/tqc_pytorch, Software Heritage
    swh:1:rev:65114eba5d868e9f0355e33b96acab7530191f16, ``main.py`` and
    ``tqc/``)."""

    name: str = "TQC"
    capabilities: frozenset = frozenset(
        "actor critic_ensemble discounted distributional_critic obs_networks "
        "off_policy policy_density q_critic replay target_critic "
        "reward_learning".split()
    )
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
        "transform_action",
    }
    phase_batch_keys: Mapping[str, frozenset] = REPLAY_BATCH_KEYS
    pretrain_context: frozenset = frozenset("buffer env_args update".split())
    extension_context: frozenset = frozenset(
        "actor_optimizer_args buffer env_args expert_policy".split()
    )
    supports_memory: bool = True
    replays_transitions: bool = True
    stores_expert_actions: bool = True
    supported_buffer_fields: frozenset = frozenset({"raw_next_obs", "return_to_go"})
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        # trainer.py:25-27: one Adam at 3e-4 for each of the actor, the
        # critics and the temperature.
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        # structures.py:94 (critics, each 3 x 512) and :108 (actor, 2 x 256).
        actor_architecture=("256", "relu", "256", "relu"),
        critic_architecture=("512", "relu", "512", "relu", "512", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        # main.py:65 trains from t >= batch_size, 0-indexed: after the 257th
        # step; its policy acts from the first (no uniform warm-up).
        learning_starts: int = 257,
        tau: float = 0.005,
        reward_scale: float = 1.0,
        alpha_init: float = 1.0,  # trainer.py:22, log_alpha from 0
        target_entropy_per_dim: float = -1.0,  # main.py:45
        num_critics: int = 5,  # main.py:94, n_nets
        n_quantiles: int = 25,  # main.py:92
        top_quantiles_to_drop_per_net: int = 2,  # main.py:93
        # Pluggable memory block (see ajax.networks.memory); trains on
        # replayed sequences with R2D2-style burn-in (see ajax.agents.recurrent).
        memory: Optional[Union[MemoryConfig, dict]] = None,
        burn_in: int = 8,
        sequence_length: int = 16,
        stored_state: bool = False,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        # Expert for the CloningPretrain extension and the eval expert-bias
        # metric; its actions, from its state along each trajectory, are
        # stored for the extensions that read them.
        expert_policy: Optional[Callable] = None,
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        """
        Initialize the TQC agent.

        Args:
            num_critics (int): Distributional critics (``n_nets``).
            n_quantiles (int): Quantiles each critic predicts.
            top_quantiles_to_drop_per_net (int): Pooled target quantiles
                dropped per critic. The paper tunes it per task (Appendix B,
                Table 6: 5 on Hopper, 0 on HalfCheetah, 2 on Walker2d, Ant
                and Humanoid).
        """
        self.config = run_config(locals(), self.name)
        n_drop = top_quantiles_to_drop_per_net * num_critics
        if n_quantiles < 1 or not 0 <= n_drop < num_critics * n_quantiles:
            raise ValueError(
                "TQC keeps some of its num_critics * n_quantiles pooled target"
                " quantiles: need n_quantiles >= 1 and 0 <="
                " top_quantiles_to_drop_per_net < n_quantiles, got"
                f" n_quantiles={n_quantiles},"
                f" top_quantiles_to_drop_per_net={top_quantiles_to_drop_per_net}."
            )

        # Read by the extension check in the base initialiser.
        self.expert_policy = expert_policy
        # The authors' layers are torch's nn.Linear at its default
        # initialisation, biases included (structures.py:36-58, "TODO:
        # initialization"). Recurrent TQC (memory set) keeps Ajax's, as
        # recurrent SAC, REDQ and ASAC do until Ni et al.'s is audited.
        reference_init: dict[str, Any] = (
            {}
            if memory is not None
            else {
                "encoder_kernel_init": TORCH_LINEAR,
                "mean_kernel_init": TORCH_LINEAR,
                "log_std_kernel_init": TORCH_LINEAR,
                "log_std_init": None,
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
        # Torch's Adam eps, 1e-8 (trainer.py:25-27, torch 1.3.0).
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=1e-8)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=1e-8)
        # The critics' quantile heads; the tanh log-det is exact, as the
        # authors' (structures.py:142), Squash's default.
        self.network_args = self.network_args.replace(critic_quantiles=n_quantiles)

        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate,
            alpha_init=alpha_init,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("TQC only supports continuous action spaces.")
        action_dim = get_action_dim(self.env_args.env, env_params)
        self.agent_config: TQCConfig = TQCConfig(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy_per_dim * action_dim,
            reward_scale=reward_scale,
            num_critics=num_critics,
            n_quantiles=n_quantiles,
            top_quantiles_to_drop_per_net=top_quantiles_to_drop_per_net,
            burn_in=burn_in,
            sequence_length=sequence_length,
            stored_state=stored_state,
        )
        self.buffer = make_replay_buffer(
            self.agent_config, n_envs, self.network_args.memory, buffer_size, batch_size
        )

        self.train_kwargs = {
            "buffer": self.buffer,
            "alpha_args": self.alpha_args,
            "expert_policy": self.expert_policy,
        }
