from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.ASAC.state import ASACConfig
from ajax.agents.ASAC.train_ASAC import make_train
from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.recurrent import make_replay_buffer
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig
from ajax.networks.utils import TORCH_LINEAR
from ajax.state import AlphaConfig
from ajax.types import EnvType


class ASAC(ActorCritic):
    """Average-Reward Soft Actor-Critic (ASAC) from Adamczyk et al. 2025. See https://arxiv.org/abs/2501.09080v2"""

    name: str = "ASAC"
    capabilities: frozenset = frozenset(
        "actor critic_ensemble obs_networks off_policy policy_density q_critic"
        " replay".split()
    )
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
    }
    pretrain_context: frozenset = frozenset("buffer env_args update".split())
    supports_memory: bool = True
    replays_transitions: bool = True
    stores_expert_actions: bool = True
    supported_buffer_fields: frozenset = frozenset({"raw_next_obs"})
    extension_context: frozenset = frozenset(
        "actor_optimizer_args buffer env_args expert_policy".split()
    )
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 5e-4,
        alpha_learning_rate: float = 1e-4,
        actor_architecture=("256", "relu", "256", "relu"),
        critic_architecture=("256", "relu", "256", "relu"),
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 10.0,
        weight_decay: float = 0.0,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = int(1e4),
        tau: float = 0.005,
        tau_theta: float = 0.005,
        reward_scale: float = 1.0,
        alpha_init: float = 0.2,
        fixed_alpha: bool = True,
        p_0=20,
        target_entropy_per_dim: float = -1.0,
        # Pluggable memory block, e.g. MemoryConfig("lstm", 64) or
        # {"kind": "gru", "hidden_size": 64}. When set, the replay buffer
        # stores per-env trajectories and updates train on sequences of
        # `sequence_length` steps after `burn_in` warm-up steps whose
        # carries are computed from zero under stop_gradient (R2D2-style).
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
        extensions: Sequence[Extension] = (),
        # Simulator steps per episode and per agent step on brax / playground
        # envs (ActorCritic).
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
    ) -> None:
        """
        Initialize the ASAC agent.

        The learning rates, the gradient clip (each critic and the actor on
        its own; the temperature unclipped) and ``tau_theta`` are the
        authors' (BobMak/asac @4b68862: every hparams/*/asac.yaml sets
        learning_rate 5e-4 for the critics, actor_learning_rate 1e-4 and
        max_grad_norm 10; darer/ASAC.py:87, :92, :104 for the temperature's
        1e-4, :228-229 and :249-250 for the clips).

        Args:
            tau_theta (float): Rate of the ``theta`` and penalty averages
                (darer/ASAC.py:188, :213). The authors tune it per
                environment, 0.0005 to 0.1; 0.005 is their default
                (darer/BaseAgent.py:80) and six of their 11 configs'.
            alpha_init (float): The temperature, or its initial value when
                ``fixed_alpha`` is False.
            fixed_alpha (bool): Keep the temperature at ``alpha_init``. The
                authors' published runs fix it at 1/beta: 0.2, and 0.05 for
                Humanoid and Swimmer (BobMak/asac @4b68862,
                hparams/*/asac.yaml, darer/ASAC.py:56-57,
                experiments/finetuned_runs.py:67-76). False learns it
                towards the target entropy, as SAC does.
            target_entropy_per_dim (float): Target entropy per action dimension.
        """
        self.config = run_config(locals(), self.name)

        # Read by the extension check in the base initialiser.
        self.expert_policy = expert_policy
        # The initialisation of ASAC's reference code: BobMak/asac
        # @4b68862 builds its actor from stable-baselines3 2.4.0's SAC
        # Actor (darer/ASAC.py:15, :78-82; sac/policies.py:85, :102-103
        # at v2.4.0) and its critics from Qsa (darer/Models.py:409-411),
        # all torch nn.Linear layers at torch's default initialisation:
        # "torch_linear", whose layers draw their biases too.
        # Recurrent ASAC (memory set) keeps Ajax's previous initialisation
        # until Ni et al.'s, its recurrent reference (pomdp-baselines), is
        # audited, as SAC and REDQ do.
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
        # Adam's eps is torch's default, 1e-8, which the reference's Adams
        # keep (BobMak/asac @4b68862 darer/ASAC.py:87-92, torch 2.8.0);
        # recurrent ASAC's reference (Ni et al., pomdp-baselines e7c19c3
        # policy_rnn.py:75, :93) keeps it too. Ajax's shared default is 1e-5.
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=1e-8)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=1e-8)

        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate,
            alpha_init=alpha_init,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("ASAC only supports continuous action spaces.")
        action_dim = get_action_dim(self.env_args.env, env_params)
        target_entropy = target_entropy_per_dim * action_dim
        self.agent_config: ASACConfig = ASACConfig(
            tau=tau,
            tau_theta=tau_theta,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
            reward_scale=reward_scale,
            p_0=p_0,
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
            "fixed_alpha": fixed_alpha,
        }
