from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional

from gymnax import EnvParams

from ajax.agents.AVG.state import AVGConfig
from ajax.agents.AVG.train_AVG import make_train
from ajax.agents.base import ActorCritic
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.extensions.base import Extension
from ajax.modules.pid_actor import PIDActorConfig
from ajax.state import AlphaConfig
from ajax.types import EnvType


class AVG(ActorCritic):
    """Action Value Gradient (AVG) from Vasan et al. 2024. See https://arxiv.org/abs/2411.15370

    AVG has no memory option (``supports_memory`` stays False): its
    fully-incremental single-transition updates give length-1 BPTT, so
    memory weights could not learn temporal structure without eligibility
    traces / RTRL.
    """

    name: str = "AVG"

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 6.3e-3,
        critic_learning_rate: float = 8.7e-3,
        alpha_learning_rate: float = 3e-4,
        actor_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        critic_architecture=("256", "leaky_relu", "256", "leaky_relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        learning_starts: int = 0,
        reward_scale: float = 1.0,
        alpha_init: float = 0.07,
        target_entropy_per_dim: float = -1.0,
        beta_1: float = 0,
        beta_2: float = 0.999,
        num_critics: int = 1,
        # Expert for the eval expert-bias metric only.
        expert_policy: Optional[Callable] = None,
        pid_actor_config: Optional[PIDActorConfig] = None,
        # Gap A (Phase 4a): expose the most recent ``(T=1, n_envs, ...)``
        # rollout transition on ``agent_state.last_rollout``. Off by
        # default — see :attr:`BaseAgentState.last_rollout`.
        expose_recent_rollout: bool = False,
        # --- New surface: composable research features as Extensions ---
        extensions: Sequence[Extension] = (),
    ) -> None:
        """
        Initialize the AVG agent.

        Args:
            env_id (str | EnvType): Environment ID or environment instance.
            n_envs (int): Number of parallel environments.
            actor_learning_rate, critic_learning_rate (float): Adam step sizes.
            actor_architecture, critic_architecture (tuple): Network layers.
            gamma (float): Discount factor for rewards.
            env_params (Optional[EnvParams]): Parameters for the environment.
            max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
            learning_starts (int): Timesteps before training starts.
            reward_scale (float): Scaling factor for rewards.
            alpha_init (float): The (fixed) entropy coefficient.
            target_entropy_per_dim (float): Target entropy per action dimension.
            beta_1, beta_2 (float): Adam's moment decays (AVG uses beta_1 = 0).
        """
        self.config = {**locals()}
        self.config.update({"algo_name": "AVG"})

        # AVG normalises its observations (never its rewards) and squashes
        # its actions.
        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            normalize_observations=True,
            squash=True,
            extensions=extensions,
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("AVG only supports continuous action spaces.")

        # AVG's penultimate normalisation and Adam betas.
        self.network_args = self.network_args.replace(penultimate_normalization=True)
        self.actor_optimizer_args = self.actor_optimizer_args.replace(
            beta_1=beta_1, beta_2=beta_2
        )
        self.critic_optimizer_args = self.critic_optimizer_args.replace(
            beta_1=beta_1, beta_2=beta_2
        )
        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate,
            alpha_init=alpha_init,
        )
        action_dim = get_action_dim(self.env_args.env, self.env_args.env_params)
        self.agent_config = AVGConfig(
            gamma=gamma,
            learning_starts=learning_starts,
            target_entropy=target_entropy_per_dim * action_dim,
            reward_scale=reward_scale,
            num_critics=num_critics,
            expose_recent_rollout=expose_recent_rollout,
        )
        self.expert_policy = expert_policy
        self.pid_actor_config = pid_actor_config

    def get_make_train(self) -> Callable:
        return partial(
            make_train,
            alpha_args=self.alpha_args,
            expert_policy=self.expert_policy,
            pid_actor_config=self.pid_actor_config,
            extensions=tuple(self.extension_stack.extensions),
        )
