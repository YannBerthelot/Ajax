from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.APO.state import APOConfig
from ajax.agents.APO.train_APO import make_train
from ajax.agents.base import ActorCritic
from ajax.agents.cloning import CloningConfig
from ajax.agents.loop import LOOP_PHASES
from ajax.extensions.base import Extension
from ajax.modules.pid_actor import PIDActorConfig
from ajax.types import EnvType, InitializationFunction


class APO(ActorCritic):
    """
    Average-Policy Optimization (APO, Ma et al. 2021) agent for training and testing in continuous action spaces.
    See  https://arxiv.org/pdf/2106.03442
    """

    name: str = "APO"
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
    }

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        n_envs: int = 4,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_architecture=("128", "tanh", "128", "tanh"),
        critic_architecture=("128", "tanh", "128", "tanh"),
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 0.5,
        ent_coef: float = 0,
        clip_range: float = 0.2,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        num_minibatches: int = 0,
        adam_eps: float = 1e-8,
        gae_lambda: float = 0.95,
        alpha: float = 0.1,
        nu: float = 0.1,
        normalize_advantage: bool = True,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        actor_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
        critic_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_bias_init: Optional[Union[str, InitializationFunction]] = None,
        actor_cloning_epochs: int = 10,
        actor_cloning_lr: float = 1e-3,
        actor_cloning_batch_size: int = 64,
        pre_train_n_steps: int = 0,
        # Expert for the cloning pre-training and the eval expert-bias
        # metric; online BC is the ImitationLoss extension.
        expert_policy: Optional[Callable] = None,
        pid_actor_config: Optional[PIDActorConfig] = None,
        # Gap A (Phase 4a): expose the most recent ``(T, n_envs, ...)``
        # rollout transition on ``agent_state.last_rollout``. Off by
        # default — see :attr:`BaseAgentState.last_rollout`.
        expose_recent_rollout: bool = False,
        # --- New surface: composable research features as Extensions ---
        extensions: Sequence[Extension] = (),
    ) -> None:
        """
        Initialize the APO agent.

        Args:
            env_id (str | EnvType): Environment ID or environment instance.
            n_envs (int): Number of parallel environments.
            learning_rate (float): Learning rate for optimizers.
            actor_architecture (tuple): Architecture of the actor network.
            critic_architecture (tuple): Architecture of the critic network.
            gamma (float): Discount factor for rewards.
            env_params (Optional[EnvParams]): Parameters for the environment.
            max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Batch size for training.
            learning_starts (int): Timesteps before training starts.
            tau (float): Soft update coefficient for target networks.
            reward_scale (float): Scaling factor for rewards.
            alpha_init (float): Initial value for the temperature parameter.
            target_entropy_per_dim (float): Target entropy per action dimension.
        """
        self.config = {**locals()}
        self.config.update({"algo_name": "APO"})

        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            actor_kernel_init=actor_kernel_init,
            actor_bias_init=actor_bias_init,
            critic_kernel_init=critic_kernel_init,
            critic_bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            encoder_bias_init=encoder_bias_init,
            # Continuous APO on bounded control envs (brax, playground)
            # saturates an unbounded Gaussian at the action clip, as PPO
            # does: the squashed policy stays in [-1, 1] with a correct
            # log-prob Jacobian (discrete APO ignores it). Brax's actor head:
            # one learnable log-std per action dimension, at std 1 (about
            # 3.5 times the exploration of Ajax's state-dependent log-std
            # at std 0.37); a lecun_uniform mean head, whose moderate
            # initial actions keep evaluation close to training (orthogonal
            # 0.01 makes the deterministic action essentially zero); no
            # LayerNorm at the encoder's output, which brax's MLP lacks.
            squash=True,
            log_std_state_independent=True,
            log_std_init=0.0,
            mean_kernel_init="lecun_uniform",
            disable_encoder_output_norm=True,
            extensions=extensions,
        )

        self.agent_config = APOConfig(
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
            num_minibatches=num_minibatches,
            alpha=alpha,
            nu=nu,
            expose_recent_rollout=expose_recent_rollout,
        )

        # Adam's eps: brax's 1e-8 by default, not ActorCritic's 1e-5.
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=adam_eps)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=adam_eps)
        self.cloning_config = CloningConfig(
            actor_epochs=actor_cloning_epochs,
            actor_lr=actor_cloning_lr,
            actor_batch_size=actor_cloning_batch_size,
            pre_train_n_steps=pre_train_n_steps,
        )
        self.expert_policy = expert_policy
        self.pid_actor_config = pid_actor_config

    def get_make_train(self) -> Callable:
        """
        Create a training function for the APO agent.

        Returns:
            Callable: A function that trains the APO agent.
        """
        return partial(
            make_train,
            cloning_args=self.cloning_config,
            expert_policy=self.expert_policy,
            pid_actor_config=self.pid_actor_config,
            extensions=tuple(self.extension_stack.extensions),
        )
