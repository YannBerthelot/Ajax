from collections.abc import Sequence
from typing import Callable, Optional, Tuple, Union

from gymnax import EnvParams

from ajax.agents.APO.state import APOConfig
from ajax.agents.APO.train_APO import make_train
from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import CNNSpec
from ajax.types import EnvType, InitializationFunction


class APO(ActorCritic):
    """
    Average-Policy Optimization (APO, Ma et al. 2021) for discrete and
    continuous action spaces.
    See  https://arxiv.org/pdf/2106.03442
    """

    name: str = "APO"
    capabilities: frozenset = frozenset("actor obs_networks v_critic".split())
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
    }
    extension_context: frozenset = frozenset(
        "actor_optimizer_args env_args expert_policy".split()
    )
    stores_expert_actions: bool = True
    supports_memory: bool = True
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 4,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_architecture=("128", "tanh", "128", "tanh"),
        critic_architecture=("128", "tanh", "128", "tanh"),
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 0.5,
        weight_decay: float = 0.0,
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
        reward_normalization_gamma: Optional[float] = None,
        actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
        critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        # Expert for the CloningPretrain extension and the eval expert-bias
        # metric; online BC is the ImitationLoss extension. Its actions,
        # from its state along each trajectory, are stored for the
        # extensions that read them.
        expert_policy: Optional[Callable] = None,
        # Expose the most recent ``(T, n_envs, ...)``
        # rollout transition on ``agent_state.last_rollout``. Off by
        # default — see :attr:`BaseAgentState.last_rollout`.
        expose_recent_rollout: bool = False,
        # Pluggable memory block, e.g. MemoryConfig("gru", 64): the actor
        # and the critic replay each env's whole rollout from the carries
        # it started with, as PPO's.
        memory: Optional[Union[MemoryConfig, dict]] = None,
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
            actor_bias_init=actor_bias_init,
            critic_bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            # Continuous APO on bounded control envs (brax, playground)
            # saturates an unbounded Gaussian at the action clip, as PPO
            # does: the squashed policy stays in [-1, 1] with a correct
            # log-prob Jacobian (discrete APO ignores it). The squash and the
            # lecun_uniform mean head are brax's; its moderate initial
            # actions keep evaluation close to training (orthogonal 0.01
            # makes the deterministic action essentially zero). APO's
            # reference code instead bounds the mean with a tanh and samples
            # an unsquashed Gaussian. The log-std is the reference's: one
            # learnable parameter per action dimension starting at 0 (std
            # 1), xtma/apo @bfcd997 on the authors' xtma/rlpyt @e28bb20
            # (rlpyt/models/pg/mujoco_ff_model.py:24,50,78). The reference
            # never clips it (rlpyt/agents/pg/gaussian.py:34-38); Ajax clips
            # it to [-20, 2] (clip_log_std_param, left at its default). Like
            # brax's MLP, the encoder has no output LayerNorm (the default).
            squash=True,
            log_std_state_independent=True,
            log_std_init=0.0,
            mean_kernel_init="lecun_uniform",
            episode_length=episode_length,
            action_repeat=action_repeat,
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=cnn_extra_obs_dim,
            cnn_spec=cnn_spec,
            extensions=extensions,
        )

        self.agent_config: APOConfig = APOConfig(
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
        if num_minibatches <= 0 and max(n_steps, batch_size) % min(n_steps, batch_size):
            raise ValueError(
                f"n_steps ({n_steps}) and batch_size ({batch_size}) must divide"
                " one another: the rollout splits into max // min minibatches"
                " (or set num_minibatches)."
            )

        # Adam's eps: brax's 1e-8 by default, not ActorCritic's 1e-5.
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=adam_eps)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=adam_eps)
        self.train_kwargs = {
            "expert_policy": self.expert_policy,
        }
