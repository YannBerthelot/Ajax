from collections.abc import Sequence
from functools import partial
from typing import Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic
from ajax.agents.PPO.state import PPOConfig
from ajax.agents.PPO.train_PPO import make_train
from ajax.extensions.base import Extension
from ajax.logging.wandb_logging import (
    LoggingConfig,
)
from ajax.modules.pid_actor import PIDActorConfig
from ajax.networks.memory import MemoryConfig
from ajax.state import OptimizerConfig
from ajax.types import EnvType, InitializationFunction
from ajax.utils import get_and_prepare_hyperparams


class PPO(ActorCritic):
    """Soft Actor-Critic (PPO) agent for training and testing in continuous action spaces."""

    name: str = "PPO"
    supports_memory: bool = True

    def __init__(  # pylint: disable=W0102, R0913
        self,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        n_envs: int = 4,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_architecture=("128", "tanh", "128", "tanh"),
        critic_architecture=("128", "tanh", "128", "tanh"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = 0.5,
        ent_coef: float = 0,
        clip_range: float = 0.2,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        num_minibatches: int = 0,
        adam_eps: float = 1e-8,
        vf_coef: float = 1.0,
        use_vtrace_gae: bool = False,
        fused_grad_clip: bool = False,
        unroll_length: Optional[int] = None,
        num_resets_per_eval: int = 0,
        num_evals: int = 1,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        lstm_hidden_size: Optional[int] = None,
        # Pluggable memory block, e.g. MemoryConfig("gru", 64) or
        # {"kind": "lstm", "hidden_size": 64}. When set, PPO trains with
        # BPTT over sequences minibatched from the rollout (see
        # num_minibatches and bptt_length; batch_size is ignored).
        memory: Optional[Union[MemoryConfig, dict]] = None,
        # Recurrent-only truncated-BPTT length: splits each env's rollout
        # into n_steps/bptt_length contiguous sequences whose start
        # carries are recomputed chunk-wise (never zero mid-episode).
        # None = full-rollout BPTT. Must divide n_steps.
        bptt_length: Optional[int] = None,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        actor_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
        critic_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_bias_init: Optional[Union[str, InitializationFunction]] = None,
        # Brax-style actor head knobs (off by default = Ajax legacy).
        # Set per-env in EVAREST's manip dict for brax/playground envs.
        log_std_state_independent: bool = False,
        log_std_init: float = -1.0,
        mean_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        disable_encoder_output_norm: bool = False,
        squash: bool = False,
        episode_length: Optional[int] = None,
        pid_actor_config: Optional[PIDActorConfig] = None,
        action_pipeline: Optional[Callable] = None,
        eval_action_transform: Optional[Callable] = None,
        obs_preprocessor: Optional[Callable] = None,
        init_transform: Optional[Callable] = None,
        auxiliary_update: Optional[Callable] = None,
        extra_eval_metrics: Optional[Callable] = None,
        extra_actor_loss_fn: Optional[Callable] = None,
        extra_critic_loss_fn: Optional[Callable] = None,
        reward_shaping_fn: Optional[Callable] = None,
        # CNN encoder for image observations -- see NetworkConfig.cnn_image_shape.
        cnn_image_shape: Optional[tuple] = None,
        cnn_extra_obs_dim: int = 0,
        cnn_spec: Optional[tuple] = None,
        # --- New surface: composable research features as Extensions ---
        extensions: Sequence[Extension] = (),
        # Gap A (Phase 4a): expose the most recent ``(T, n_envs, ...)``
        # rollout transition on ``agent_state.last_rollout`` so
        # measurement extensions can read an on-state-visitation batch
        # without forcing a fresh rollout per eval. Off by default —
        # see :attr:`BaseAgentState.last_rollout`.
        expose_recent_rollout: bool = False,
    ) -> None:
        """
        Initialize the PPO agent.

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
            lstm_hidden_size (Optional[int]): Hidden size for LSTM (if used).
        """
        self.config = {**locals()}
        self.config.update({"algo_name": "PPO"})

        super().__init__(
            env_id=env_id,
            n_envs=n_envs,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            env_params=env_params,
            max_grad_norm=max_grad_norm,
            lstm_hidden_size=lstm_hidden_size,
            memory=memory,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            actor_kernel_init=actor_kernel_init,
            actor_bias_init=actor_bias_init,
            critic_kernel_init=critic_kernel_init,
            critic_bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            encoder_bias_init=encoder_bias_init,
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=cnn_extra_obs_dim,
            cnn_spec=cnn_spec,
            log_std_state_independent=log_std_state_independent,
            log_std_init=log_std_init,
            mean_kernel_init=mean_kernel_init,
            disable_encoder_output_norm=disable_encoder_output_norm,
            squash=squash,
            episode_length=episode_length,
            # Brax-faithful normalise-at-forward: env wrapper is told
            # NOT to apply normalisation to ``state.obs`` (transition
            # obs stays raw). PPO uses Ajax's existing AGENT-side
            # running normaliser (``collector_state.obs_norm_info``
            # synced into ``actor_state.obs_norm_info`` /
            # ``critic_state.obs_norm_info`` at every collect step) so
            # ``get_pi`` / ``predict_value`` apply ``apply_obs_norm``
            # consistently at both COLLECT and LOSS forward calls. The
            # env wrapper is left in place when normalize_observations
            # is True only so that ClipAction wraps the env -- the
            # actual normalisation is fully agent-side.
            apply_obs_normalization=not normalize_observations,
            extensions=extensions,
        )

        self.agent_config = PPOConfig(
            gamma=gamma,
            ent_coef=ent_coef,
            clip_range=clip_range,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
            num_minibatches=num_minibatches,
            vf_coef=vf_coef,
            use_vtrace_gae=use_vtrace_gae,
            fused_grad_clip=fused_grad_clip,
            unroll_length=unroll_length,
            bptt_length=bptt_length,
            num_resets_per_eval=num_resets_per_eval,
            num_evals=num_evals,
            expose_recent_rollout=expose_recent_rollout,
        )

        if bptt_length is not None:
            if self.network_args.memory is None:
                raise ValueError(
                    "bptt_length requires a memory config (recurrent PPO);"
                    " for feedforward fragment minibatching use unroll_length."
                )
            if n_steps % bptt_length != 0:
                raise ValueError(
                    f"bptt_length ({bptt_length}) must divide n_steps ({n_steps})."
                )

        # Override base ActorCritic's eps=1e-5 with PPO's brax-default eps=1e-8.
        self.actor_optimizer_args = OptimizerConfig(
            learning_rate=actor_learning_rate,
            max_grad_norm=max_grad_norm,
            clipped=max_grad_norm is not None,
            eps=adam_eps,
        )
        self.critic_optimizer_args = OptimizerConfig(
            learning_rate=critic_learning_rate,
            max_grad_norm=max_grad_norm,
            clipped=max_grad_norm is not None,
            eps=adam_eps,
        )
        self.pid_actor_config = pid_actor_config
        self.action_pipeline = action_pipeline
        self.eval_action_transform = eval_action_transform
        self.obs_preprocessor = obs_preprocessor
        self.init_transform = init_transform
        self.auxiliary_update = auxiliary_update
        self.extra_eval_metrics = extra_eval_metrics
        self.extra_actor_loss_fn = extra_actor_loss_fn
        self.extra_critic_loss_fn = extra_critic_loss_fn
        self.reward_shaping_fn = reward_shaping_fn
        # Brax-faithful normalise-at-forward: when the user opted into
        # ``normalize_observations``, route it through the AGENT-side
        # running stats (collector_state.obs_norm_info, applied inside
        # get_pi / predict_value via apply_obs_norm) instead of the env
        # wrapper. The wrapper still needs the flag plumbed (so we know
        # the obs in transition is raw, not pre-normalised by the env)
        # -- that's handled inside ``super().__init__`` by toggling
        # ``apply_obs_normalization=False`` below.
        self._normalize_obs_running = bool(normalize_observations)

    def get_make_train(self) -> Callable:
        """
        Create a training function for the PPO agent.

        Returns:
            Callable: A function that trains the PPO agent.
        """
        return partial(
            make_train,
            pid_actor_config=self.pid_actor_config,
            action_pipeline=self.action_pipeline,
            eval_action_transform=self.eval_action_transform,
            obs_preprocessor=self.obs_preprocessor,
            init_transform=self.init_transform,
            auxiliary_update=self.auxiliary_update,
            extra_eval_metrics=self.extra_eval_metrics,
            extra_actor_loss_fn=self.extra_actor_loss_fn,
            extra_critic_loss_fn=self.extra_critic_loss_fn,
            normalize_obs_running=self._normalize_obs_running,
            reward_shaping_fn=self.reward_shaping_fn,
            extensions=tuple(self.extension_stack.extensions),
        )


if __name__ == "__main__":
    n_seeds = 100
    log_frequency = 20_000
    use_wandb = True
    logging_config = LoggingConfig(
        project_name="mission_debug_PPO_Ant_3",
        run_name="PPO",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
            "faulty_boostrap": False,
        },
        log_frequency=log_frequency,
        horizon=10_000,
        use_tensorboard=True,
        use_wandb=use_wandb,
    )
    # env_id = "HalfCheetah-v4"
    # env_id = "CartPole-v1"
    env_id = "Ant-v4"
    init_hyperparams, train_hyperparams = get_and_prepare_hyperparams(
        "./hyperparams/ppo.yml", env_id=env_id
    )

    print(train_hyperparams)

    def process_brax_env_id(env_id: str) -> str:
        """Remove version from env_id for brax compatibility."""
        short_env_id = env_id.split("-")[0].lower()
        brax_envs = [
            "hopper",
            "halfcheetah",
            "hopper",
            "walker2d",
            "humanoid",
            "reacher",
            "swimmer",
        ]
        if short_env_id in brax_envs:
            return short_env_id
        return env_id

    env_id = process_brax_env_id(env_id)

    env_id = "CartPole-v1"
    # env_id = "Pendulum-v1"

    # env, env_params = gymnax.make(env_id)

    PPO_agent = PPO(
        env_id=env_id,
        # batch_size=256,
        # gamma=0.999,
        # clip_range=0.1,
        # # n_envs=8,
        # # n_steps=1024,
        # actor_learning_rate=3e-4,
        # critic_learning_rate=1e-3,
        # # **init_hyperparams,
        # normalize_observations=True,
        # normalize_rewards=True,
        # ent_coef=1e-7,
        # n_envs=1,
        # n_steps=512,
        # gae_lambda=0.8,
        # n_envs=1,
        # n_steps=8,
    )  # Remove version from env_id for brax compatibility
    PPO_agent.train(
        seed=list(range(n_seeds)),
        logging_config=logging_config,
        n_timesteps=int(1e6),
        # **train_hyperparams,
    )
