import warnings
from collections.abc import Sequence
from typing import Optional, Tuple, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.PPO.core import resolve_num_minibatches
from ajax.agents.PPO.state import PPOConfig
from ajax.agents.PPO.train_PPO import make_train, minibatch_geometry
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig
from ajax.networks.networks import CNNSpec
from ajax.types import EnvType, InitializationFunction


def _check_minibatch_geometry(config: PPOConfig, n_envs: int, recurrent: bool) -> None:
    """Refuse an ``unroll_length`` the update could not use (it would fall
    back to whole rollouts); warn when a recurrent rollout's sequences force
    one minibatch where more were asked for."""
    wanted = resolve_num_minibatches(config)
    geometry = minibatch_geometry(config, config.n_steps, n_envs, recurrent)
    unroll = config.unroll_length
    if unroll is not None and recurrent:
        raise ValueError(
            "unroll_length is the feedforward fragment length; a recurrent"
            " PPO splits its rollout with bptt_length."
        )
    if unroll is not None and (geometry.kind, geometry.length) != ("time", unroll):
        raise ValueError(
            f"unroll_length ({unroll}) must divide n_steps ({config.n_steps})"
            " into fragments (n_steps // unroll_length * n_envs of them) that"
            f" split evenly into num_minibatches ({wanted}) > 1 minibatches."
        )
    if recurrent and geometry.num_minibatches < wanted:
        sequences = config.n_steps // geometry.length * n_envs
        warnings.warn(
            f"{sequences} sequences of {geometry.length} steps do not split"
            f" into {wanted} minibatches: recurrent PPO trains on one.",
            stacklevel=3,
        )


class PPO(ActorCritic):
    """Proximal Policy Optimization (Schulman et al., 2017) for discrete and
    continuous action spaces."""

    name: str = "PPO"
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
    }
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
        gamma: float = 0.99,
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
        vf_coef: float = 1.0,
        fused_grad_clip: bool = False,
        unroll_length: Optional[int] = None,
        num_resets_per_eval: int = 0,
        num_evals: int = 1,
        gae_lambda: float = 0.95,
        normalize_advantage: bool = True,
        # Pluggable memory block, e.g. MemoryConfig("gru", 128) or
        # {"kind": "lstm", "hidden_size": 128}. When set, PPO trains with
        # BPTT over sequences minibatched from the rollout (see
        # num_minibatches and bptt_length).
        memory: Optional[Union[MemoryConfig, dict]] = None,
        bptt_length: Optional[int] = None,
        normalize_observations: bool = False,
        normalize_rewards: bool = False,
        reward_normalization_gamma: Optional[float] = None,
        actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
        critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        # The Gaussian's log-std: one learned, unclipped parameter per action
        # dimension starting at 0 (std 1), as PPO's reference code has it
        # (Actor in networks.py cites it). False and -1.0 build the
        # state-dependent head PPO had before (SAC's, clipped).
        log_std_state_independent: bool = True,
        log_std_init: float = 0.0,
        mean_kernel_init: Optional[Union[str, InitializationFunction]] = None,
        encoder_layer_norm: bool = False,
        # Squashed or not, the action is read in [-1, 1] and mapped onto the
        # env's bounds (agent_action_to_env), clipped there first.
        squash: bool = False,
        episode_length: Optional[int] = None,
        # CNN encoder for image observations -- see NetworkConfig.cnn_image_shape.
        cnn_image_shape: Optional[Tuple[int, int, int]] = None,
        cnn_extra_obs_dim: int = 0,
        cnn_spec: Optional[CNNSpec] = None,
        extensions: Sequence[Extension] = (),
        # Expose the most recent ``(T, n_envs, ...)``
        # rollout transition on ``agent_state.last_rollout`` so
        # measurement extensions can read an on-state-visitation batch
        # without forcing a fresh rollout per eval. Off by default —
        # see :attr:`BaseAgentState.last_rollout`.
        expose_recent_rollout: bool = False,
        # Simulator steps per agent step on brax / playground envs
        # (ActorCritic).
        action_repeat: int = 1,
    ) -> None:
        self.config = run_config(locals(), self.name)

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
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=cnn_extra_obs_dim,
            cnn_spec=cnn_spec,
            log_std_state_independent=log_std_state_independent,
            log_std_init=log_std_init,
            mean_kernel_init=mean_kernel_init,
            encoder_layer_norm=encoder_layer_norm,
            squash=squash,
            episode_length=episode_length,
            action_repeat=action_repeat,
            # The env's normaliser hands every observation normalised
            # with the statistics of its step, and the rollout keeps it
            # so: the loss sees what the policy acted on (the ratio is 1
            # at the first epoch), as in SB3's VecNormalize and CleanRL.
            extensions=extensions,
        )

        config = PPOConfig(
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
            fused_grad_clip=fused_grad_clip,
            unroll_length=unroll_length,
            bptt_length=bptt_length,
            num_resets_per_eval=num_resets_per_eval,
            num_evals=num_evals,
            expose_recent_rollout=expose_recent_rollout,
        )
        self.agent_config: PPOConfig = config

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
        _check_minibatch_geometry(
            config, self.env_args.n_envs, self.network_args.memory is not None
        )

        # Adam's eps: brax's 1e-8 by default, not ActorCritic's 1e-5.
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=adam_eps)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=adam_eps)
        # PPO's reference code never clips its log-std parameter.
        self.network_args = self.network_args.replace(clip_log_std_param=False)
