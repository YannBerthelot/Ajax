from collections.abc import Sequence
from typing import Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES
from ajax.agents.recurrent import make_replay_buffer
from ajax.agents.SAC.action_pipeline import check_action_extensions
from ajax.agents.SAC.state import SACConfig
from ajax.agents.SAC.train_SAC import make_train
from ajax.deprecation import warn_inert
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.extensions.base import Extension
from ajax.networks.memory import MemoryConfig
from ajax.state import AlphaConfig
from ajax.types import EnvType


class SAC(ActorCritic):
    """Soft Actor-Critic agent for continuous action spaces.

    ``extensions=`` is the sole surface for composing research features.

    Several kwargs remain because they thread into network init,
    collection, or the action pipeline at a level the extension
    framework doesn't reach (``use_pid_policy``,
    ``augment_obs_with_expert_action``, ``augment_obs_with_expert_state``,
    ``use_train_frac``, ``normalize_obs_running``, ``store_policy_action``).
    They mirror the matching extension's "static" flag.
    """

    name: str = "SAC"
    supports_memory: bool = True
    replays_transitions: bool = True
    # ``action`` reaches only the action pipeline's slots, and
    # ``transform_action`` the pipeline, with an expert policy
    # (check_action_extensions).
    supported_extension_phases: frozenset = LOOP_PHASES | {
        "on_target",
        "critic_loss",
        "actor_loss",
        "action",
        "transform_action",
    }
    supported_buffer_fields: frozenset = frozenset({"raw_next_obs"})
    stores_expert_actions: bool = True
    make_train = staticmethod(make_train)

    def __init__(
        self,
        env_id: str | EnvType,
        n_envs: int = 1,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        actor_architecture=("256", "relu", "256", "relu"),
        critic_architecture=("256", "relu", "256", "relu"),
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: Optional[float] = None,
        weight_decay: float = 0.0,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = int(1e4),
        tau: float = 0.005,
        reward_scale: float = 1.0,
        alpha_init: float = 1.0,
        target_entropy_per_dim: float = -1.0,
        # Pluggable memory block (see ajax.networks.memory). When set, the
        # replay buffer stores per-env trajectories and updates train on
        # `sequence_length`-step segments after a `burn_in` warm-up whose
        # carries are computed from zero under stop_gradient (R2D2-style).
        # Incompatible with the expert-guidance feature set (loud error).
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
        # Expert objects: still used by collection / cloning / a fair
        # number of action-pipeline call sites. ``expert_policy=`` is
        # also accepted by individual Extensions and SHOULD be supplied
        # to them too.
        expert_policy: Optional[Callable] = None,
        eval_expert_policy: Optional[Callable] = None,  # for eval logging only
        # Deprecated, no effect: SAC counts each env's steps in its episode
        # when an extension reads them (JSRLCurriculum).
        jsrl_curriculum: Optional[bool] = None,
        # PID policy: gain-mode short-circuit in the action pipeline,
        # changes the action_dim of actor/critic/buffer.
        use_pid_policy: bool = False,
        fixed_alpha: bool = False,
        num_critics: int = 2,
        num_critic_updates: int = 1,
        # Expert prefill: steps of expert_policy per env written to the
        # replay buffer before training.
        expert_buffer_n_steps: int = 20_000,
        # Obs augmentation: changes init_SAC / collect_experience
        # network input dim.
        augment_obs_with_expert_action: bool = False,
        # Train-fraction conditioning: append timestep/total_timesteps
        # to obs (changes the network input dim).
        use_train_frac: bool = False,
        policy_update_start: int = 2_000,
        alpha_update_start: int = 2_000,
        # Warmup expert-vs-uniform mix fraction (gates uniform sampling
        # in the action pipeline).
        expert_fraction: float = 0.7,
        # Expert-state augmentation (PID integrator state etc.).
        # Changes the network input dim.
        augment_obs_with_expert_state: bool = False,
        # Agent-side running observation normalisation. When True the
        # full augmented obs (env_obs + flatten(expert_state)) is
        # z-scored using running mean/var maintained inside the agent's
        # CollectorState.
        normalize_obs_running: bool = False,
        # Off-policy-correctness ablation: store the policy's sampled
        # action in the buffer instead of the executed action.
        store_policy_action: bool = False,
        # target_entropy time ramp (update_agent).
        target_entropy_initial_per_dim: Optional[float] = None,
        target_entropy_ramp_frac: float = 0.5,
        # Deliberate transitional shim (CLAUDE.md rule 10, expand then
        # contract): the critic pretraining this switched off is gone, so
        # True is a no-op and False cannot be honoured. AjaxExperiments
        # still passes it; remove it once AjaxExperiments stops.
        skip_critic_pretrain: bool = True,
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
            squash=True,
            episode_length=episode_length,
            action_repeat=action_repeat,
            extensions=extensions,
        )
        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate, alpha_init=alpha_init
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("SAC only supports continuous action spaces.")
        check_action_extensions(
            self.extension_stack,
            dispatched=expert_policy is not None and not use_pid_policy,
        )

        # Gain-policy mode: actor outputs PID gains (see make_action_pipeline).
        # Critic / buffer / target_entropy then operate in gain-space.
        action_dim_override: Optional[int] = None
        if use_pid_policy:
            if expert_policy is None or not hasattr(expert_policy, "learnable_fields"):
                raise ValueError(
                    "use_pid_policy=True requires a FunctionalExpertPolicy with"
                    " learnable_fields (registered via register_learnable_gains)."
                )
            action_dim_override = len(expert_policy.learnable_fields)
            if augment_obs_with_expert_action:
                raise ValueError(
                    "use_pid_policy=True cannot take augment_obs_with_expert_action"
                    "=True: the actor's output sets the gains the expert acts"
                    " with, so the expert's action does not exist before the"
                    " actor reads its observation."
                )

        action_dim = (
            action_dim_override
            if action_dim_override is not None
            else get_action_dim(self.env_args.env, env_params)
        )
        self.agent_config: SACConfig = SACConfig(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy_per_dim * action_dim,
            reward_scale=reward_scale,
            burn_in=burn_in,
            sequence_length=sequence_length,
            stored_state=stored_state,
        )
        self.buffer = make_replay_buffer(
            self.agent_config, n_envs, self.network_args.memory, buffer_size, batch_size
        )
        if not skip_critic_pretrain:
            raise ValueError(
                "SAC no longer pretrains its critic on the cloning dataset; "
                "use the MCPretrain extension."
            )
        self.eval_expert_policy = self._in_agent_units(eval_expert_policy)
        warn_inert(
            "SAC",
            "SAC counts each env's steps in its episode when an extension reads"
            " them (JSRLCurriculum).",
            jsrl_curriculum=jsrl_curriculum,
        )
        expert_state_aug_dim = 0
        if augment_obs_with_expert_state and expert_policy is not None:
            from ajax.environments.interaction import expert_state_dim

            expert_state_aug_dim = expert_state_dim(expert_policy)
        self.train_kwargs = {
            "buffer": self.buffer,
            "alpha_args": self.alpha_args,
            "expert_policy": self.expert_policy,
            "eval_expert_policy": self.eval_expert_policy,
            "use_pid_policy": use_pid_policy,
            "fixed_alpha": fixed_alpha,
            "num_critics": num_critics,
            "num_critic_updates": num_critic_updates,
            "expert_buffer_n_steps": expert_buffer_n_steps,
            "augment_obs_with_expert_action": augment_obs_with_expert_action,
            "use_train_frac": use_train_frac,
            "policy_update_start": policy_update_start,
            "alpha_update_start": alpha_update_start,
            "expert_fraction": expert_fraction,
            "augment_obs_with_expert_state": augment_obs_with_expert_state,
            "store_policy_action": store_policy_action,
            "expert_state_aug_dim": expert_state_aug_dim,
            "normalize_obs_running": normalize_obs_running,
            "target_entropy_initial": (
                target_entropy_initial_per_dim * action_dim
                if target_entropy_initial_per_dim is not None
                else None
            ),
            "target_entropy_ramp_frac": target_entropy_ramp_frac,
            "action_dim_override": action_dim_override,
        }
