from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional, Union

from gymnax import EnvParams

from ajax.agents.base import ActorCritic, run_config
from ajax.agents.loop import LOOP_PHASES, REPLAY_BATCH_KEYS, TARGET_KEYS
from ajax.agents.recurrent import make_replay_buffer
from ajax.agents.SAC.action_pipeline import check_action_extensions
from ajax.agents.SAC.deprecated_keywords import translate_deprecated_keywords
from ajax.agents.SAC.state import SACConfig
from ajax.agents.SAC.train_SAC import make_train
from ajax.controller import policy_action_dim
from ajax.deprecation import moved_names
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
)
from ajax.extensions.base import Extension
from ajax.modules.pid_actor import PIDPolicy
from ajax.networks.memory import MemoryConfig
from ajax.state import AlphaConfig
from ajax.types import EnvType

# Moved with SAC's other deprecated keywords; the old path goes in 0.5.0.
__getattr__ = moved_names(
    __name__, {"augmentation_features": "ajax.agents.SAC.deprecated_keywords"}
)


class SAC(ActorCritic):
    """Soft Actor-Critic agent for continuous action spaces.

    The extensions keyword is the sole surface for composing research
    features. A few keywords remain because they thread into collection or
    the action pipeline at a level the extension framework does not reach
    (store_policy_action). The deprecated ones (expert_buffer_n_steps,
    expert_fraction, the augmentation keywords, use_pid_policy,
    normalize_obs_running, jsrl_curriculum, skip_critic_pretrain) warn and
    build what replaces them, in ajax.agents.SAC.deprecated_keywords.
    """

    name: str = "SAC"
    supports_memory: bool = True
    replays_transitions: bool = True
    capabilities: frozenset = frozenset(
        {
            "actor",
            "critic_ensemble",
            "discounted",
            "obs_networks",
            "off_policy",
            "policy_density",
            "q_critic",
            "replay",
            "target_critic",
            "reward_learning",
            "warmup",
        }
    )
    # The action phase reaches only the action pipeline's slots, and
    # transform_action the pipeline, with an expert policy
    # (check_action_extensions); choose_warmup_action reaches SAC's pipeline,
    # or the default one under gain mode.
    supported_extension_phases: frozenset = LOOP_PHASES.union(
        {
            "on_target",
            "critic_loss",
            "actor_loss",
            "action",
            "transform_action",
            "actuate",
            "choose_warmup_action",
        }
    )
    # Its critic step's target_extras join the shared step's on_target keys.
    phase_batch_keys: Mapping[str, frozenset] = {
        **REPLAY_BATCH_KEYS,
        "on_target": TARGET_KEYS.union({"rng_key", "q_preds", "recurrent"}),
    }
    pretrain_context: frozenset = frozenset({"buffer", "env_args", "update"})
    extension_context: frozenset = frozenset(
        {
            "actor_optimizer_args",
            "buffer",
            "critic_optimizer_args",
            "env_args",
            "expert_policy",
            "extensions",
            "gamma",
            "network_args",
            "num_critics",
            "recurrent",
            "reward_scale",
            "update_target_fn",
            "update_value_fn",
        }
    )
    supported_buffer_fields: frozenset = frozenset({"raw_next_obs", "return_to_go"})
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
        # Deprecated: gain mode, ExpertGains() in extensions.
        use_pid_policy: bool = False,
        fixed_alpha: bool = False,
        num_critics: int = 2,
        num_critic_updates: int = 1,
        # Deprecated: steps of expert_policy per env written to the replay
        # before training, now ExpertPrefill(n_steps) in extensions. None,
        # with an expert, is 20 000, until B7
        # (deprecated_keywords.prefill_extension).
        expert_buffer_n_steps: Optional[int] = None,
        # Obs augmentation: changes init_SAC / collect_experience
        # network input dim.
        augment_obs_with_expert_action: bool = False,
        # Train-fraction conditioning: append timestep/total_timesteps
        # to obs (changes the network input dim).
        use_train_frac: bool = False,
        policy_update_start: int = 2_000,
        alpha_update_start: int = 2_000,
        # Deprecated: the share of warm-up steps the expert takes, now
        # ExpertWarmup(fraction) in extensions. None, with an expert, is
        # 0.7, until B7 (deprecated_keywords.warmup_extension).
        expert_fraction: Optional[float] = None,
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
        features, normalize_observations = translate_deprecated_keywords(
            expert_policy,
            extensions=extensions,
            learning_starts=learning_starts,
            expert_fraction=expert_fraction,
            expert_buffer_n_steps=expert_buffer_n_steps,
            normalize_observations=normalize_observations,
            normalize_obs_running=normalize_obs_running,
            use_train_frac=use_train_frac,
            augment_obs_with_expert_state=augment_obs_with_expert_state,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            use_pid_policy=use_pid_policy,
            jsrl_curriculum=jsrl_curriculum,
            skip_critic_pretrain=skip_critic_pretrain,
        )
        # The initialisation of SAC's reference code, Glorot-uniform
        # kernels and zero biases in both: haarnoja/sac @8258e33
        # (sac/misc/mlp.py:24-25, :41-49: tf.contrib's
        # xavier_initializer(), uniform by default at tensorflow 1.4, its
        # environment.yml's) and rail-berkeley/softlearning @13cf187,
        # the learned-temperature code (softlearning/models/
        # feedforward.py:28-34: Keras Dense, glorot_uniform and zeros by
        # default at tensorflow 2.4.1, its requirements'). Both draw the
        # mean and the log-std from one output layer; here each head
        # takes Glorot's bound for its own width.
        # Recurrent SAC (memory set) keeps Ajax's previous initialisation
        # until Ni et al.'s, its reference (pomdp-baselines), is audited: at
        # the feedforward reference's, its memory cells' actors read 25-28/32.
        reference_init: dict[str, Any] = {}
        if memory is None:
            reference_init = {
                "encoder_kernel_init": "xavier_uniform",
                "value_kernel_init": "xavier_uniform",
            }
            # The Gaussian head's settings are the MLP actor's: a PID actor in
            # its place keeps its own log-std start (PID_LOG_STD_INIT).
            if not isinstance(actor_architecture, PIDPolicy):
                reference_init.update(
                    {
                        "mean_kernel_init": "xavier_uniform",
                        "log_std_kernel_init": "xavier_uniform",
                        "log_std_init": 0.0,
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
            extensions=(*extensions, *features),
        )
        # Adam's eps is TensorFlow's default, 1e-8, which softlearning's
        # tf.train.AdamOptimizer keeps (a1253bc sac.py:236-244, :318-324);
        # recurrent SAC's reference (Ni et al., pomdp-baselines e7c19c3
        # policy_rnn.py:75, :93) keeps torch's, also 1e-8. Ajax's shared
        # default is 1e-5.
        self.actor_optimizer_args = self.actor_optimizer_args.replace(eps=1e-8)
        self.critic_optimizer_args = self.critic_optimizer_args.replace(eps=1e-8)
        self.alpha_args = AlphaConfig(
            learning_rate=alpha_learning_rate, alpha_init=alpha_init
        )
        if not check_if_environment_has_continuous_actions(self.env_args.env):
            raise ValueError("SAC only supports continuous action spaces.")
        # Gain mode (ExpertGains) acts through the default pipeline, which
        # dispatches no action extension yet (it runs the warm-up's chooser).
        gain_mode = self.env_args.policy_action_dim is not None
        check_action_extensions(
            self.extension_stack,
            dispatched=expert_policy is not None and not gain_mode,
        )

        # Gain mode's critic, buffer and target entropy act in gain space.
        action_dim = policy_action_dim(self.env_args)
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
        self.eval_expert_policy = self._in_agent_units(eval_expert_policy)
        self.train_kwargs = {
            "buffer": self.buffer,
            "alpha_args": self.alpha_args,
            "expert_policy": self.expert_policy,
            "eval_expert_policy": self.eval_expert_policy,
            "fixed_alpha": fixed_alpha,
            "num_critics": num_critics,
            "num_critic_updates": num_critic_updates,
            "policy_update_start": policy_update_start,
            "alpha_update_start": alpha_update_start,
            "store_policy_action": store_policy_action,
            "target_entropy_initial": (
                target_entropy_initial_per_dim * action_dim
                if target_entropy_initial_per_dim is not None
                else None
            ),
            "target_entropy_ramp_frac": target_entropy_ramp_frac,
        }
