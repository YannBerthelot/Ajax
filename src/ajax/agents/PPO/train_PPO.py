from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial

from ajax.agents.PPO.state import PPOConfig, PPOState
from ajax.agents.PPO.utils import (
    _compute_gae,
    get_minibatches_from_batch,
    get_minibatches_preserving_time,
)
from ajax.agents.SAC.utils import SquashedNormal
from ajax.environments.interaction import (
    collect_experience,
    get_pi,
    init_collector_state,
    reset,
)
from ajax.environments.utils import (
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
)
from ajax.extensions.base import ExtensionStack
from ajax.log import compose_eval_metrics, evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.modules.pid_actor import PIDActorConfig
from ajax.networks.networks import (
    get_initialized_actor_critic,
    predict_value,
)
from ajax.perf_utils import build_resumable_train
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    zeros_like_abstract_pytree,
)

PROFILER_PATH = "./tensorboard"

DEBUG = False


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: float
    log_probs: float
    old_log_probs: float
    clip_fraction: float
    entropy: float


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: float
    predictions: float
    targets: float


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


def init_PPO(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    window_size: int = 10,
    pid_actor_config: Optional[PIDActorConfig] = None,
    normalize_obs_running: bool = False,
) -> PPOState:
    """
    Initialize the PPO agent's state, including actor, critic, alpha, and collector states.

    Args:
        key (jax.Array): Random number generator key.
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        buffer (BufferType): Replay buffer.

    Returns:
        PPOState: Initialized PPO agent state.
    """
    (
        rng,
        init_key,
        collector_key,
    ) = jax.random.split(key, num=3)

    continuous = check_if_environment_has_continuous_actions(
        env_args.env, env_params=env_args.env_params
    )
    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=continuous,
        action_value=False,
        squash=network_args.squash,
        num_critics=1,
        pid_actor_config=pid_actor_config,
        log_std_state_independent=network_args.log_std_state_independent,
        log_std_init=network_args.log_std_init,
        mean_kernel_init=network_args.mean_kernel_init,
        disable_encoder_output_norm=network_args.disable_encoder_output_norm,
        actor_kernel_init=network_args.actor_kernel_init,
        actor_bias_init=network_args.actor_bias_init,
        critic_kernel_init=network_args.critic_kernel_init,
        critic_bias_init=network_args.critic_bias_init,
        encoder_kernel_init=network_args.encoder_kernel_init,
        encoder_bias_init=network_args.encoder_bias_init,
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        window_size=window_size,
        normalize_obs_running=normalize_obs_running,
    )

    # Pre-allocate obs_norm_info on actor/critic state so the scan
    # carry's pytree stays stable from iteration zero. The first
    # collect step will sync the running stats from
    # ``collector_state.obs_norm_info`` here (a fresh zero-init from
    # ``init_agent_obs_norm``); without this preallocation, the scan
    # input carry has ``None`` while the output carry (after the
    # collect sync) has a ``NormalizationInfo`` -> pytree-structure
    # mismatch and the scan rejects the body.
    if normalize_obs_running and collector_state.obs_norm_info is not None:
        actor_state = actor_state.replace(obs_norm_info=collector_state.obs_norm_info)
        critic_state = critic_state.replace(obs_norm_info=collector_state.obs_norm_info)

    return PPOState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
        n_updates=0,
    )


# @partial(jax.jit, static_argnames=["recurrent"])
def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    dones: jax.Array,
    recurrent: bool,
    vf_coef: float = 1.0,
    agent_state: Optional[Any] = None,
    extra_loss_fn: Optional[Callable] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """
    Compute the value loss for the critic networks.

    Args:
        critic_params (FrozenDict): Parameters of the critic networks.
        critic_states (LoadedTrainState): Critic train states.
        rng (jax.Array): Random number generator key.
        actor_state (LoadedTrainState): Actor train state.
        actions (jax.Array): Actions taken.
        observations (jax.Array): Current observations.
        next_observations (jax.Array): Next observations.
        dones (jax.Array): Done flags.
        rewards (jax.Array): Rewards received.
        gamma (float): Discount factor.
        alpha (jax.Array): Temperature parameter.
        recurrent (bool): Whether the model is recurrent.
        reward_scale (float): Reward scaling factor.

    Returns:
        Tuple[jax.Array, Dict[str, jax.Array]]: Loss and auxiliary metrics.
    """

    # Predict V-values from critics
    v_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_params,
        x=observations,
    ).squeeze(
        0
    )  # squeeze to stay consistent with ensemble_critic that adds a leading dimension even for a single critic.

    loss = vf_coef * 0.5 * jnp.mean((v_preds - value_targets) ** 2)
    if extra_loss_fn is not None:
        loss = loss + extra_loss_fn(
            critic_params, critic_states, observations, value_targets, agent_state
        )

    return loss, ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
    )


# @partial(
#     jax.jit,
#     static_argnames=["recurrent", "advantage_normalization"],
# )
def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    clip_coef: float,
    ent_coef: float,
    advantage_normalization: bool,
    obs_preprocessor: Optional[Callable] = None,
    extra_loss_fn: Optional[Callable] = None,
    raw_actions: Optional[jax.Array] = None,
    entropy_rng: Optional[jax.Array] = None,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """
    Compute the policy loss for the actor network.

    Args:
        actor_params (FrozenDict): Parameters of the actor network.
        actor_state (LoadedTrainState): Actor train state.
        critic_states (LoadedTrainState): Critic train states.
        observations (jax.Array): Current observations.
        dones (Optional[jax.Array]): Done flags.
        recurrent (bool): Whether the model is recurrent.
        alpha (jax.Array): Temperature parameter.
        rng (jax.random.PRNGKey): Random number generator key.

    Returns:
        Tuple[jax.Array, Dict[str, jax.Array]]: Loss and auxiliary metrics.
    """
    obs_for_actor = (
        obs_preprocessor(observations) if obs_preprocessor is not None else observations
    )
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=obs_for_actor,
        done=dones,
        recurrent=recurrent,
    )

    # Need to deal with various shapes depending on brax vs gymnax and discrete vs continuous

    if isinstance(pi, distrax.Categorical):
        new_log_probs = jnp.expand_dims(
            pi.log_prob(actions.squeeze(-1)), -1
        )  # .sum(-1, keepdims=True)
    elif isinstance(pi, SquashedNormal) and raw_actions is not None:
        new_log_probs = pi.log_prob_from_raw(raw_actions)
    else:
        new_log_probs = pi.log_prob(actions).sum(-1, keepdims=True)
    if DEBUG:
        assert new_log_probs.shape == log_probs.shape, (
            f"Shape mismatch between new_log_probs {new_log_probs.shape} and log_probs"
            f" {log_probs.shape}"
        )

    ratio = jnp.exp(new_log_probs - log_probs)

    if advantage_normalization:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    if DEBUG:
        assert (
            ratio.shape[0] == gae.shape[0]
        ), f"Mismatch between ratio shape ({ratio.shape}) and gae shape ({gae.shape})"
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - clip_coef,
            1.0 + clip_coef,
        )
        * gae
    )

    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

    # CALCULATE AUXILIARIES
    clip_fraction = (jnp.abs(ratio - 1) > clip_coef).mean()

    # Match brax NormalTanhDistribution.entropy: latent Gaussian entropy
    # plus the tanh log-det-jacobian evaluated at a sample (1-sample MC
    # estimate). Without the Jacobian term, the entropy bonus is the
    # latent-Gaussian entropy only -- which is independent of mean, so
    # the entropy gradient never flows back to the mean head.
    if isinstance(pi, SquashedNormal) and entropy_rng is not None:
        entropy = pi.effective_entropy(entropy_rng, num_samples=1).mean()
    elif isinstance(pi, SquashedNormal):
        entropy = pi.unsquashed_entropy().mean()
    else:
        entropy = pi.entropy().mean()

    total_loss = loss_actor - ent_coef * entropy
    if extra_loss_fn is not None:
        total_loss = total_loss + extra_loss_fn(actor_params, actor_state)

    return total_loss, PolicyAuxiliaries(
        policy_loss=total_loss,
        log_probs=new_log_probs.mean(),
        old_log_probs=log_probs.mean(),
        clip_fraction=clip_fraction,
        entropy=entropy,
    )


VALUE_AND_GRAD_FN = jax.value_and_grad(value_loss_function, has_aux=True)
POLICY_AND_GRAD_FN = jax.value_and_grad(policy_loss_function, has_aux=True)


def _value_and_grad_with_extra(extra_loss_fn):
    """Return value_and_grad of value_loss_function with ``extra_loss_fn`` bound
    by closure so it doesn't have to be passed as an arg through jax's flatten.
    """

    def bound(
        critic_params,
        critic_states,
        observations,
        value_targets,
        dones,
        recurrent,
        agent_state,
        vf_coef=1.0,
    ):
        return value_loss_function(
            critic_params,
            critic_states,
            observations,
            value_targets,
            dones,
            recurrent,
            vf_coef=vf_coef,
            agent_state=agent_state,
            extra_loss_fn=extra_loss_fn,
        )

    return jax.value_and_grad(bound, has_aux=True)


def _policy_value_and_grad_with_extra(extra_loss_fn):
    def bound(
        actor_params,
        actor_state,
        observations,
        actions,
        log_probs,
        gae,
        dones,
        recurrent,
        clip_coef,
        ent_coef,
        advantage_normalization,
        obs_preprocessor,
        raw_actions=None,
        entropy_rng=None,
    ):
        return policy_loss_function(
            actor_params,
            actor_state,
            observations,
            actions,
            log_probs,
            gae,
            dones,
            recurrent,
            clip_coef,
            ent_coef,
            advantage_normalization,
            obs_preprocessor,
            extra_loss_fn=extra_loss_fn,
            raw_actions=raw_actions,
            entropy_rng=entropy_rng,
        )

    return jax.value_and_grad(bound, has_aux=True)


def check_no_nan(x, id):
    assert not jnp.isnan(x).any(), f"NaN detected {id}"


def _normalize_obs_with_stats(obs, obs_norm_info):
    """Apply the env's running-stats normaliser to raw observations.

    Mirrors ``ajax.utils.online_normalize``'s formula at use-time: mean
    of the batched-stat across envs, std = clip(sqrt(var + 1e-8),
    1e-6, 1e6) likewise. With ``obs_norm_info`` containing the LATEST
    stats from ``env_state.info["normalization_info"].obs``, the
    forward pass sees brax-style normalise-at-forward semantics.
    Returns raw obs unchanged when obs_norm_info is None.
    """
    if obs_norm_info is None:
        return obs
    norm_mean = obs_norm_info.mean.mean(axis=0)
    norm_std = jnp.clip(jnp.sqrt(obs_norm_info.var + 1e-8), 1e-6, 1e6).mean(axis=0)
    return (obs - norm_mean) / norm_std


# ---------------------------------------------------------------------------
# Extension-stack composition helpers
# ---------------------------------------------------------------------------
def _compose_extra_critic_loss(
    user_fn: Optional[Callable],
    extension_stack: Optional[ExtensionStack],
    agent_state: PPOState,
    total_timesteps: int,
) -> Optional[Callable]:
    """Combine the user ``extra_critic_loss_fn`` with stack.critic_loss.

    Returns ``None`` (so the agent skips the extra-loss code path) when
    no contribution exists. The returned callable matches the signature
    expected by :func:`value_loss_function`:
    ``(critic_params, critic_states, observations, value_targets,
    agent_state) -> scalar``. ``agent_state`` is the iteration-time
    state captured by closure; the per-minibatch ``agent_state`` is also
    passed through but the stack reads ``ext_state`` off the closure
    instance for determinism.
    """
    has_stack = extension_stack is not None and bool(extension_stack.extensions)
    if user_fn is None and not has_stack:
        return None

    def combined(critic_params, critic_states, observations, value_targets, _astate):
        loss: jax.Array | float = 0.0
        if user_fn is not None:
            loss = loss + user_fn(
                critic_params, critic_states, observations, value_targets, _astate
            )
        if has_stack:
            assert extension_stack is not None
            _batch = {
                "observations": observations,
                "targets": value_targets,
                "critic_params": critic_params,
                "critic_state": critic_states,
            }
            loss = loss + extension_stack.fold_critic_loss(
                agent_state,
                _batch,
                agent_state.collector_state.timestep,
                agent_state.rng,
                total_timesteps,
            )
        return loss

    return combined


def _compose_extra_actor_loss(
    user_fn: Optional[Callable],
    extension_stack: Optional[ExtensionStack],
    agent_state: PPOState,
    total_timesteps: int,
) -> Optional[Callable]:
    """Combine the user ``extra_actor_loss_fn`` with stack.actor_loss.

    Returns ``None`` when nothing contributes. The returned callable
    matches the signature expected by :func:`policy_loss_function`:
    ``(actor_params, actor_state) -> scalar``.
    """
    has_stack = extension_stack is not None and bool(extension_stack.extensions)
    if user_fn is None and not has_stack:
        return None

    def combined(actor_params, actor_state):
        loss: jax.Array | float = 0.0
        if user_fn is not None:
            loss = loss + user_fn(actor_params, actor_state)
        if has_stack:
            assert extension_stack is not None
            _batch = {
                "actor_params": actor_params,
                "actor_state": actor_state,
            }
            loss = loss + extension_stack.fold_actor_loss(
                agent_state,
                _batch,
                agent_state.collector_state.timestep,
                agent_state.rng,
                total_timesteps,
            )
        return loss

    return combined


@partial(
    jax.jit,
    static_argnames=[
        "env_args",
        "mode",
        "recurrent",
        "log_frequency",
        "num_episode_test",
        "log_fn",
        "log",
        "verbose",
        "lstm_hidden_size",
        "agent_config",
        "horizon",
        "total_timesteps",
        "n_steps",
        "action_pipeline",
        "eval_action_transform",
        "obs_preprocessor",
        "auxiliary_update",
        "extra_eval_metrics",
        "extra_actor_loss_fn",
        "extra_critic_loss_fn",
        "reward_shaping_fn",
        "extension_stack",
    ],
)
def training_iteration(  # noqa: C901  (brax-faithful PPO has many gated branches)
    agent_state: PPOState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    agent_config: PPOConfig,
    total_timesteps: int,
    n_steps: int,
    total_n_updates: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: int = 1000,
    horizon: int = 10000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
    action_pipeline: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    obs_preprocessor: Optional[Callable] = None,
    auxiliary_update: Optional[Callable] = None,
    extra_eval_metrics: Optional[Callable] = None,
    extra_actor_loss_fn: Optional[Callable] = None,
    extra_critic_loss_fn: Optional[Callable] = None,
    reward_shaping_fn: Optional[Callable] = None,
    extension_stack: Optional[ExtensionStack] = None,
) -> tuple[PPOState, None]:
    """
    Perform one training iteration, including experience collection and agent updates.

    Args:
        agent_state (PPOState): Current PPO agent state.
        _ (Any): Placeholder for scan compatibility.
        env_args (EnvironmentConfig): Environment configuration.
        mode (str): Environment mode ("gymnax" or "brax").
        recurrent (bool): Whether the model is recurrent.
        buffer (BufferType): Replay buffer.
        agent_config (PPOConfig): PPO agent configuration.
        action_dim (int): Action dimensionality.
        lstm_hidden_size (Optional[int]): LSTM hidden size for recurrent models.
        log_frequency (int): Frequency of logging and evaluation.
        num_episode_test (int): Number of episodes for evaluation.

    Returns:
        Tuple[PPOState, None]: Updated agent state.
    """
    # collector_state = agent_state.collector_state

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        action_pipeline=action_pipeline,
    )
    agent_state, transition = jax.lax.scan(
        collect_scan_fn, agent_state, xs=None, length=n_steps
    )  # transition = s_t, a_t, r_{s_t -> s_{t+1}}, s_{t+1}, d_{s_t -> s_{t+1}}

    # Gap A: expose the freshly collected ``(T, n_envs, ...)`` rollout on
    # ``agent_state.last_rollout`` so cross-agent measurement extensions
    # (EVarEst-style probes) can read an on-state-visitation batch
    # without triggering a fresh rollout per eval. Off by default — the
    # placeholder allocated in ``make_train`` keeps the scan carry's
    # pytree structure stable; when disabled, ``last_rollout`` stays
    # ``None`` and this branch is skipped (zero extra cost).
    if getattr(agent_config, "expose_recent_rollout", False):
        agent_state = agent_state.replace(last_rollout=transition)

    # GAE / value-target handling depends on the minibatch geometry:
    #
    #   * brax-faithful path (``n_envs % num_minibatches == 0``):
    #     time-preserving minibatches built by splitting the env axis,
    #     value baseline + GAE recomputed inside ``mb_body`` with the
    #     CURRENT critic params (matches
    #     brax/training/agents/ppo/losses.py::compute_ppo_loss).
    #
    #   * Legacy path (cannot evenly split env axis -- mostly small
    #     test fixtures with ``n_envs=1``): GAE precomputed once on the
    #     full rollout, then flat-shuffled minibatches (Ajax pre-rework
    #     behaviour). Required because per-mb GAE needs ``n_envs %
    #     num_minibatches == 0`` to keep within-fragment causality.

    if reward_shaping_fn is not None:
        shaped_rewards = transition.reward + reward_shaping_fn(agent_state, transition)
    else:
        shaped_rewards = transition.reward

    if agent_config.num_minibatches > 0:
        num_minibatches = agent_config.num_minibatches
    else:
        if DEBUG:
            assert (
                max(agent_config.batch_size, agent_config.n_steps)
                % min(agent_config.batch_size, agent_config.n_steps)
                == 0
            ), (
                "can't evenly break n_steps into batch size chunks,"
                f" n_steps={agent_config.n_steps} batch_size={agent_config.batch_size}"
            )
        num_minibatches = max(agent_config.batch_size, agent_config.n_steps) // min(
            agent_config.batch_size, agent_config.n_steps
        )

    n_envs = transition.obs.shape[1]
    T_rollout = transition.obs.shape[0]
    unroll_length = agent_config.unroll_length
    # Decide which minibatch geometry to use:
    #
    #   * If ``unroll_length`` is set, sub-split T axis into fragments of
    #     length ``unroll_length`` (brax convention). Need
    #     ``(T/unroll_length) * n_envs`` divisible by ``num_minibatches``
    #     AND ``T % unroll_length == 0``.
    #
    #   * Otherwise, if ``n_envs % num_minibatches == 0``, env-axis split
    #     (each fragment spans the full T).
    #
    #   * Else, legacy flat-shuffle with precomputed GAE.
    #
    # Additional gate: the brax-faithful path recomputes GAE inside
    # mb_body with the CURRENT critic params. That's only meaningful
    # when each minibatch carries DIFFERENT data (so each per-mb GAE
    # call sees a different (obs, reward, next_obs) slice). When
    # ``num_minibatches == 1`` and ``n_epochs > 1`` the single
    # minibatch IS the full rollout, and per-mb recompute degenerates
    # into recomputing the same GAE on the same data with a critic
    # that just took an SGD step -- this oscillates the value targets
    # and was the cause of CI flake on the n_envs=1, n_steps=32
    # probing fixture. Fall back to the legacy precompute path when
    # num_minibatches <= 1.
    if num_minibatches <= 1:
        use_brax_faithful_mb = False
        mb_unroll_length = None
    elif (
        unroll_length is not None
        and T_rollout % unroll_length == 0
        and (T_rollout // unroll_length) * n_envs % num_minibatches == 0
    ):
        use_brax_faithful_mb = True
        mb_unroll_length = unroll_length
    elif n_envs >= num_minibatches and n_envs % num_minibatches == 0:
        use_brax_faithful_mb = True
        mb_unroll_length = None  # full-T fragments
    else:
        use_brax_faithful_mb = False
        mb_unroll_length = None

    if use_brax_faithful_mb:
        batch = (
            transition.obs,
            transition.next_obs,
            (
                jnp.expand_dims(transition.action, axis=-1)
                if jnp.ndim(transition.action) < 3
                else transition.action
            ),
            transition.terminated,
            transition.truncated,
            (
                jnp.expand_dims(transition.log_prob, axis=-1)
                if jnp.ndim(transition.log_prob) < 3
                else transition.log_prob.sum(-1, keepdims=True)
            ),
            transition.raw_action,
            shaped_rewards,
        )
        shuffle_key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        shuffled_batch = get_minibatches_preserving_time(
            batch,
            rng=shuffle_key,
            num_minibatches=num_minibatches,
            unroll_length=mb_unroll_length,
        )
    else:
        # Legacy: precompute GAE on full rollout, flat-shuffle.
        values = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.params,
            x=transition.obs,
        ).squeeze(0)
        next_values = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.params,
            x=transition.next_obs,
        ).squeeze(0)
        gae, value_targets = _compute_gae(
            values=values,
            next_values=next_values,
            rewards=shaped_rewards,
            terminateds=transition.terminated,
            truncateds=transition.truncated,
            gamma=agent_config.gamma,
            gae_lambda=agent_config.gae_lambda,
        )
        if extension_stack is not None:
            _tgt_rng, _ppo_rng = jax.random.split(agent_state.rng)
            agent_state = agent_state.replace(rng=_ppo_rng)
            _tgt_batch = {
                "observations": transition.obs,
                "next_observations": transition.next_obs,
                "rewards": shaped_rewards,
                "terminated": transition.terminated,
                "truncated": transition.truncated,
                "gae": gae,
                "values": values,
                "next_values": next_values,
                "gamma": agent_config.gamma,
            }
            value_targets = extension_stack.fold_on_target(
                agent_state,
                _tgt_batch,
                value_targets,
                agent_state.collector_state.timestep,
                _tgt_rng,
                total_timesteps,
            )
        batch = (
            transition.obs,
            (
                jnp.expand_dims(transition.action, axis=-1)
                if jnp.ndim(transition.action) < 3
                else transition.action
            ),
            transition.terminated,
            transition.truncated,
            value_targets,
            gae,
            (
                jnp.expand_dims(transition.log_prob, axis=-1)
                if jnp.ndim(transition.log_prob) < 3
                else transition.log_prob.sum(-1, keepdims=True)
            ),
            transition.raw_action,
        )
        shuffle_key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        shuffled_batch = get_minibatches_from_batch(
            batch, rng=shuffle_key, num_minibatches=num_minibatches
        )

    def do_update(
        agent_state: PPOState, num_epochs: int
    ) -> tuple[PPOState, AuxiliaryLogs]:
        # Closures rebuilt each ``do_update`` so the scan body sees no mutable state.
        _composed_critic_extra = _compose_extra_critic_loss(
            extra_critic_loss_fn,
            extension_stack,
            agent_state,
            total_timesteps,
        )
        _composed_actor_extra = _compose_extra_actor_loss(
            extra_actor_loss_fn,
            extension_stack,
            agent_state,
            total_timesteps,
        )

        # Capture extra_loss_fns in closure (jax rejects function args inside scan).
        critic_grad_fn = (
            VALUE_AND_GRAD_FN
            if _composed_critic_extra is None
            else _value_and_grad_with_extra(_composed_critic_extra)
        )
        actor_grad_fn = (
            POLICY_AND_GRAD_FN
            if _composed_actor_extra is None
            else _policy_value_and_grad_with_extra(_composed_actor_extra)
        )

        # NOTE: Brax-faithful obs normalisation is now wired through
        # the AGENT-side ``obs_norm_info`` carried on ``actor_state`` /
        # ``critic_state`` (synced every collect step via
        # ``collect_experience``). ``get_pi`` and ``predict_value`` both
        # call ``apply_obs_norm`` internally when those fields are set,
        # so mb_body forward calls below get normalised obs without an
        # explicit re-normalisation here. See ``ajax.agents.obs_norm``.

        def mb_body(agent_state, mb):
            """One minibatch update: critic grad + actor grad + (optional
            joint global-norm clip) + apply both. Handles both layouts:

              * brax-faithful: ``mb`` carries raw rollout tensors; the
                value baseline + GAE are recomputed here with CURRENT
                critic params so post-SGD-step updates see fresh value
                targets (matches
                brax/training/agents/ppo/losses.py::compute_ppo_loss).

              * Legacy: ``mb`` already carries precomputed
                ``value_targets`` and ``gae`` from the full rollout.
            """
            if use_brax_faithful_mb:
                (
                    observations,
                    next_observations,
                    actions,
                    terminated,
                    truncated,
                    log_probs_mb,
                    raw_actions_mb,
                    rewards_mb,
                ) = mb
            else:
                (
                    observations,
                    actions,
                    terminated,
                    truncated,
                    value_targets_mb,
                    gae_mb,
                    log_probs_mb,
                    raw_actions_mb,
                ) = mb
            dones = jnp.logical_or(terminated, truncated)
            ent_rng, on_target_rng, new_rng = jax.random.split(agent_state.rng, 3)
            agent_state = agent_state.replace(rng=new_rng)

            if use_brax_faithful_mb:
                # Forward current critic to get baseline + bootstrap
                # value. stop_gradient so the GAE-target side does not
                # flow gradients through the critic via GAE (brax does
                # the same via jax.lax.stop_gradient inside compute_gae).
                values_mb = jax.lax.stop_gradient(
                    predict_value(
                        critic_state=agent_state.critic_state,
                        critic_params=agent_state.critic_state.params,
                        x=observations,
                    ).squeeze(0)
                )
                next_values_mb = jax.lax.stop_gradient(
                    predict_value(
                        critic_state=agent_state.critic_state,
                        critic_params=agent_state.critic_state.params,
                        x=next_observations,
                    ).squeeze(0)
                )
                gae_mb, value_targets_mb = _compute_gae(
                    values=values_mb,
                    next_values=next_values_mb,
                    rewards=rewards_mb,
                    terminateds=terminated,
                    truncateds=truncated,
                    gamma=agent_config.gamma,
                    gae_lambda=agent_config.gae_lambda,
                )
                gae_mb = jax.lax.stop_gradient(gae_mb)
                value_targets_mb = jax.lax.stop_gradient(value_targets_mb)
                if extension_stack is not None:
                    _tgt_batch = {
                        "observations": observations,
                        "next_observations": next_observations,
                        "rewards": rewards_mb,
                        "terminated": terminated,
                        "truncated": truncated,
                        "gae": gae_mb,
                        "values": values_mb,
                        "next_values": next_values_mb,
                        "gamma": agent_config.gamma,
                    }
                    value_targets_mb = extension_stack.fold_on_target(
                        agent_state,
                        _tgt_batch,
                        value_targets_mb,
                        agent_state.collector_state.timestep,
                        on_target_rng,
                        total_timesteps,
                    )

            if _composed_critic_extra is None:
                (_v_loss, v_aux), v_grads = critic_grad_fn(
                    agent_state.critic_state.params,
                    agent_state.critic_state,
                    observations,
                    value_targets_mb,
                    dones,
                    recurrent,
                    agent_config.vf_coef,
                )
            else:
                (_v_loss, v_aux), v_grads = critic_grad_fn(
                    agent_state.critic_state.params,
                    agent_state.critic_state,
                    observations,
                    value_targets_mb,
                    dones,
                    recurrent,
                    agent_state,
                    agent_config.vf_coef,
                )

            clip_coef = (
                agent_config.clip_range(agent_state.collector_state.timestep)
                if callable(agent_config.clip_range)
                else agent_config.clip_range
            )
            # raw_actions as kwarg: hits slot 14, not extra_loss_fn at slot 13.
            (_p_loss, p_aux), p_grads = actor_grad_fn(
                agent_state.actor_state.params,
                agent_state.actor_state,
                observations,
                actions,
                log_probs_mb,
                gae_mb,
                dones,
                recurrent,
                clip_coef,
                agent_config.ent_coef,
                agent_config.normalize_advantage,
                obs_preprocessor,
                raw_actions=raw_actions_mb,
                entropy_rng=ent_rng,
            )

            # brax-style joint global-norm clip (max-norm 1.0).
            if agent_config.fused_grad_clip:
                actor_sq = sum(
                    jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(p_grads)
                )
                critic_sq = sum(
                    jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(v_grads)
                )
                joint_norm = jnp.sqrt(actor_sq + critic_sq + 1e-12)
                scale = jnp.minimum(1.0, 1.0 / joint_norm)
                p_grads = jax.tree_util.tree_map(lambda g: g * scale, p_grads)
                v_grads = jax.tree_util.tree_map(lambda g: g * scale, v_grads)

            agent_state = agent_state.replace(
                critic_state=agent_state.critic_state.apply_gradients(grads=v_grads),
                actor_state=agent_state.actor_state.apply_gradients(grads=p_grads),
            )

            aux = AuxiliaryLogs(
                policy=p_aux,
                value=ValueAuxiliaries(
                    **{k: v.flatten() for k, v in to_state_dict(v_aux).items()}
                ),
            )
            return agent_state, aux

        def body_fn(agent_state, _):
            """One epoch: scan ``mb_body`` over ``shuffled_batch``'s minibatch axis."""
            agent_state, mb_aux = jax.lax.scan(
                f=mb_body, init=agent_state, xs=shuffled_batch
            )
            # Aggregate per-minibatch aux into one aux for this epoch by
            # averaging across the leading minibatch axis.
            aux = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), mb_aux)
            return agent_state, aux

        agent_state, aux = jax.lax.scan(
            f=body_fn, init=agent_state, xs=None, length=num_epochs
        )
        aux = aux.replace(
            value=ValueAuxiliaries(
                **{key: val.flatten() for key, val in to_state_dict(aux.value).items()}
            )
        )
        aux = jax.tree_util.tree_map(
            lambda x: x.mean(), aux
        )  # need to aggregate over the n-epochs
        return (
            agent_state.replace(n_updates=agent_state.n_updates + 1),
            aux,
        )  # aux should be the one from the last epoch

    agent_state, aux = do_update(agent_state, num_epochs=agent_config.n_epochs)

    # Brax-faithful periodic forced env reset. Brax calls
    # ``reset_fn(env_state, key_envs)`` every
    # ``num_training_steps_per_epoch`` training_steps (with
    # ``num_resets_per_eval > 0``), drawing fresh randomised initial
    # conditions. Ajax's BraxAutoResetWrapper caches the FIRST reset
    # state and reuses it indefinitely -- without periodic forced
    # resets, the agent sees only ``n_envs`` distinct starting
    # conditions for the entire training. For envs with randomised
    # reset states (e.g. PandaOpenCabinet perturbs target_pos and arm
    # joints in ``reset``), this dramatically limits diversity.
    #
    # Implementation: every ``reset_every`` iterations, call env.reset
    # with a fresh RNG. The wrapper's reset re-initialises the obs
    # normalizer -- we preserve the running stats by extracting them
    # from old state, then re-normalising the fresh obs with the
    # preserved stats.
    if agent_config.num_resets_per_eval > 0:
        # ``num_evals``/``num_resets_per_eval`` are static (PPOConfig);
        # ``total_n_updates`` is a traced int, so reset_every must be a
        # JAX scalar.
        num_evals_after_init = max(int(agent_config.num_evals) - 1, 1)
        reset_every = jnp.maximum(
            1,
            jnp.ceil(
                total_n_updates
                / (num_evals_after_init * agent_config.num_resets_per_eval)
            ).astype(jnp.int32),
        )

        def _force_reset(agent_state):
            reset_key, new_rng = jax.random.split(agent_state.rng)
            saved_norm = None
            if mode == "brax" and "normalization_info" in (
                agent_state.collector_state.env_state.info or {}
            ):
                saved_norm = agent_state.collector_state.env_state.info[
                    "normalization_info"
                ]
            reset_keys = (
                jax.random.split(reset_key, env_args.n_envs)
                if mode == "gymnax"
                else reset_key
            )
            new_obs, new_env_state = reset(
                reset_keys, env_args.env, mode, env_args.env_params
            )
            if saved_norm is not None:
                fresh_norm = new_env_state.info["normalization_info"]
                fresh_obs_info = fresh_norm.obs
                saved_obs_info = saved_norm.obs
                # Undo fresh normalisation, re-apply saved-stats normalisation.
                # Match online_normalize: it uses ``mean(clipped_std, axis=0)``
                # to broadcast across envs (all rows of the batched stat are
                # identical post-Welford). Apply the same clip + mean here so
                # the recovered raw_obs is bit-for-bit what the env produced.
                fresh_std = jnp.clip(
                    jnp.sqrt(fresh_obs_info.var + 1e-8), 1e-6, 1e6
                ).mean(axis=0)
                fresh_mean = fresh_obs_info.mean.mean(axis=0)
                raw_obs = new_obs * fresh_std + fresh_mean
                saved_std = jnp.clip(
                    jnp.sqrt(saved_obs_info.var + 1e-8), 1e-6, 1e6
                ).mean(axis=0)
                saved_mean = saved_obs_info.mean.mean(axis=0)
                new_obs = (raw_obs - saved_mean) / saved_std
                new_env_state.info["normalization_info"] = saved_norm
                new_env_state = new_env_state.replace(obs=new_obs)
            new_collector = agent_state.collector_state.replace(
                _env_state=new_env_state, last_obs=new_obs
            )
            return agent_state.replace(collector_state=new_collector, rng=new_rng)

        should_reset = (agent_state.n_updates % reset_every == 0) & (
            agent_state.n_updates > 0
        )
        agent_state = jax.lax.cond(should_reset, _force_reset, lambda s: s, agent_state)

    # Extension post_update — folded after the per-iteration update loop.
    # Empty stack ⇒ identity.
    if extension_stack is not None:
        _pu_rng, _pu_rng2 = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=_pu_rng2)
        agent_state = extension_stack.fold_post_update(
            agent_state,
            agent_state.collector_state.timestep,
            _pu_rng,
            total_timesteps,
        )

    if auxiliary_update is not None:
        aux_rng, rng = jax.random.split(agent_state.rng)
        # Stash the full stacked rollout (T, n_envs, *) into
        # collector_state.rollout so the auxiliary_update hook gets the
        # whole iteration's experience, not just the last single
        # transition. collect_experience's leading-iteration check
        # only reads ``rollout is not None`` and ``rollout.raw_obs is
        # not None``, both of which are unaffected by adding a time
        # axis. Restored to the latest single transition after the
        # aux step so we don't leak the stacked shape downstream.
        latest = agent_state.collector_state.rollout
        agent_state = agent_state.replace(
            rng=rng,
            collector_state=agent_state.collector_state.replace(rollout=transition),
        )
        agent_state, auxiliary_metrics = auxiliary_update(agent_state, aux_rng)
        agent_state = agent_state.replace(
            collector_state=agent_state.collector_state.replace(rollout=latest),
        )
    else:
        auxiliary_metrics = {}

    _merged_extra_eval = compose_eval_metrics(
        extra_eval_metrics, extension_stack, total_timesteps
    )
    agent_state, metrics_to_log = evaluate_and_log(
        agent_state,
        aux,
        index,
        mode,
        env_args,
        num_episode_test,
        recurrent,
        lstm_hidden_size,
        log,
        verbose,
        log_fn,
        log_frequency,
        total_timesteps,
        eval_action_transform=eval_action_transform,
        extra_eval_metrics=_merged_extra_eval,
    )

    metrics_to_log = {**metrics_to_log, **auxiliary_metrics}

    # jax.clear_caches()
    # gc.collect()
    return agent_state, metrics_to_log


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    agent_config: PPOConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
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
    extensions: Sequence = (),
    normalize_obs_running: bool = False,
):
    """
    Create the training function for the PPO agent.

    Args:
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        buffer (BufferType): Replay buffer.
        agent_config (PPOConfig): PPO agent configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        total_timesteps (int): Total timesteps for training.
        num_episode_test (int): Number of episodes for evaluation during training.

    Returns:
        Callable: JIT-compiled training function.
    """
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    # Start async logging if logging is enabled
    if logging_config is not None:
        start_async_logging()

    num_updates = (total_timesteps // (env_args.n_envs * agent_config.n_steps)) + 1

    extension_stack = ExtensionStack(extensions) if extensions else None

    def init_fn(key, index):
        # Preserve the original RNG layout: key -> (_, init_key, _).
        _, init_key, _transform_key = jax.random.split(key, 3)
        agent_state = init_PPO(
            key=init_key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            pid_actor_config=pid_actor_config,
            normalize_obs_running=normalize_obs_running,
        )
        if extension_stack is not None:
            # Reuse the unused 3rd split for ext init/pretrain RNG so the
            # legacy (1st, 2nd) slots are unchanged — preserves byte-
            # identical numerics for any code that derived its RNG from
            # the first two splits.
            _ext_key, _pre_key = jax.random.split(_transform_key)
            agent_state = extension_stack.fold_init_states(agent_state, _ext_key)
            agent_state = extension_stack.fold_pretrain(
                agent_state, jnp.asarray(0), _pre_key, total_timesteps
            )
        # Gap A: pre-allocate the ``last_rollout`` placeholder with the
        # exact shape/dtype of one ``(T, n_envs, ...)`` rollout so the
        # JIT-traced scan body sees a stable pytree carry from iteration
        # zero. ``None``-vs-``Transition`` would otherwise change the
        # carry structure on the first iteration and crash the scan.
        if getattr(agent_config, "expose_recent_rollout", False):
            _trace_scan = partial(
                collect_experience,
                recurrent=network_args.lstm_hidden_size is not None,
                mode=mode,
                env_args=env_args,
                action_pipeline=action_pipeline,
            )
            _, _trans_abs = jax.eval_shape(
                lambda st: jax.lax.scan(
                    _trace_scan, st, xs=None, length=agent_config.n_steps
                ),
                agent_state,
            )
            agent_state = agent_state.replace(
                last_rollout=zeros_like_abstract_pytree(_trans_abs)
            )
        return agent_state

    def _init_transform(agent_state, key):
        # One-shot transform consumes ``transform_key`` (the 3rd split of
        # the original key) so RNG matches the pre-refactor layout.
        _, _init_key, transform_key = jax.random.split(key, 3)
        return init_transform(agent_state, transform_key)

    def make_scan_fn(_agent_state, _resume_from_state, _key, index):
        return partial(
            training_iteration,
            recurrent=network_args.lstm_hidden_size is not None,
            agent_config=agent_config,
            n_steps=agent_config.n_steps,
            mode=mode,
            env_args=env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=log,
            total_timesteps=total_timesteps,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            horizon=(logging_config.horizon if logging_config is not None else None),
            total_n_updates=num_updates,
            action_pipeline=action_pipeline,
            eval_action_transform=eval_action_transform,
            obs_preprocessor=obs_preprocessor,
            auxiliary_update=auxiliary_update,
            extra_eval_metrics=extra_eval_metrics,
            extra_actor_loss_fn=extra_actor_loss_fn,
            extra_critic_loss_fn=extra_critic_loss_fn,
            reward_shaping_fn=reward_shaping_fn,
            extension_stack=extension_stack,
        )

    return build_resumable_train(
        init_fn=init_fn,
        make_scan_fn=make_scan_fn,
        num_updates=num_updates,
        init_transform=_init_transform if init_transform is not None else None,
    )
