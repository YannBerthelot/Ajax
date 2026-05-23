from collections.abc import Sequence
from dataclasses import fields
from math import floor
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial

from ajax.agents.cloning import (
    CloningConfig,
    get_cloning_args,
    get_pre_trained_agent,
)
from ajax.agents.SAC import core
from ajax.agents.SAC.state import SACConfig, SACState
from ajax.agents.SAC.utils import SquashedNormal
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    collect_experience,
    get_pi,
    init_collector_state,
    should_use_uniform_sampling,
)
from ajax.environments.utils import (
    check_env_is_gymnax,
    get_action_dim,
    get_state_action_shapes,
)
from ajax.extensions._sac_hooks import (
    make_action_pipeline,
    make_bc_loss_fn,
    make_eval_action_transform,
    make_next_expert_fn,
    make_policy_action_transform,
    make_policy_obs_preprocessor,
    make_runtime_maintenance,
)
from ajax.extensions.base import ExtensionContext, ExtensionStack
from ajax.log import evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.modules.expert import (
    augment_obs_if_needed,
    compute_behavior_kpis,
    compute_expert_diagnostics,
)
from ajax.modules.exploration import (
    EDGEAuxiliaries,
    compute_edge_diagnostics,
)
from ajax.modules.pretrain import (
    PhiRefreshAuxiliaries,
    collect_and_store_expert_transitions,
    pretrain_critic_bellman,
    pretrain_critic_mc,
    pretrain_critic_online_light,
)
from ajax.networks.networks import (
    get_initialized_actor_critic,
    get_initialized_critic,
    predict_value,
)
from ajax.perf_utils import build_resumable_train, final_aux_scan
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.types import BufferType

# Extension `name` attributes for the four target-mod extensions
# implemented in :mod:`ajax.extensions.target_mods`. Used by
# ``update_value_functions`` to decide whether to materialise
# ``q_preds_for_var`` (only ``MCVarianceCorrection`` actually needs it,
# but the legacy code path conservatively computed it whenever any
# target modifier was active — keep the same trigger set so the parity
# tolerance is undisturbed).
_TARGET_MOD_NAMES = frozenset(
    {
        "ibrl",
        "lcb_gated_bootstrap",
        "critic_blend",
        "mc_variance_correction",
    }
)


# ---------------------------------------------------------------------------
# Auxiliary dataclasses for logging
# ---------------------------------------------------------------------------


@struct.dataclass
class TemperatureAuxiliaries:
    alpha: jax.Array
    log_alpha: jax.Array
    effective_target_entropy: (
        jax.Array
    )  # actual target used in alpha update (distance-modulated when active)


@struct.dataclass
class PolicyAuxiliaries:
    # Core loss
    raw_loss: jax.Array  # α·log π - Q: pure SAC gradient
    policy_loss: jax.Array  # raw_loss + value constraint terms

    # Entropy diagnostics
    log_pi: jax.Array  # entropy proxy; tracks target_entropy
    policy_std: jax.Array  # mean unsquashed std; lower = more deterministic

    # Q-value diagnostics
    q_min: jax.Array  # Q(s, π(s)): what policy optimises
    q_expert: jax.Array  # Q(s, a_expert): expert value estimate

    # Expert diagnostics
    l2_expert: jax.Array  # ||π(s) - a_expert||^2: L2 distance to expert action
    above_expert_frac: jax.Array  # fraction of batch where policy beats expert

    # Online decaying BC term
    bc_term: jax.Array  # decaying online BC loss magnitude (0 after warmup_frac)

    # Policy behavior KPIs (from raw_obs — tell us what the policy actually does)
    altitude_error: jax.Array  # mean |z - target| over batch
    z_dot_mean: jax.Array  # mean |z_dot| over batch: 0 = stable, high = aggressive


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q_pred_min: jax.Array  # min over ensemble
    q_expert_mean: jax.Array  # critic's estimate of expert value
    q_gap: jax.Array  # q_expert - q_min: >0 = room to improve
    var_preds: jax.Array  # inter-critic variance
    expert_frac_in_buffer: jax.Array  # fraction of sampled batch flagged as expert
    alpha_blend: jax.Array  # current blend coefficient (1=pure expert, 0=pure Bellman)
    effective_threshold: jax.Array  # box threshold at current train_frac
    box_entry_rate: jax.Array  # fraction of batch inside value box
    mc_correction_frac: jax.Array  # fraction of batch where MC target replaced Bellman
    phi_star_q_gap_ood: (
        jax.Array
    )  # |Q_φ*(s,π*) - Q_φ(s,π*)| mean: φ* OOD coverage error


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    edge: EDGEAuxiliaries
    phi_refresh: PhiRefreshAuxiliaries


# ---------------------------------------------------------------------------
# Scalar alpha (temperature)
# ---------------------------------------------------------------------------


def create_alpha_train_state(
    learning_rate: float = 3e-4,
    alpha_init: float = 1.0,
) -> TrainState:
    return core.create_alpha_train_state(learning_rate, alpha_init)


# ---------------------------------------------------------------------------
# SAC initialization
# ---------------------------------------------------------------------------


def init_SAC(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    buffer: BufferType,
    window_size: int = 10,
    expert_policy: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    residual: bool = False,
    fixed_alpha: bool = False,
    max_timesteps: Optional[int] = None,
    num_critics: int = 2,
    expert_buffer_n_steps: int = 20_000,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
    expert_state_aug_dim: int = 0,
    pid_actor_config=None,
    action_dim_override: Optional[int] = None,
    extra_critic_head_names: Tuple[str, ...] = (),
    extra_critic_head_dims: Tuple[int, ...] = (),
    normalize_obs_running: bool = False,
    jsrl_curriculum: bool = False,
) -> SACState:
    rng, init_key, collector_key, expert_key = jax.random.split(key, num=4)

    # When augment_obs_with_expert_action=True, the actor and critic receive
    # obs augmented with a_expert at runtime (action_dim extra dimensions).
    # When augment_obs_with_expert_state=True, the obs is also augmented
    # with the flattened expert internal state (expert_state_aug_dim
    # extra dimensions). We must initialise the networks with the
    # matching inflated input size.
    extra_obs_dim = 0
    if augment_obs_with_expert_action:
        _, action_shape = get_state_action_shapes(env_args.env)
        extra_obs_dim += action_shape[0]
    if augment_obs_with_expert_state and expert_state_aug_dim > 0:
        extra_obs_dim += expert_state_aug_dim

    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        actor_optimizer_config=actor_optimizer_args,
        critic_optimizer_config=critic_optimizer_args,
        network_config=network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=num_critics,
        expert_policy=expert_policy,
        residual=False,
        fixed_alpha=False,
        max_timesteps=max_timesteps,
        extra_obs_dim=extra_obs_dim,
        pid_actor_config=pid_actor_config,
        action_dim_override=action_dim_override,
        extra_critic_head_names=extra_critic_head_names,
        extra_critic_head_dims=extra_critic_head_dims,
    )

    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
        window_size=window_size,
        max_timesteps=max_timesteps,
        action_dim_override=action_dim_override,
        expert_state_aug_dim=(
            expert_state_aug_dim if augment_obs_with_expert_state else 0
        ),
        normalize_obs_running=normalize_obs_running,
        include_expert_fields=expert_policy is not None,
    )
    if collector_state.obs_norm_info is not None:
        # Seed actor/critic with the initial (zero) stats so get_pi /
        # predict_value see a consistent obs_norm_info from step 1. The
        # field is updated each collection step.
        actor_state = actor_state.replace(obs_norm_info=collector_state.obs_norm_info)
        critic_state = critic_state.replace(obs_norm_info=collector_state.obs_norm_info)

    if (
        expert_policy is not None
        and buffer is not None
        and collector_state.buffer_state is not None
        and expert_buffer_n_steps > 0
    ):
        collector_state = collector_state.replace(
            buffer_state=collect_and_store_expert_transitions(
                expert_policy=expert_policy,
                env_args=env_args,
                buffer=buffer,
                buffer_state=collector_state.buffer_state,
                rng=expert_key,
                n_steps=expert_buffer_n_steps,
                max_timesteps=max_timesteps,
            )
        )

    # Seed batched expert state for stateful experts (e.g. PID integrator).
    # The collection pipeline resets this per-env when last_done is set.
    if expert_policy is not None and hasattr(expert_policy, "init_state"):
        collector_state = collector_state.replace(
            expert_state=expert_policy.init_state(env_args.n_envs)
        )

    # JSRL curriculum needs a per-env step_in_episode counter; initialize
    # to zeros. Other methods leave step_in_episode=None so the collector
    # update path skips the reset/increment logic.
    if jsrl_curriculum:
        collector_state = collector_state.replace(
            step_in_episode=jnp.zeros((env_args.n_envs,), dtype=jnp.int32)
        )

    alpha = create_alpha_train_state(**to_state_dict(alpha_args))

    return SACState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=alpha,
        collector_state=collector_state,
        lambda_param=1.0,
    )


# ---------------------------------------------------------------------------
# Critic update
# ---------------------------------------------------------------------------
# Uses core.compute_td_target for the base Bellman target, then applies
# expert target modifiers (IBRL, blend, MC correction) before the loss.


def update_value_functions(
    agent_state: SACState,
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float = 1.0,
    expert_q: Optional[jax.Array] = None,
    extension_stack: Optional[ExtensionStack] = None,
    total_timesteps: int = 1,
    augment_obs_with_expert_action: bool = False,
    extra_critic_loss_fn: Optional[Callable] = None,
    next_action_transform: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> Tuple[SACState, ValueAuxiliaries]:
    value_loss_key, rng = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])

    # 1. Core Bellman target (pure SAC). Pass next_action_transform so
    # residual RL evaluates the bootstrap critic at the same residual-
    # transformed action distribution the critic was trained on.
    target_q = core.compute_td_target(
        actor_state=agent_state.actor_state,
        critic_state=agent_state.critic_state,
        next_observations=next_observations,
        dones=dones,
        rewards=rewards,
        gamma=gamma,
        alpha=alpha,
        rng=value_loss_key,
        recurrent=recurrent,
        reward_scale=reward_scale,
        next_action_transform=next_action_transform,
        next_a_expert=next_a_expert,
    )

    # 2. Q predictions for expert-path diagnostics. Only computed when a
    # consumer exists (the target-mod ExtensionStack feeds them in via the
    # ``q_preds`` batch entry for MCVarianceCorrection, or expert_q is set
    # so q_gap can be reported). The gradient-bearing pass inside
    # critic_loss_fn already exposes var_preds via core_aux.
    has_target_mods = extension_stack is not None and any(
        ext.name in _TARGET_MOD_NAMES for ext in extension_stack.extensions
    )
    needs_expert_q_preds = has_target_mods or expert_q is not None
    if needs_expert_q_preds:
        q_preds_for_var = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.params,
            x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
        )

    # 3. Expert target modifiers fold through the ExtensionStack
    # (IBRL → LCBGatedBootstrap → CriticBlend → MCVarianceCorrection).
    # The two diagnostic scalars `alpha_blend_logged` and
    # `mc_correction_frac` that used to be threaded out of the
    # `make_target_modifier` callable are dropped for now — they are pure
    # observability (not exercised by parity / equivalence tests) and
    # will be re-added cleanly via the `eval_metrics` phase. See
    # Phase 2b commit notes.
    alpha_blend_logged = jnp.zeros(1)
    mc_correction_frac = jnp.zeros(1)
    if has_target_mods:
        # `has_target_mods` already asserts `extension_stack is not
        # None` — assert it again for mypy.
        assert extension_stack is not None
        batch = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
            "dones": dones,
            "rng_key": value_loss_key,
            "q_preds": q_preds_for_var,
            "gamma": gamma,
            "augment_obs_with_expert_action": augment_obs_with_expert_action,
            "recurrent": recurrent,
        }
        ctx = ExtensionContext(
            step=agent_state.collector_state.timestep,
            rng=value_loss_key,
            total_steps=total_timesteps,
        )
        target_q = extension_stack.on_target(
            agent_state, agent_state.ext_state, batch, target_q, ctx
        )

    # 4. Core critic loss (MSE against composed target), optionally augmented
    #    with an extra loss term. The hook receives
    #    (params, critic_state, obs, act, target_q) -> scalar so it can
    #    recompute the batch residual (e.g. an EVarEst variance penalty).
    #    The extra loss is added inside the same value_and_grad so the single
    #    Adam step sees a combined gradient direction (coeff matters).
    def _critic_loss(params, critic_state, obs, act, tgt):
        loss, core_aux = core.critic_loss_fn(params, critic_state, obs, act, tgt)
        if extra_critic_loss_fn is not None:
            loss = loss + extra_critic_loss_fn(params, critic_state, obs, act, tgt)
        return loss, core_aux

    (loss, core_aux), grads = jax.value_and_grad(_critic_loss, has_aux=True)(
        agent_state.critic_state.params,
        agent_state.critic_state,
        observations,
        actions,
        target_q,
    )

    # 5. Assemble full ValueAuxiliaries with expert diagnostics.
    # q_pred_min_full only feeds q_gap, which is itself gated on expert_q.
    if expert_q is not None:
        q_pred_min_full = jnp.min(q_preds_for_var, axis=0)
        q_expert_mean = expert_q.mean().flatten()
        q_gap = (expert_q - q_pred_min_full).mean().flatten()
    else:
        q_expert_mean = jnp.zeros(1)
        q_gap = jnp.zeros(1)

    aux = ValueAuxiliaries(
        critic_loss=core_aux.critic_loss,
        q_pred_min=core_aux.q_pred_min,
        q_expert_mean=q_expert_mean,
        q_gap=q_gap,
        var_preds=core_aux.var_preds,
        alpha_blend=alpha_blend_logged,
        effective_threshold=jnp.zeros(1),
        box_entry_rate=jnp.zeros(1),
        expert_frac_in_buffer=jnp.zeros(1),
        mc_correction_frac=mc_correction_frac,
        phi_star_q_gap_ood=jnp.zeros(1),
    )

    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)
    return agent_state.replace(rng=rng, critic_state=updated_critic_state), aux


# ---------------------------------------------------------------------------
# Policy update — SAC + value constraint
# ---------------------------------------------------------------------------


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    alpha: jax.Array,
    rng: jax.random.PRNGKey,
    raw_observations: Optional[jax.Array] = None,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    a_expert_precomputed: Optional[jax.Array] = None,
    train_frac: Optional[jax.Array] = None,
    expert_critic_params: Optional[Any] = None,
    expert_v_min: Optional[jax.Array] = None,
    expert_v_max: Optional[jax.Array] = None,
    # Composed policy modifiers (replace 6 boolean flags)
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    bc_loss_fn: Optional[Callable] = None,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """SAC actor loss with composable expert modifiers.

    Structure mirrors the critic side: core SAC loss + layered expert additions.
    1. Pre-process: obs_preprocessor (detach expert-action dims)
    2. Core: forward pass → sample → Q eval → α·log π - Q
    3. Modifier: policy_action_transform (residual RL before Q eval)
    4. Modifier: bc_loss_fn (online decaying BC term)
    5. Diagnostics: expert Q gap, L2 distance, behavior KPIs
    """
    _raw_obs = (
        raw_observations if raw_observations is not None else observations[..., :-1]
    )

    # 1. Pre-process: optionally detach expert-action dims in augmented obs
    obs_for_actor = (
        obs_preprocessor(observations) if obs_preprocessor is not None else observations
    )

    # 2. Core forward pass + sample
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=obs_for_actor,
        done=dones,
        recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = pi.sample_and_log_prob(seed=sample_key)
    log_probs = log_probs.sum(-1, keepdims=True)

    policy_std = (
        pi.unsquashed_stddev().mean()
        if isinstance(pi, SquashedNormal)
        else pi.stddev().mean()
    )

    # 3. Action transform modifier (residual RL)
    q_input_actions = (
        policy_action_transform(actions, _raw_obs, a_expert_precomputed)
        if policy_action_transform is not None
        else actions
    )

    # Core Q evaluation and SAC loss
    q_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_states.params,
        x=jnp.concatenate([observations, q_input_actions], axis=-1),
    )
    q_min = jnp.min(q_preds, axis=0)
    loss_actor = alpha * log_probs - q_min

    # 4. Expert diagnostics and BC loss
    needs_expert = expert_policy is not None and use_expert_guidance
    needs_bc = bc_loss_fn is not None and expert_critic_params is not None

    if needs_expert or needs_bc:
        a_expert = (
            a_expert_precomputed
            if a_expert_precomputed is not None
            else jax.lax.stop_gradient(expert_policy(_raw_obs))
        )
    else:
        a_expert = None

    # Expert diagnostics (no gradient effect)
    if needs_expert:
        q_expert_logged, l2_expert_logged, above_expert_frac = (
            compute_expert_diagnostics(
                critic_states,
                observations,
                q_min,
                a_expert,
                pi.distribution.loc,
            )
        )
    else:
        l2_expert_logged = jnp.zeros(())
        q_expert_logged = jnp.zeros(())
        above_expert_frac = jnp.zeros(())

    # Online decaying BC term
    if needs_bc:
        bc_term = bc_loss_fn(
            pi.distribution.loc,
            a_expert,
            critic_states,
            expert_critic_params,
            observations,
            train_frac,
            expert_v_min,
            expert_v_max,
        )
    else:
        bc_term = jnp.zeros(())

    total_loss = loss_actor.mean() + bc_term

    # 5. Behavior KPIs
    altitude_error_val, z_dot_mean_val = compute_behavior_kpis(
        _raw_obs,
        altitude_obs_idx,
        target_obs_idx,
    )

    return total_loss, PolicyAuxiliaries(
        policy_loss=total_loss,
        log_pi=log_probs.mean(),
        policy_std=policy_std,
        q_min=q_min.mean(),
        q_expert=q_expert_logged,
        l2_expert=l2_expert_logged,
        above_expert_frac=above_expert_frac,
        altitude_error=altitude_error_val,
        z_dot_mean=z_dot_mean_val,
        raw_loss=loss_actor.mean(),
        bc_term=bc_term,
    )


def update_policy(
    agent_state: SACState,
    observations: jax.Array,
    done: Optional[jax.Array],
    recurrent: bool,
    raw_observations: jax.Array,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    a_expert_precomputed: Optional[jax.Array] = None,
    train_frac: Optional[jax.Array] = None,
    # Composed policy modifiers
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    bc_loss_fn: Optional[Callable] = None,
    extra_actor_loss_fn: Optional[Callable] = None,
) -> Tuple[SACState, PolicyAuxiliaries, jax.Array]:
    """Returns (new_state, aux, log_probs) — log_probs reused by update_temperature
    to avoid a redundant actor forward pass."""
    rng, policy_key = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])

    def _actor_loss(params, *args, **kwargs):
        loss, aux = policy_loss_function(params, *args, **kwargs)
        if extra_actor_loss_fn is not None:
            loss = loss + extra_actor_loss_fn(params, agent_state.actor_state)
        return loss, aux

    (loss, aux), grads = jax.value_and_grad(_actor_loss, has_aux=True, argnums=0)(
        agent_state.actor_state.params,
        agent_state.actor_state,
        agent_state.critic_state,
        observations,
        done,
        recurrent,
        alpha,
        policy_key,
        raw_observations=raw_observations,
        expert_policy=expert_policy,
        use_expert_guidance=use_expert_guidance,
        altitude_obs_idx=altitude_obs_idx,
        target_obs_idx=target_obs_idx,
        a_expert_precomputed=a_expert_precomputed,
        train_frac=train_frac,
        expert_critic_params=agent_state.expert_critic_params,
        expert_v_min=agent_state.expert_v_min,
        expert_v_max=agent_state.expert_v_max,
        obs_preprocessor=obs_preprocessor,
        policy_action_transform=policy_action_transform,
        bc_loss_fn=bc_loss_fn,
    )

    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)

    # Recompute log_probs from updated actor for temperature update reuse
    temp_rng, temp_sample_key = jax.random.split(rng)
    pi, _ = get_pi(
        actor_state=updated_actor_state,
        actor_params=updated_actor_state.params,
        obs=observations,
        done=done,
        recurrent=recurrent,
    )
    _, log_probs = pi.sample_and_log_prob(seed=temp_sample_key)
    return (
        agent_state.replace(rng=temp_rng, actor_state=updated_actor_state),
        aux,
        jax.lax.stop_gradient(log_probs),
    )


# ---------------------------------------------------------------------------
# Temperature update with adaptive target entropy
# ---------------------------------------------------------------------------


def temperature_loss_function(
    log_alpha_params: FrozenDict,
    corrected_log_probs: jax.Array,
    effective_target_entropy: jax.Array,
) -> Tuple[jax.Array, TemperatureAuxiliaries]:
    loss, core_aux = core.temperature_loss_fn(
        log_alpha_params,
        corrected_log_probs,
        effective_target_entropy,
    )
    return loss, TemperatureAuxiliaries(
        alpha=core_aux.alpha,
        log_alpha=core_aux.log_alpha,
        effective_target_entropy=core_aux.effective_target_entropy,
    )


def update_temperature(
    agent_state: SACState,
    log_probs: jax.Array,
    effective_target_entropy: jax.Array,
) -> Tuple[SACState, TemperatureAuxiliaries]:
    """Standard SAC temperature update."""
    (loss, aux), grads = jax.value_and_grad(temperature_loss_function, has_aux=True)(
        agent_state.alpha.params,
        log_probs.sum(-1),
        effective_target_entropy,
    )
    new_alpha_state = agent_state.alpha.apply_gradients(grads=grads)
    return agent_state.replace(alpha=new_alpha_state), jax.lax.stop_gradient(aux)


# ---------------------------------------------------------------------------
# Target network update
# ---------------------------------------------------------------------------


def update_target_networks(agent_state: SACState, tau: float) -> SACState:
    return agent_state.replace(
        critic_state=agent_state.critic_state.soft_update(tau=tau)
    )


# ---------------------------------------------------------------------------
# Agent update (one gradient step)
# ---------------------------------------------------------------------------


def update_agent(
    agent_state: SACState,
    _: Any,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    action_dim: int,
    target_entropy: float,
    tau: float,
    num_critic_updates: int = 1,
    reward_scale: float = 1.0,
    additional_transition: Optional[Any] = None,
    transition_mix_fraction: float = 1.0,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    expert_mix_fraction: float = 0.1,
    box_threshold: float = 500.0,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
    total_timesteps: int = 1,
    target_entropy_far: Optional[float] = None,
    target_entropy_initial: Optional[float] = None,
    target_entropy_ramp_frac: float = 0.5,
    exploration_tau: float = 1.0,
    # Composed modules
    extension_stack: Optional[ExtensionStack] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    bc_loss_fn: Optional[Callable] = None,
    extra_actor_loss_fn: Optional[Callable] = None,
    extra_critic_loss_fn: Optional[Callable] = None,
    # Optional Hindsight-Experience-Replay relabel. Signature
    # (rng, transition) -> transition. Applied to the finalised sampled
    # batch before any expert mixing / learner use. Default None ⇒ the
    # batch is untouched and this path is byte-identical to before.
    her_relabel_fn: Optional[Callable] = None,
) -> Tuple[SACState, AuxiliaryLogs]:
    sample_key, expert_sample_key, rng = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=rng)

    # --- Sample from buffer ---
    if buffer is not None and agent_state.collector_state.buffer_state is not None:
        (
            observations,
            terminated,
            truncated,
            next_observations,
            rewards,
            actions,
            raw_observations,
            is_expert,
        ) = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, sample_key
        )
        # Aligned a_expert / next_a_expert (same sample_key, same
        # slices) when the buffer was initialised with expert fields.
        if expert_policy is not None:
            from ajax.buffers.utils import get_expert_fields_from_buffer

            a_expert_buf, next_a_expert_buf = get_expert_fields_from_buffer(
                buffer, agent_state.collector_state.buffer_state, sample_key
            )
        else:
            a_expert_buf, next_a_expert_buf = None, None
        expert_frac_in_buffer = is_expert.mean()
        original_transition = Transition(
            observations,
            actions,
            rewards,
            terminated,
            truncated,
            next_observations,
            raw_obs=raw_observations,
            a_expert=a_expert_buf,
            next_a_expert=next_a_expert_buf,
        )

        if additional_transition is not None and transition_mix_fraction < 1.0:
            len_original = len(observations)
            n_from_buffer = floor(transition_mix_fraction * len_original)
            n_from_online = len_original - n_from_buffer
            # Generate sample indices once and reuse across leaves: the
            # previous form called jax.random.choice once per leaf with
            # the same sample_key, which produces identical indices per
            # leaf but pays the index-generation cost N_leaves times.
            mix_idx = jax.random.randint(sample_key, (n_from_online,), 0, len_original)
            additional_transition = jax.tree.map(
                lambda x: x[mix_idx],
                additional_transition,
            )
            transition = jax.tree.map(
                lambda x, y: (
                    None
                    if (x is None or y is None)
                    else jnp.concatenate([x[:n_from_buffer], y], axis=0)
                ),
                original_transition,
                additional_transition,
                is_leaf=lambda x: x is None,
            )
        else:
            transition = original_transition

    elif additional_transition is not None:
        transition = additional_transition
        expert_frac_in_buffer = jnp.zeros(())
    else:
        raise ValueError("Either buffer or additional_transition must be provided.")

    # --- Hindsight Experience Replay relabel ---
    # Opt-in. The caller-supplied fn rewrites the goal portion of
    # obs/next_obs and recomputes reward in closed form, turning
    # arbitrary achieved outcomes into on-target successes. No-op when
    # her_relabel_fn is None.
    if her_relabel_fn is not None:
        rng, her_key = jax.random.split(rng)
        agent_state = agent_state.replace(rng=rng)
        transition = her_relabel_fn(her_key, transition)

    # --- Expert batch mixing ---
    if expert_mix_fraction > 0.0 and expert_policy is not None:
        (
            exp_obs,
            exp_terminated,
            exp_truncated,
            exp_next_obs,
            exp_rewards,
            exp_actions,
            exp_raw_obs,
            _,
        ) = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, expert_sample_key
        )
        from ajax.buffers.utils import get_expert_fields_from_buffer

        exp_a_expert, exp_next_a_expert = get_expert_fields_from_buffer(
            buffer, agent_state.collector_state.buffer_state, expert_sample_key
        )

        n_total = transition.obs.shape[0]
        n_expert = floor(expert_mix_fraction * n_total)
        n_online = n_total - n_expert

        def _cat(a, b):
            if a is None or b is None:
                return a
            return jnp.concatenate([a[:n_online], b[:n_expert]], axis=0)

        transition = Transition(
            obs=_cat(transition.obs, exp_obs),
            action=_cat(transition.action, exp_actions),
            reward=_cat(transition.reward, exp_rewards),
            terminated=_cat(transition.terminated, exp_terminated),
            truncated=_cat(transition.truncated, exp_truncated),
            next_obs=_cat(transition.next_obs, exp_next_obs),
            raw_obs=_cat(transition.raw_obs, exp_raw_obs),
            a_expert=_cat(transition.a_expert, exp_a_expert),
            next_a_expert=_cat(transition.next_a_expert, exp_next_a_expert),
        )

    dones = jnp.logical_or(transition.terminated, transition.truncated)

    # --- Obs augmentation: append a_expert to obs and next_obs ---
    # Must happen before any network call (critic, actor, policy loss).
    # raw_obs gives the env observations without train_frac, which is what
    # expert_policy expects. For next_obs we strip the last dim (train_frac).
    if augment_obs_with_expert_action and expert_policy is not None:
        _raw = (
            transition.raw_obs
            if transition.raw_obs is not None
            else transition.obs[..., :-1]
        )
        _raw_next = transition.next_obs[..., :-1]  # strip train_frac
        aug_obs = augment_obs_if_needed(transition.obs, _raw, expert_policy, True)
        aug_next_obs = augment_obs_if_needed(
            transition.next_obs, _raw_next, expert_policy, True
        )
        transition = transition.replace(obs=aug_obs, next_obs=aug_next_obs)

    # --- Pre-compute Q(s, a_expert) and a_expert once for critic logging + policy ---
    # Avoids computing expert_policy twice (once here, once inside policy_loss_function)
    expert_q = None
    a_expert_precomputed = None
    needs_expert = expert_policy is not None and (
        use_expert_guidance
        or policy_action_transform is not None
        or bc_loss_fn is not None
    )
    if needs_expert:
        _raw = (
            transition.raw_obs
            if transition.raw_obs is not None
            else transition.obs[..., :-1]
        )
        # Prefer the a_expert stored in the buffer at collection time
        # (computed with the correct stateful expert internal state).
        # Fall back to a fresh stateless expert call only if the buffer
        # transition predates the schema change (legacy run).
        if transition.a_expert is not None:
            a_expert_precomputed = jax.lax.stop_gradient(transition.a_expert)
        else:
            a_expert_precomputed = jax.lax.stop_gradient(expert_policy(_raw))
        # transition.obs is already augmented at this point if augment_obs_with_expert_action
        expert_q = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.critic_state.params,
                    x=jnp.concatenate([transition.obs, a_expert_precomputed], axis=-1),
                ),
                axis=0,
            )
        )

    # --- φ* OOD quality: |Q_φ*(s,π*) - Q_φ(s,π*)| on training batch ---
    # Measures how much frozen φ* disagrees with the live critic on expert actions.
    # Non-zero only when MC pretrain has been run (expert_critic_params is not None).
    phi_star_q_gap_ood = jnp.zeros(())
    if (
        agent_state.expert_critic_params is not None
        and expert_q is not None
        and a_expert_precomputed is not None
    ):
        q_phi_star = jax.lax.stop_gradient(
            jnp.min(
                predict_value(
                    critic_state=agent_state.critic_state,
                    critic_params=agent_state.expert_critic_params,
                    x=jnp.concatenate([transition.obs, a_expert_precomputed], axis=-1),
                ),
                axis=0,
            )
        )
        phi_star_q_gap_ood = jnp.abs(
            q_phi_star - jax.lax.stop_gradient(expert_q)
        ).mean()

    # --- Critic updates ---
    # For residual RL: apply the same residual transform to the
    # bootstrap action in the TD target so the critic is queried in-
    # distribution. policy_action_transform expects a_expert at s_t,
    # but at the target it must use a_expert at s_{t+1} = next_a_expert.
    _next_action_transform = policy_action_transform
    _next_a_expert_for_target = (
        transition.next_a_expert
        if transition.next_a_expert is not None and policy_action_transform is not None
        else None
    )

    def _one_critic_update(s):
        return update_value_functions(
            observations=transition.obs,
            actions=transition.action,
            next_observations=transition.next_obs,
            rewards=transition.reward,
            dones=dones,
            agent_state=s,
            recurrent=recurrent,
            gamma=gamma,
            reward_scale=reward_scale,
            expert_q=expert_q,
            extension_stack=extension_stack,
            total_timesteps=total_timesteps,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            extra_critic_loss_fn=extra_critic_loss_fn,
            next_action_transform=_next_action_transform,
            next_a_expert=_next_a_expert_for_target,
        )

    # See ajax.perf_utils.final_aux_scan: carry-only scan that exposes
    # last-step aux without materialising the full ys axis.
    def critic_update_step(state, _):
        return _one_critic_update(state)

    agent_state, aux_value = final_aux_scan(
        critic_update_step,
        agent_state,
        length=num_critic_updates,
    )

    # --- Policy update — returns log_probs for temperature reuse ---
    train_frac = agent_state.collector_state.timestep / total_timesteps
    new_agent_state, aux_policy, policy_log_probs = update_policy(
        observations=transition.obs,
        done=dones,
        agent_state=agent_state,
        recurrent=recurrent,
        raw_observations=transition.raw_obs,
        expert_policy=expert_policy,
        use_expert_guidance=use_expert_guidance,
        altitude_obs_idx=altitude_obs_idx,
        target_obs_idx=target_obs_idx,
        a_expert_precomputed=a_expert_precomputed,
        train_frac=train_frac,
        obs_preprocessor=obs_preprocessor,
        policy_action_transform=policy_action_transform,
        bc_loss_fn=bc_loss_fn,
        extra_actor_loss_fn=extra_actor_loss_fn,
    )
    agent_state = jax.lax.cond(
        agent_state.collector_state.timestep >= policy_update_start,
        lambda: new_agent_state,
        lambda: agent_state,
    )

    # --- Distance-modulated entropy target ---
    if target_entropy_far is not None and transition.raw_obs is not None:
        raw_distance = jnp.abs(
            transition.raw_obs[..., altitude_obs_idx]
            - transition.raw_obs[..., target_obs_idx]
        ).mean()
        distance_frac = jnp.clip(raw_distance / box_threshold, 0.0, 1.0)
        effective_target_entropy = (
            target_entropy * (1.0 - distance_frac) + target_entropy_far * distance_frac
        )
    elif target_entropy_initial is not None:
        # Time-based ramp from `target_entropy_initial` (low, near
        # -dim*1) to `target_entropy` over the first
        # `target_entropy_ramp_frac` of training. Stays at the standard
        # target after the ramp completes. Used to keep alpha low at
        # the start of training so the (BC-warmed) policy mean rolls
        # out near-deterministically and the agent stays alive on
        # brittle envs; exploration grows as the policy learns.
        ramp_horizon = jnp.maximum(
            target_entropy_ramp_frac * jnp.asarray(total_timesteps, jnp.float32),
            1.0,
        )
        progress = jnp.clip(
            agent_state.collector_state.timestep / ramp_horizon, 0.0, 1.0
        )
        effective_target_entropy = (
            1.0 - progress
        ) * target_entropy_initial + progress * target_entropy
    else:
        effective_target_entropy = jnp.asarray(target_entropy)

    # --- Temperature update — reuses log_probs, no redundant actor forward pass ---
    new_agent_state_temp, aux_temperature = update_temperature(
        agent_state,
        log_probs=policy_log_probs,
        effective_target_entropy=effective_target_entropy,
    )
    agent_state = jax.lax.cond(
        agent_state.collector_state.timestep >= alpha_update_start,
        lambda: new_agent_state_temp,
        lambda: agent_state,
    )

    agent_state = update_target_networks(agent_state, tau=tau)

    # --- EDGE diagnostics (computed on training batch) ---
    edge_aux = compute_edge_diagnostics(
        aux_value.q_gap,
        aux_value.q_pred_min,
        exploration_tau,
        expert_frac_in_buffer,
    )

    # Direct field access instead of to_state_dict — avoids serialization overhead
    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=ValueAuxiliaries(
            critic_loss=aux_value.critic_loss.flatten(),
            q_pred_min=aux_value.q_pred_min.flatten(),
            q_expert_mean=aux_value.q_expert_mean.flatten(),
            q_gap=aux_value.q_gap.flatten(),
            var_preds=aux_value.var_preds.flatten(),
            alpha_blend=aux_value.alpha_blend.flatten(),
            effective_threshold=jnp.zeros(()),
            box_entry_rate=jnp.zeros(()),
            expert_frac_in_buffer=jnp.atleast_1d(expert_frac_in_buffer),
            mc_correction_frac=aux_value.mc_correction_frac.flatten(),
            phi_star_q_gap_ood=jnp.atleast_1d(phi_star_q_gap_ood),
        ),
        edge=edge_aux,
        phi_refresh=PhiRefreshAuxiliaries(
            loss_before=jnp.zeros(1),
            loss_after=jnp.zeros(1),
            expert_buffer_size=jnp.zeros(1),
        ),
    )
    return agent_state, aux


# ---------------------------------------------------------------------------
# Training iteration
# ---------------------------------------------------------------------------


def training_iteration(
    agent_state: SACState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    buffer: BufferType,
    agent_config: SACConfig,
    action_dim: int,
    total_timesteps: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: int = 1000,
    horizon: int = 10000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
    n_epochs: int = 1,
    transition_mix_fraction: float = 1.0,
    expert_policy: Optional[Callable] = None,  # used for training
    eval_expert_policy: Optional[Callable] = None,  # used for eval logging only
    use_expert_guidance: bool = True,
    action_scale: float = 1.0,
    early_termination_condition: Optional[Callable] = None,
    num_critic_updates: int = 1,
    expert_mix_fraction: float = 0.1,
    box_threshold: float = 500.0,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
    detach_obs_aug_action: bool = False,
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    exploration_tau: float = 1.0,
    target_entropy_far: Optional[float] = None,
    target_entropy_initial: Optional[float] = None,
    target_entropy_ramp_frac: float = 0.5,
    # Composed modules (replace boolean flags)
    action_pipeline: Optional[Callable] = None,
    extension_stack: Optional[ExtensionStack] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    bc_loss_fn: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    runtime_maintenance: Optional[Callable] = None,
    extra_actor_loss_fn: Optional[Callable] = None,
    extra_critic_loss_fn: Optional[Callable] = None,
    her_relabel_fn: Optional[Callable] = None,
    auxiliary_update: Optional[Callable] = None,
    extra_eval_metrics: Optional[Callable] = None,
    pid_gain_policy: bool = False,
    next_expert_fn: Optional[Callable] = None,
    # API compat
    imitation_coef: float = 0.0,
    distance_to_stable: Callable = lambda x: 1.0,
    imitation_coef_offset: float = 0.0,
    # Eval-suppression mode: when True, evaluate_and_log only fires evals
    # in the last 20% of training. Use for HPO phases where the only
    # number that matters is the final-window IQM.
    sweep: bool = False,
) -> tuple[SACState, None]:
    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_config.learning_starts)

    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        buffer=buffer,
        uniform=uniform,
        action_pipeline=action_pipeline,
        next_expert_fn=next_expert_fn,
    )

    agent_state, transition = collect_scan_fn(agent_state, None)
    timestep = agent_state.collector_state.timestep

    def do_update(agent_state):
        # Periodic φ* refresh via composed runtime maintenance
        _zero_phi = PhiRefreshAuxiliaries(
            loss_before=jnp.zeros(1),
            loss_after=jnp.zeros(1),
            expert_buffer_size=jnp.zeros(1),
        )
        if runtime_maintenance is not None:
            agent_state, phi_refresh_aux = runtime_maintenance(agent_state)
        else:
            phi_refresh_aux = _zero_phi

        update_scan_fn = partial(
            update_agent,
            buffer=buffer,
            recurrent=recurrent,
            gamma=agent_config.gamma,
            action_dim=action_dim,
            target_entropy=agent_config.target_entropy,
            tau=agent_config.tau,
            reward_scale=agent_config.reward_scale,
            additional_transition=(
                transition if transition_mix_fraction < 1.0 else None
            ),
            transition_mix_fraction=transition_mix_fraction,
            expert_policy=expert_policy,
            use_expert_guidance=use_expert_guidance,
            policy_update_start=policy_update_start,
            alpha_update_start=alpha_update_start,
            num_critic_updates=num_critic_updates,
            expert_mix_fraction=expert_mix_fraction,
            box_threshold=box_threshold,
            altitude_obs_idx=altitude_obs_idx,
            target_obs_idx=target_obs_idx,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            augment_obs_with_expert_state=augment_obs_with_expert_state,
            total_timesteps=total_timesteps,
            target_entropy_far=target_entropy_far,
            target_entropy_initial=target_entropy_initial,
            target_entropy_ramp_frac=target_entropy_ramp_frac,
            exploration_tau=exploration_tau,
            extension_stack=extension_stack,
            obs_preprocessor=obs_preprocessor,
            policy_action_transform=policy_action_transform,
            bc_loss_fn=bc_loss_fn,
            extra_actor_loss_fn=extra_actor_loss_fn,
            extra_critic_loss_fn=extra_critic_loss_fn,
            her_relabel_fn=her_relabel_fn,
        )
        agent_state, aux = jax.lax.scan(
            update_scan_fn, agent_state, xs=None, length=n_epochs
        )
        aux = jax.tree.map(lambda x: x[-1].reshape((1,)), aux)
        aux = aux.replace(
            value=ValueAuxiliaries(
                critic_loss=aux.value.critic_loss.flatten(),
                q_pred_min=aux.value.q_pred_min.flatten(),
                q_expert_mean=aux.value.q_expert_mean.flatten(),
                q_gap=aux.value.q_gap.flatten(),
                var_preds=aux.value.var_preds.flatten(),
                alpha_blend=aux.value.alpha_blend.flatten(),
                effective_threshold=aux.value.effective_threshold.flatten(),
                box_entry_rate=aux.value.box_entry_rate.flatten(),
                expert_frac_in_buffer=aux.value.expert_frac_in_buffer.flatten(),
                mc_correction_frac=aux.value.mc_correction_frac.flatten(),
                phi_star_q_gap_ood=aux.value.phi_star_q_gap_ood.flatten(),
            ),
            edge=EDGEAuxiliaries(
                value_gap=aux.edge.value_gap.flatten(),
                p_expert_mean=aux.edge.p_expert_mean.flatten(),
                expert_action_fraction=aux.edge.expert_action_fraction.flatten(),
                # Pull the live gating diag from the collector state — these
                # were stashed by collect_experience after the latest action
                # pipeline call. flatten/atleast_1d for tb-flatten parity.
                live_expert_frac=jnp.atleast_1d(
                    agent_state.collector_state.last_expert_frac
                ),
                live_q_advantage=jnp.atleast_1d(
                    agent_state.collector_state.last_q_advantage
                ),
                live_critic_sigma_actor=jnp.atleast_1d(
                    agent_state.collector_state.last_critic_sigma_actor
                ),
                live_critic_sigma_expert=jnp.atleast_1d(
                    agent_state.collector_state.last_critic_sigma_expert
                ),
                live_p_expert_max=jnp.atleast_1d(
                    agent_state.collector_state.last_p_expert_max
                ),
            ),
            # Override the zeros from update_agent with the actual refresh diagnostics
            phi_refresh=PhiRefreshAuxiliaries(
                loss_before=phi_refresh_aux.loss_before.flatten(),
                loss_after=phi_refresh_aux.loss_after.flatten(),
                expert_buffer_size=phi_refresh_aux.expert_buffer_size.flatten(),
            ),
        )
        return agent_state, aux

    def fill_with_nan(dataclass):
        nan = jnp.ones(1) * jnp.nan
        result = {}
        for field in fields(dataclass):
            sub = field.type
            if hasattr(sub, "__dataclass_fields__"):
                result[field.name] = fill_with_nan(sub)
            else:
                result[field.name] = nan
        return dataclass(**result)

    def skip_update(agent_state):
        return agent_state, fill_with_nan(AuxiliaryLogs)

    agent_state, aux = jax.lax.cond(
        timestep >= agent_config.learning_starts,
        do_update,
        skip_update,
        operand=agent_state,
    )

    if auxiliary_update is not None:
        aux_rng, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        agent_state, _auxiliary_metrics = auxiliary_update(agent_state, aux_rng)

    # Obs augmentation now happens inside evaluate.step_environment, where
    # the per-step expert_state is already threaded through the scan
    # carry. The previous apply_fn-wrapping approach silently used a
    # stateless expert call, which produced a different augmented obs at
    # eval than at training for stateful experts (PID etc.).
    _eval_agent_state, metrics_to_log = evaluate_and_log(
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
        sweep=sweep,
        expert_policy=eval_expert_policy,
        action_scale=action_scale,
        early_termination_condition=early_termination_condition,
        train_frac=agent_state.collector_state.train_time_fraction,
        eval_action_transform=eval_action_transform,
        extra_eval_metrics=extra_eval_metrics,
        pid_gain_policy=pid_gain_policy,
        augment_obs_with_expert_action=augment_obs_with_expert_action,
        augment_obs_with_expert_state=augment_obs_with_expert_state,
    )
    # Keep the original agent_state (with original apply_fn) for training
    agent_state = agent_state.replace(
        eval_rng=_eval_agent_state.eval_rng,
        n_logs=_eval_agent_state.n_logs,
    )

    return agent_state, metrics_to_log


# ---------------------------------------------------------------------------
# Extension-stack → make_train kwargs translation
# ---------------------------------------------------------------------------


def _resolve_extension_stack(
    extensions: Sequence,
) -> dict:
    """Translate an :class:`ExtensionStack` into ``make_train`` kwargs.

    Plain SAC corresponds to an empty stack ⇒ this returns ``{}`` ⇒
    every legacy flag stays at its default ⇒ byte-identical behaviour to
    the pre-refactor ``train_SAC.py`` (the parity gate).

    Each known extension type maps onto the existing flag(s) it
    represents; unknown extensions are ignored at this layer (they may
    still drive the cleanly-folded phases — see ``training_iteration``).
    The mapping is deliberately a flat dict so an agent's ``make_train``
    keyword surface stays the single source of truth.
    """
    from ajax.extensions.expert import (
        ExpertGuidance,
        ExpertObsAugmentation,
        JSRLCurriculum,
        OnlineBC,
        ResidualPolicy,
        first_of_type,
    )
    from ajax.extensions.exploration import EDGEExploration
    from ajax.extensions.pretrain import (
        BellmanPretrain,
        MCPretrain,
        PhiRefresh,
    )
    from ajax.extensions.target_mods import (
        IBRL,
        CriticBlend,
        LCBGatedBootstrap,
        ValueBox,
    )

    out: dict = {}
    eg = first_of_type(extensions, ExpertGuidance)
    if eg is not None:
        out["expert_policy"] = eg.expert_policy
        out["use_expert_guidance"] = eg.use_expert_guidance
        out["expert_fraction"] = eg.expert_fraction
        out["expert_buffer_n_steps"] = eg.expert_buffer_n_steps
        out["expert_mix_fraction"] = eg.expert_mix_fraction

    obs_aug = first_of_type(extensions, ExpertObsAugmentation)
    if obs_aug is not None:
        out["augment_obs_with_expert_action"] = True
        out["detach_obs_aug_action"] = obs_aug.detach
        out.setdefault("expert_policy", obs_aug.expert_policy)

    bc = first_of_type(extensions, OnlineBC)
    if bc is not None:
        out["use_online_bc"] = True
        out["bc_coef"] = bc.bc_coef
        out["critic_warmup_frac"] = bc.critic_warmup_frac
        out.setdefault("expert_policy", bc.expert_policy)
    else:
        # ``train_SAC.py`` defaults ``use_online_bc=True`` (the BC term is
        # active iff an expert + MC pre-training are also present, so this
        # default is harmless when no expert is configured). Mirror that.
        out.setdefault("use_online_bc", True)

    res = first_of_type(extensions, ResidualPolicy)
    if res is not None:
        out["use_residual_rl"] = True
        out["residual_scale"] = res.scale
        out.setdefault("expert_policy", res.expert_policy)

    jsrl = first_of_type(extensions, JSRLCurriculum)
    if jsrl is not None:
        out["jsrl_curriculum"] = True
        out["jsrl_episode_length"] = jsrl.episode_length
        out["jsrl_decay_frac"] = jsrl.decay_frac
        out.setdefault("expert_policy", jsrl.expert_policy)

    edge = first_of_type(extensions, EDGEExploration)
    if edge is not None:
        out["use_expert_guided_exploration"] = True
        out["exploration_decay_frac"] = edge.decay_frac
        out["exploration_tau"] = edge.tau
        out["fixed_exploration_prob"] = edge.fixed_prob
        out["lcb_beta_init"] = edge.lcb_beta_init
        out["lcb_beta_decay_k"] = edge.lcb_beta_decay_k
        out["lcb_temperature"] = edge.lcb_temperature
        out["lcb_asymmetric"] = edge.lcb_asymmetric
        out["epsilon_floor"] = edge.epsilon_floor
        out["exploration_argmax"] = edge.gate == "argmax"
        out["exploration_boltzmann"] = edge.gate == "boltzmann"
        out["exploration_lcb"] = edge.gate == "lcb"
        out["exploration_argmax_lcb"] = edge.gate == "argmax_lcb"
        out["exploration_thompson"] = edge.gate == "thompson"
        out.setdefault("expert_policy", edge.expert_policy)

    # IBRL / LCBGatedBootstrap / CriticBlend / MCVarianceCorrection
    # implement their TD-target math on their Extension's ``on_target``
    # method; the SAC loop folds them via ``stack.on_target(...)``. The
    # opposite direction (legacy-flag → auto-appended extension) is
    # handled by ``_auto_append_target_mod_extensions``.
    #
    # We DO still echo the secondary parameters each extension owns into
    # the resolver dict — e.g. ``critic_warmup_frac`` is also consumed by
    # the online-BC loss schedule (``make_bc_loss_fn``), and the LCB
    # bootstrap shares its ``lcb_*`` hyperparameters with the
    # action-selection gate (``make_action_pipeline``). Without these
    # echoes the new surface would silently diverge from the legacy flag
    # path even though the target-mod math itself is byte-identical.
    ibrl = first_of_type(extensions, IBRL)
    if ibrl is not None:
        out.setdefault("expert_policy", ibrl.expert_policy)

    lcb_b = first_of_type(extensions, LCBGatedBootstrap)
    if lcb_b is not None:
        out.setdefault("lcb_beta_init", lcb_b.lcb_beta_init)
        out.setdefault("lcb_beta_decay_k", lcb_b.lcb_beta_decay_k)
        out.setdefault("lcb_temperature", lcb_b.lcb_temperature)
        out.setdefault("expert_policy", lcb_b.expert_policy)

    blend = first_of_type(extensions, CriticBlend)
    if blend is not None:
        out.setdefault("critic_warmup_frac", blend.critic_warmup_frac)
        out.setdefault("expert_policy", blend.expert_policy)

    box = first_of_type(extensions, ValueBox)
    if box is not None:
        out["use_box"] = True
        out.setdefault("expert_policy", box.expert_policy)

    mcp = first_of_type(extensions, MCPretrain)
    if mcp is not None:
        out["use_mc_critic_pretrain"] = True
        out["mc_pretrain_n_mc_steps"] = mcp.n_mc_steps
        out["mc_pretrain_n_mc_episodes"] = mcp.n_mc_episodes
        out["mc_pretrain_n_steps"] = mcp.n_steps
        out["use_online_critic_light_pretrain"] = mcp.use_online_light
        out["online_critic_pretrain_steps"] = mcp.online_light_steps
        out["online_critic_pretrain_lr_scale"] = mcp.online_light_lr_scale
        out.setdefault("expert_policy", mcp.expert_policy)

    bp = first_of_type(extensions, BellmanPretrain)
    if bp is not None:
        out["use_bellman_critic_pretrain"] = True
        out["mc_pretrain_n_steps"] = bp.n_steps
        out.setdefault("expert_policy", bp.expert_policy)

    pr = first_of_type(extensions, PhiRefresh)
    if pr is not None:
        out["use_phi_refresh"] = True
        out["phi_refresh_interval"] = pr.interval
        out["phi_refresh_steps"] = pr.steps
        out.setdefault("expert_policy", pr.expert_policy)

    return out


def _auto_append_target_mod_extensions(
    extensions: Sequence,
    *,
    ibrl_bootstrap: bool,
    lcb_gated_bootstrap: bool,
    use_critic_blend: bool,
    mc_variance_threshold: Optional[float],
    critic_warmup_frac: float,
    lcb_beta_init: float,
    lcb_beta_decay_k: float,
    lcb_temperature: float,
    expert_policy: Optional[Callable],
) -> tuple:
    """Append target-mod extensions for legacy flags that lack a matching one.

    The four target-mod features (``ibrl_bootstrap`` / ``lcb_gated_
    bootstrap`` / ``use_critic_blend`` / ``mc_variance_threshold``) now
    live exclusively on their :class:`Extension` ``on_target`` method.
    When a user still wires them through the legacy flag surface, and
    the matching extension is not already in ``extensions``, this helper
    constructs and appends it in the canonical order
    (IBRL → LCBGatedBootstrap → CriticBlend → MCVarianceCorrection) so
    the resulting ``stack.on_target(...)`` fold reproduces the
    pre-refactor ``make_target_modifier`` behaviour byte-identically.
    """
    from ajax.extensions.target_mods import (
        IBRL,
        CriticBlend,
        LCBGatedBootstrap,
        MCVarianceCorrection,
    )

    out = list(extensions)

    def _has(cls: type) -> bool:
        return any(isinstance(e, cls) for e in out)

    if ibrl_bootstrap and expert_policy is not None and not _has(IBRL):
        out.append(IBRL(expert_policy=expert_policy))
    if (
        lcb_gated_bootstrap
        and expert_policy is not None
        and not _has(LCBGatedBootstrap)
    ):
        out.append(
            LCBGatedBootstrap(
                expert_policy=expert_policy,
                lcb_beta_init=lcb_beta_init,
                lcb_beta_decay_k=lcb_beta_decay_k,
                lcb_temperature=lcb_temperature,
            )
        )
    if use_critic_blend and expert_policy is not None and not _has(CriticBlend):
        out.append(
            CriticBlend(
                expert_policy=expert_policy,
                critic_warmup_frac=critic_warmup_frac,
            )
        )
    if mc_variance_threshold is not None and not _has(MCVarianceCorrection):
        out.append(MCVarianceCorrection(threshold=mc_variance_threshold))

    return tuple(out)


# ---------------------------------------------------------------------------
# Training factory
# ---------------------------------------------------------------------------


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: SACConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
    eval_expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    early_termination_condition: Optional[Callable] = None,
    residual: bool = False,
    fixed_alpha: bool = False,
    num_critics: int = 2,
    extra_critic_head_names: Tuple[str, ...] = (),
    extra_critic_head_dims: Tuple[int, ...] = (),
    expert_buffer_n_steps: int = 20_000,
    num_critic_updates: int = 1,
    expert_mix_fraction: float = 0.1,
    box_threshold: float = 500.0,
    altitude_obs_idx: int = 1,
    target_obs_idx: int = 6,
    # MC critic pretraining (replaces Bellman pretraining)
    use_mc_critic_pretrain: bool = False,
    mc_pretrain_n_mc_steps: int = 10_000,
    mc_pretrain_n_mc_episodes: int = 100,
    mc_pretrain_n_steps: int = 5_000,
    # Online critic light pre-regression (requires MC critic pretrain)
    use_online_critic_light_pretrain: bool = True,
    online_critic_pretrain_steps: int = 500,
    online_critic_pretrain_lr_scale: float = 0.1,
    # Bellman critic pretraining (legacy fallback, mutually exclusive with MC)
    use_bellman_critic_pretrain: bool = False,
    # Expert-guided policy loss terms
    augment_obs_with_expert_action: bool = False,
    detach_obs_aug_action: bool = False,
    # Train-fraction conditioning: append timestep/total_timesteps to obs
    use_train_frac: bool = False,
    # Update start thresholds
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    # Blended Bellman target (replaces potential-based shaping)
    use_critic_blend: bool = False,
    critic_warmup_frac: float = 0.15,
    # Value-threshold box (v_min/v_max inferred from MC pretraining)
    use_box: bool = False,
    # Online decaying BC term (active for all MC pretrain runs unless disabled)
    use_online_bc: bool = True,
    bc_coef: float = 1.0,
    # EDGE (Expert Decayed Guided Exploration): stochastic expert substitution during collection
    use_expert_guided_exploration: bool = False,
    exploration_decay_frac: float = 0.30,
    exploration_tau: float = 1.0,
    exploration_boltzmann: bool = False,
    fixed_exploration_prob: float = 0.5,
    exploration_argmax: bool = False,
    # Quality-aware (LCB) gate
    exploration_lcb: bool = False,
    exploration_argmax_lcb: bool = False,
    exploration_thompson: bool = False,
    lcb_beta_init: float = 1.0,
    lcb_beta_decay_k: float = 2.0,
    lcb_temperature: float = 1.0,
    epsilon_floor: float = 0.0,
    lcb_asymmetric: bool = False,
    expert_fraction: float = 0.7,
    target_entropy_initial: Optional[float] = None,
    target_entropy_ramp_frac: float = 0.5,
    augment_obs_with_expert_state: bool = False,
    expert_state_aug_dim: int = 0,
    normalize_obs_running: bool = False,
    store_policy_action: bool = False,
    lcb_gated_bootstrap: bool = False,
    # Residual RL (Johannink et al.): execute clip(a_expert + scale * a_pi, -1, 1)
    use_residual_rl: bool = False,
    residual_scale: float = 1.0,
    # True JSRL (Uchendu et al. 2023): per-episode curriculum handoff.
    # Expert acts while step_in_episode < H_t; H_t decays from
    # jsrl_episode_length to 0 over jsrl_decay_frac * T_total.
    jsrl_curriculum: bool = False,
    jsrl_episode_length: int = 1000,
    jsrl_decay_frac: float = 0.5,
    # PID policy: execute expert action directly (no actor used for env interaction)
    use_pid_policy: bool = False,
    # Distance-modulated entropy target (None = disabled)
    target_entropy_far: Optional[float] = None,
    # Online MC correction for high-variance critic states (None = disabled)
    mc_variance_threshold: Optional[float] = None,
    # Periodic self-consistent φ* refresh via expert-flagged buffer transitions
    use_phi_refresh: bool = False,
    phi_refresh_interval: int = 500,
    phi_refresh_steps: int = 20,
    # IBRL bootstrap: use max(Q_policy, Q_expert) for TD target to be consistent
    # with argmax action-selection policy (exploration_argmax=True / EDGE).
    ibrl_bootstrap: bool = False,
    # Pre-collected MC data: (obs, action, mc_return) JAX arrays.
    # When provided, the in-run expert rollout + MC-return computation is skipped.
    mc_preloaded_data: Optional[Tuple] = None,
    # PID actor: actor network predicts PID gains instead of raw actions.
    pid_actor_config=None,
    # Gain-policy mode: actor output dim = len(expert.learnable_fields)
    action_dim_override: Optional[int] = None,
    # --- Composable hook overrides (None = build from flags above) ---
    action_pipeline: Optional[Callable] = None,
    obs_preprocessor: Optional[Callable] = None,
    policy_action_transform: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    runtime_maintenance: Optional[Callable] = None,
    extra_actor_loss_fn: Optional[Callable] = None,
    extra_critic_loss_fn: Optional[Callable] = None,
    her_relabel_fn: Optional[Callable] = None,
    init_transform: Optional[Callable] = None,
    auxiliary_update: Optional[Callable] = None,
    extra_eval_metrics: Optional[Callable] = None,
    # --- Extension framework (new surface) ---
    extensions: Sequence = (),
):
    """
    SAC training factory.

    expert_policy:      used for training (warmup seeding, critic
                        pre-training). Pass None for true vanilla SAC.
    eval_expert_policy: used ONLY for eval logging (expert bias metric).
                        Always passed regardless of whether training uses expert.
                        Defaults to expert_policy if not set explicitly.
    extensions:         the new composable-research-features surface
                        (Phase 2 of the agent-architecture rework). Any
                        :class:`~ajax.extensions.base.Extension` in the
                        sequence resolves to the matching legacy flag(s)
                        via :func:`_resolve_extension_stack`. The legacy
                        kwargs above remain accepted for backward
                        compatibility; explicit kwargs always win.
    """
    # Translate the Extension stack into the legacy flag surface.
    # Extensions OWN the parameters they expose: any value an extension
    # supplies (an :class:`OnlineBC`'s ``bc_coef``, a
    # :class:`JSRLCurriculum`'s ``episode_length``, …) overrides the
    # legacy default of the same flag. Empty stack ⇒ {} ⇒ no overrides
    # ⇒ plain-SAC behaviour byte-identical to the pre-refactor flag path
    # (the parity gate). Callers that want a hybrid configuration should
    # express it via the extension list rather than mixing the two
    # surfaces.
    if extensions:
        _resolved = _resolve_extension_stack(extensions)
        _locals = {**locals(), **_resolved}
        # rebind every flag the resolver owns so it propagates into the
        # downstream init/scan calls.
        expert_policy = _locals.get("expert_policy", expert_policy)
        eval_expert_policy = _locals.get("eval_expert_policy", eval_expert_policy)
        use_expert_guidance = _locals.get("use_expert_guidance", use_expert_guidance)
        expert_buffer_n_steps = _locals.get(
            "expert_buffer_n_steps", expert_buffer_n_steps
        )
        expert_mix_fraction = _locals.get("expert_mix_fraction", expert_mix_fraction)
        expert_fraction = _locals.get("expert_fraction", expert_fraction)
        augment_obs_with_expert_action = _locals.get(
            "augment_obs_with_expert_action", augment_obs_with_expert_action
        )
        detach_obs_aug_action = _locals.get(
            "detach_obs_aug_action", detach_obs_aug_action
        )
        use_online_bc = _locals.get("use_online_bc", use_online_bc)
        bc_coef = _locals.get("bc_coef", bc_coef)
        critic_warmup_frac = _locals.get("critic_warmup_frac", critic_warmup_frac)
        use_residual_rl = _locals.get("use_residual_rl", use_residual_rl)
        residual_scale = _locals.get("residual_scale", residual_scale)
        jsrl_curriculum = _locals.get("jsrl_curriculum", jsrl_curriculum)
        jsrl_episode_length = _locals.get("jsrl_episode_length", jsrl_episode_length)
        jsrl_decay_frac = _locals.get("jsrl_decay_frac", jsrl_decay_frac)
        use_expert_guided_exploration = _locals.get(
            "use_expert_guided_exploration", use_expert_guided_exploration
        )
        exploration_decay_frac = _locals.get(
            "exploration_decay_frac", exploration_decay_frac
        )
        exploration_tau = _locals.get("exploration_tau", exploration_tau)
        fixed_exploration_prob = _locals.get(
            "fixed_exploration_prob", fixed_exploration_prob
        )
        lcb_beta_init = _locals.get("lcb_beta_init", lcb_beta_init)
        lcb_beta_decay_k = _locals.get("lcb_beta_decay_k", lcb_beta_decay_k)
        lcb_temperature = _locals.get("lcb_temperature", lcb_temperature)
        lcb_asymmetric = _locals.get("lcb_asymmetric", lcb_asymmetric)
        epsilon_floor = _locals.get("epsilon_floor", epsilon_floor)
        exploration_argmax = _locals.get("exploration_argmax", exploration_argmax)
        exploration_boltzmann = _locals.get(
            "exploration_boltzmann", exploration_boltzmann
        )
        exploration_lcb = _locals.get("exploration_lcb", exploration_lcb)
        exploration_argmax_lcb = _locals.get(
            "exploration_argmax_lcb", exploration_argmax_lcb
        )
        exploration_thompson = _locals.get("exploration_thompson", exploration_thompson)
        ibrl_bootstrap = _locals.get("ibrl_bootstrap", ibrl_bootstrap)
        lcb_gated_bootstrap = _locals.get("lcb_gated_bootstrap", lcb_gated_bootstrap)
        use_critic_blend = _locals.get("use_critic_blend", use_critic_blend)
        mc_variance_threshold = _locals.get(
            "mc_variance_threshold", mc_variance_threshold
        )
        use_box = _locals.get("use_box", use_box)
        use_mc_critic_pretrain = _locals.get(
            "use_mc_critic_pretrain", use_mc_critic_pretrain
        )
        mc_pretrain_n_mc_steps = _locals.get(
            "mc_pretrain_n_mc_steps", mc_pretrain_n_mc_steps
        )
        mc_pretrain_n_mc_episodes = _locals.get(
            "mc_pretrain_n_mc_episodes", mc_pretrain_n_mc_episodes
        )
        mc_pretrain_n_steps = _locals.get("mc_pretrain_n_steps", mc_pretrain_n_steps)
        use_online_critic_light_pretrain = _locals.get(
            "use_online_critic_light_pretrain", use_online_critic_light_pretrain
        )
        online_critic_pretrain_steps = _locals.get(
            "online_critic_pretrain_steps", online_critic_pretrain_steps
        )
        online_critic_pretrain_lr_scale = _locals.get(
            "online_critic_pretrain_lr_scale", online_critic_pretrain_lr_scale
        )
        use_bellman_critic_pretrain = _locals.get(
            "use_bellman_critic_pretrain", use_bellman_critic_pretrain
        )
        use_phi_refresh = _locals.get("use_phi_refresh", use_phi_refresh)
        phi_refresh_interval = _locals.get("phi_refresh_interval", phi_refresh_interval)
        phi_refresh_steps = _locals.get("phi_refresh_steps", phi_refresh_steps)

    # Back-compat: when a legacy target-mod flag is set and the matching
    # Extension is not already in ``extensions``, auto-append it. This is
    # the shim that keeps `SAC(..., ibrl_bootstrap=True)` /
    # `lcb_gated_bootstrap=True` / `use_critic_blend=True` /
    # `mc_variance_threshold=...` producing the same numerics as before
    # the migration — the implementation now lives on the Extension's
    # ``on_target`` method and the SAC loop folds the stack via
    # ``stack.on_target(...)`` (the `make_target_modifier` builder was
    # deleted). No-op when an explicit extension is already supplied.
    extensions = _auto_append_target_mod_extensions(
        extensions,
        ibrl_bootstrap=ibrl_bootstrap,
        lcb_gated_bootstrap=lcb_gated_bootstrap,
        use_critic_blend=use_critic_blend,
        mc_variance_threshold=mc_variance_threshold,
        critic_warmup_frac=critic_warmup_frac,
        lcb_beta_init=lcb_beta_init,
        lcb_beta_decay_k=lcb_beta_decay_k,
        lcb_temperature=lcb_temperature,
        expert_policy=expert_policy,
    )

    # If no separate eval policy provided, fall back to the training policy
    # (which may be None for vanilla SAC — in that case no expert bias logged)
    _eval_expert_policy = (
        eval_expert_policy if eval_expert_policy is not None else expert_policy
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    if logging_config is not None:
        start_async_logging()

    # Cloning parameters are a pure function of the (closure-constant)
    # cloning_args + total_timesteps, so resolve them once here: they are
    # needed both by the fresh-init pretraining and by the scan partial.
    cloning_parameters, pre_train_n_steps = get_cloning_args(
        cloning_args, total_timesteps
    )
    num_updates = total_timesteps // env_args.n_envs

    # ------------------------------------------------------------------
    # Fresh-init path: build the agent state and run all one-shot
    # initialization (init_transform, MC / Bellman critic pretraining,
    # behavioural-cloning pretraining). This is *only* invoked on a fresh
    # run; on resume the shared helper reuses ``initial_state`` directly
    # so none of this expensive one-shot work is re-run.
    # ------------------------------------------------------------------
    def init_fn(key, index):
        """Build a fresh SAC agent state with all one-shot pretraining."""
        init_key, expert_key, transform_key = jax.random.split(key, 3)

        agent_state = init_SAC(
            key=init_key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=buffer,
            expert_policy=expert_policy,
            max_timesteps=total_timesteps if use_train_frac else None,
            num_critics=num_critics,
            expert_buffer_n_steps=(
                expert_buffer_n_steps if expert_policy is not None else 0
            ),
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            augment_obs_with_expert_state=augment_obs_with_expert_state,
            expert_state_aug_dim=expert_state_aug_dim,
            pid_actor_config=pid_actor_config,
            action_dim_override=action_dim_override,
            normalize_obs_running=normalize_obs_running,
            jsrl_curriculum=jsrl_curriculum,
            extra_critic_head_names=extra_critic_head_names,
            extra_critic_head_dims=extra_critic_head_dims,
        )

        if init_transform is not None:
            agent_state = init_transform(agent_state, transform_key)

        # Initialise the per-extension state tuple to match the stack
        # built in make_scan_fn (one entry per Extension in `extensions`).
        # The four target-mod extensions are stateless (``init_state`` →
        # ``()``) so this is one ``()`` entry per extension — no pytree
        # overhead — but it keeps the index used by
        # ``ExtensionStack.on_target`` in range. Splitting a sub-key off
        # ``init_key`` keeps the stateless path deterministic even when a
        # future stateful extension consumes randomness.
        if extensions:
            _ext_key, _ = jax.random.split(init_key)
            _stack = ExtensionStack(extensions)
            agent_state = agent_state.replace(
                ext_state=_stack.init_states(agent_state, _ext_key)
            )

        if expert_policy is not None and use_mc_critic_pretrain:
            expert_critic_state = get_initialized_critic(
                key=expert_key,
                env_config=env_args,
                critic_optimizer_config=critic_optimizer_args,
                network_config=network_args,
                num_critics=num_critics,
                max_timesteps=total_timesteps if use_train_frac else None,
                extra_obs_dim=(
                    get_action_dim(env_args.env, env_args.env_params)
                    if augment_obs_with_expert_action
                    else 0
                ),
            )
            _preloaded = mc_preloaded_data  # None or (obs, action, mc) JAX arrays
            (
                agent_state,
                frozen_expert_params,
                mc_obs_batched,
                mc_action_batched,
                mc_aux,
                expert_critic_state_trained,
            ) = pretrain_critic_mc(
                agent_state=agent_state,
                expert_critic_state=expert_critic_state,
                expert_policy=expert_policy,
                mode=mode,
                env_args=env_args,
                recurrent=network_args.lstm_hidden_size is not None,
                gamma=agent_config.gamma,
                reward_scale=agent_config.reward_scale,
                n_mc_steps=mc_pretrain_n_mc_steps,
                n_mc_episodes=mc_pretrain_n_mc_episodes,
                n_steps=mc_pretrain_n_steps,
                max_timesteps=total_timesteps if use_train_frac else None,
                augment_obs_with_expert_action=augment_obs_with_expert_action,
                preloaded_obs=_preloaded[0] if _preloaded is not None else None,
                preloaded_action=_preloaded[1] if _preloaded is not None else None,
                preloaded_mc=_preloaded[2] if _preloaded is not None else None,
            )
            agent_state = agent_state.replace(
                expert_critic_params=frozen_expert_params,
                expert_v_min=mc_aux.v_min,
                expert_v_max=mc_aux.v_max,
                # Keep φ* optimizer state alive for periodic refresh (None when disabled)
                expert_critic_state=expert_critic_state_trained
                if use_phi_refresh
                else None,
            )
            # ``use_box`` value-box bounds == the MC-pretrain v_min/v_max,
            # which are already persisted on ``agent_state`` above; the
            # scan-fn builder reads them back from there (see make_scan_fn).
            jax.debug.print(
                "[MC pretrain] loss: {i:.4f} -> {f:.4f}  |  "
                "Q(s,a*) mean={qm:.1f}  min={qn:.1f}  max={qx:.1f}",
                i=mc_aux.initial_loss,
                f=mc_aux.final_loss,
                qm=mc_aux.q_expert_mean,
                qn=mc_aux.q_expert_min,
                qx=mc_aux.q_expert_max,
            )

            if use_online_critic_light_pretrain:
                agent_state = pretrain_critic_online_light(
                    agent_state,
                    mc_obs_batched,
                    mc_action_batched,
                    n_steps=online_critic_pretrain_steps,
                    lr_scale=online_critic_pretrain_lr_scale,
                )
                jax.debug.print(
                    "[Online critic light pretrain] done ({n} steps, lr_scale={s})",
                    n=online_critic_pretrain_steps,
                    s=online_critic_pretrain_lr_scale,
                )

        if expert_policy is not None and use_bellman_critic_pretrain:
            agent_state = pretrain_critic_bellman(
                agent_state=agent_state,
                recurrent=network_args.lstm_hidden_size is not None,
                gamma=agent_config.gamma,
                reward_scale=agent_config.reward_scale,
                buffer=buffer,
                n_steps=mc_pretrain_n_steps,
                update_value_fn=update_value_functions,
                update_target_fn=update_target_networks,
            )
            jax.debug.print(
                "[Bellman pretrain] done ({n} steps)", n=mc_pretrain_n_steps
            )

        if pre_train_n_steps > 0:
            agent_state = get_pre_trained_agent(
                agent_state,
                expert_policy,
                expert_key,
                env_args,
                cloning_args,
                mode,
                agent_config,
                actor_optimizer_args,
                critic_optimizer_args,
                augment_obs_with_expert_action=augment_obs_with_expert_action,
                augment_obs_with_expert_state=augment_obs_with_expert_state,
            )

        return agent_state

    # ------------------------------------------------------------------
    # Per-iteration scan body. Built once at trace time *after*
    # init/resume is resolved so the value-box bounds can be read off the
    # resolved ``agent_state``. On the fresh-init path the MC-pretrain
    # block above stored those bounds on ``expert_v_min/expert_v_max``;
    # on resume they are left at 0.0, matching the pre-refactor behaviour
    # (the original ``train`` defaulted ``_box_v_min/_box_v_max`` to 0.0
    # and only overwrote them inside the fresh-init MC-pretrain branch).
    # ------------------------------------------------------------------
    _, action_shape = get_state_action_shapes(env_args.env)

    _valid_cloning_params = {
        k: v
        for k, v in cloning_parameters.items()
        if k
        in (
            "n_epochs",
            "transition_mix_fraction",
            "imitation_coef",
            "distance_to_stable",
            "imitation_coef_offset",
        )
    }

    def make_scan_fn(agent_state, resume_from_state, key, index):
        # Value-box bounds: on a fresh ``use_box`` run they equal the
        # MC-pretrain v_min/v_max persisted on the agent state; on resume
        # (or when no MC pretrain ran) they default to 0.0.
        if use_box and not resume_from_state:
            _box_v_min = agent_state.expert_v_min
            _box_v_max = agent_state.expert_v_max
        else:
            _box_v_min = jnp.array(0.0)
            _box_v_max = jnp.array(0.0)

        # Compose hooks: if the caller supplied an override, use it;
        # otherwise build the default from the legacy boolean flags.
        _action_pipeline = (
            action_pipeline
            if action_pipeline is not None
            else make_action_pipeline(
                expert_policy=expert_policy,
                recurrent=network_args.lstm_hidden_size is not None,
                env_args=env_args,
                use_box=use_box,
                box_v_min=_box_v_min,
                box_v_max=_box_v_max,
                use_expert_guided_exploration=use_expert_guided_exploration,
                exploration_decay_frac=exploration_decay_frac,
                exploration_tau=exploration_tau,
                exploration_boltzmann=exploration_boltzmann,
                exploration_argmax=exploration_argmax,
                fixed_exploration_prob=fixed_exploration_prob,
                exploration_lcb=exploration_lcb,
                exploration_argmax_lcb=exploration_argmax_lcb,
                lcb_asymmetric=lcb_asymmetric,
                exploration_thompson=exploration_thompson,
                expert_fraction=expert_fraction,
                lcb_beta_init=lcb_beta_init,
                lcb_beta_decay_k=lcb_beta_decay_k,
                lcb_temperature=lcb_temperature,
                epsilon_floor=epsilon_floor,
                use_residual_rl=use_residual_rl,
                residual_scale=residual_scale,
                jsrl_curriculum=jsrl_curriculum,
                jsrl_episode_length=jsrl_episode_length,
                jsrl_decay_frac=jsrl_decay_frac,
                use_pid_policy=use_pid_policy,
                augment_obs_with_expert_action=augment_obs_with_expert_action,
                store_policy_action=store_policy_action,
                total_timesteps=total_timesteps,
            )
        )

        # The four target-mod extensions (IBRL / LCBGatedBootstrap /
        # CriticBlend / MCVarianceCorrection) now fold through the
        # ExtensionStack's ``on_target`` phase. We pass the resolved
        # stack down to ``update_value_functions`` so it can call
        # ``stack.on_target(...)`` directly — the old ``target_modifier``
        # callable plumbing is gone.
        _extension_stack = ExtensionStack(extensions)

        _obs_preprocessor = (
            obs_preprocessor
            if obs_preprocessor is not None
            else make_policy_obs_preprocessor(
                augment_obs_with_expert_action,
                detach_obs_aug_action,
                action_shape[0],
            )
        )
        _policy_action_transform = (
            policy_action_transform
            if policy_action_transform is not None
            else make_policy_action_transform(
                use_residual_rl,
                expert_policy,
                residual_scale=residual_scale,
            )
        )
        _bc_loss_fn = make_bc_loss_fn(
            use_online_bc,
            bc_coef,
            critic_warmup_frac,
            expert_policy,
        )

        _eval_action_transform = (
            eval_action_transform
            if eval_action_transform is not None
            else make_eval_action_transform(
                use_residual_rl=use_residual_rl,
                use_pid_policy=use_pid_policy,
                residual_scale=residual_scale,
            )
        )

        _runtime_maintenance = (
            runtime_maintenance
            if runtime_maintenance is not None
            else make_runtime_maintenance(
                use_phi_refresh=use_phi_refresh,
                phi_refresh_interval=phi_refresh_interval,
                phi_refresh_steps=phi_refresh_steps,
                gamma=agent_config.gamma,
                reward_scale=agent_config.reward_scale,
                expert_policy=expert_policy,
                buffer=buffer,
            )
        )

        training_iteration_scan_fn = partial(
            training_iteration,
            buffer=buffer,
            recurrent=network_args.lstm_hidden_size is not None,
            action_dim=(
                action_dim_override
                if action_dim_override is not None
                else action_shape[0]
            ),
            agent_config=agent_config,
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
            sweep=(logging_config.sweep if logging_config is not None else False),
            horizon=(logging_config.horizon if logging_config is not None else None),
            expert_policy=expert_policy,
            eval_expert_policy=_eval_expert_policy,
            use_expert_guidance=use_expert_guidance,
            early_termination_condition=early_termination_condition,
            num_critic_updates=num_critic_updates,
            expert_mix_fraction=expert_mix_fraction,
            box_threshold=box_threshold,
            altitude_obs_idx=altitude_obs_idx,
            target_obs_idx=target_obs_idx,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            augment_obs_with_expert_state=augment_obs_with_expert_state,
            policy_update_start=policy_update_start,
            alpha_update_start=alpha_update_start,
            exploration_tau=exploration_tau,
            target_entropy_far=target_entropy_far,
            target_entropy_initial=target_entropy_initial,
            target_entropy_ramp_frac=target_entropy_ramp_frac,
            action_pipeline=_action_pipeline,
            extension_stack=_extension_stack,
            obs_preprocessor=_obs_preprocessor,
            policy_action_transform=_policy_action_transform,
            bc_loss_fn=_bc_loss_fn,
            eval_action_transform=_eval_action_transform,
            pid_gain_policy=use_pid_policy,
            next_expert_fn=make_next_expert_fn(expert_policy),
            runtime_maintenance=_runtime_maintenance,
            extra_actor_loss_fn=extra_actor_loss_fn,
            extra_critic_loss_fn=extra_critic_loss_fn,
            her_relabel_fn=her_relabel_fn,
            auxiliary_update=auxiliary_update,
            extra_eval_metrics=extra_eval_metrics,
            **_valid_cloning_params,
        )

        # Do not accumulate per-step metrics in the scan ys: with vmap over N
        # seeds and T steps, each scalar metric becomes an [N, T] tensor that is
        # materialized on-device at scan completion, OOMing on large (N, T).
        # Metrics are already streamed to the host via ``jax.debug.callback``
        # inside ``evaluate_and_log``, so dropping the ys here is lossless.
        def _scan_body_no_ys(carry, x):
            new_carry, _metrics = training_iteration_scan_fn(carry, x)
            return new_carry, None

        return _scan_body_no_ys

    return build_resumable_train(
        init_fn=init_fn,
        make_scan_fn=make_scan_fn,
        num_updates=num_updates,
    )
