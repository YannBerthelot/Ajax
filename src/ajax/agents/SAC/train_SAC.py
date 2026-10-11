from collections.abc import Sequence
from math import floor
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents.cloning import CloningConfig, pretrain_on_expert
from ajax.agents.loop import TrainLoop
from ajax.agents.recurrent import (
    RecurrentCarries,
    actor_dist,
    bootstrap_cuts,
    q_values,
    sample_and_burnin_sequences,
    unsupported_recurrent_options,
)
from ajax.agents.SAC import core
from ajax.agents.SAC.action_pipeline import make_action_pipeline, make_next_expert_fn
from ajax.agents.SAC.core import (
    TemperatureAuxiliaries,
    update_target_networks,
    update_temperature,
)
from ajax.agents.SAC.expert import (
    augment_obs_if_needed,
    collect_and_store_expert_transitions,
    compute_expert_diagnostics,
    pretrain_critic_bellman,
)
from ajax.agents.SAC.state import SACConfig, SACState
from ajax.agents.SAC.utils import SquashedNormal
from ajax.buffers.utils import get_batch_from_buffer, get_expert_fields_from_buffer
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import predict_value
from ajax.perf_utils import final_aux_scan
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.types import BufferType

# ---------------------------------------------------------------------------
# Auxiliary dataclasses for logging
# ---------------------------------------------------------------------------


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


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q_pred_min: jax.Array  # min over ensemble
    q_expert_mean: jax.Array  # critic's estimate of expert value
    q_gap: jax.Array  # q_expert - q_min: >0 = room to improve
    var_preds: jax.Array  # inter-critic variance
    expert_frac_in_buffer: jax.Array  # fraction of sampled batch flagged as expert
    phi_star_q_gap_ood: (
        jax.Array
    )  # |Q_φ*(s,π*) - Q_φ(s,π*)| mean: φ* OOD coverage error


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


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
    stored_state: bool = False,
    expert_policy: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    max_timesteps: Optional[int] = None,
    num_critics: int = 2,
    expert_buffer_n_steps: int = 20_000,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
    expert_state_aug_dim: int = 0,
    pid_actor_config=None,
    action_dim_override: Optional[int] = None,
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

    actor_state, critic_state, collector_state = core.init_soft_actor_critic(
        init_key,
        collector_key,
        env_args,
        actor_optimizer_args,
        critic_optimizer_args,
        network_args,
        buffer,
        num_critics=num_critics,
        window_size=window_size,
        stored_state=stored_state,
        pid_actor_config=pid_actor_config,
        network_extras={
            "max_timesteps": max_timesteps,
            "extra_obs_dim": extra_obs_dim,
            "action_dim_override": action_dim_override,
        },
        collector_extras={
            "max_timesteps": max_timesteps,
            "action_dim_override": action_dim_override,
            "expert_state_aug_dim": (
                expert_state_aug_dim if augment_obs_with_expert_state else 0
            ),
            "normalize_obs_running": normalize_obs_running,
            "include_expert_fields": expert_policy is not None,
        },
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

    return SACState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=core.create_alpha_train_state(**to_state_dict(alpha_args)),
        collector_state=collector_state,
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
    next_action_transform: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
    carries: Optional[RecurrentCarries] = None,
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
        carries=carries,
    )

    # 2. Q predictions for expert-path diagnostics. Only computed when a
    # consumer exists (the target-mod ExtensionStack feeds them in via the
    # ``q_preds`` batch entry for MCVarianceCorrection, or expert_q is set
    # so q_gap can be reported). The gradient-bearing pass inside
    # critic_loss_fn already exposes var_preds via core_aux.
    has_target_mods = (
        extension_stack is not None
        and "on_target" in extension_stack.implemented_phases()
    )
    needs_expert_q_preds = has_target_mods or expert_q is not None
    if needs_expert_q_preds:
        q_preds_for_var = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.params,
            x=jnp.concatenate((observations, jax.lax.stop_gradient(actions)), axis=-1),
        )

    # 3. The extensions' target modifiers (IBRL, LCBGatedBootstrap,
    # CriticBlend, MCVarianceCorrection, ...) fold in stack order.
    if has_target_mods:
        # `has_target_mods` already asserts `extension_stack is not
        # None` — assert it again for mypy.
        assert extension_stack is not None
        batch = {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
            "rewards": rewards,
            "reward_scale": reward_scale,
            "dones": dones,
            "rng_key": value_loss_key,
            "q_preds": q_preds_for_var,
            "gamma": gamma,
            "augment_obs_with_expert_action": augment_obs_with_expert_action,
            "recurrent": recurrent,
        }
        target_q = extension_stack.fold_on_target(
            agent_state,
            batch,
            target_q,
            agent_state.collector_state.timestep,
            value_loss_key,
            total_timesteps,
        )

    # 4. Core critic loss (MSE against the composed target) plus the
    #    extensions' critic-loss terms, inside the same value_and_grad so
    #    the one Adam step sees the combined gradient.
    critic_state = agent_state.critic_state

    def _critic_loss(params):
        loss, core_aux = core.critic_loss_fn(
            params, critic_state, observations, actions, target_q, carries=carries
        )
        if extension_stack is not None:
            loss_batch = {
                "observations": observations,
                "actions": actions,
                "targets": target_q,
                "critic_params": params,
                "critic_state": critic_state,
            }
            loss = loss + extension_stack.fold_critic_loss(
                agent_state,
                loss_batch,
                agent_state.collector_state.timestep,
                value_loss_key,
                total_timesteps,
            )
        return loss, core_aux

    (loss, core_aux), grads = jax.value_and_grad(_critic_loss, has_aux=True)(
        critic_state.params
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
        expert_frac_in_buffer=jnp.zeros(1),
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
    alpha: jax.Array,
    rng: jax.random.PRNGKey,
    raw_observations: Optional[jax.Array] = None,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    a_expert_precomputed: Optional[jax.Array] = None,
    train_frac: Optional[jax.Array] = None,
    expert_critic_params: Optional[Any] = None,
    expert_v_min: Optional[jax.Array] = None,
    expert_v_max: Optional[jax.Array] = None,
    policy_action_transform: Optional[Callable] = None,
    carries: Optional[RecurrentCarries] = None,
    extension_stack: Optional[ExtensionStack] = None,
    agent_state: Any = None,
    total_timesteps: int = 1,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """SAC actor loss with composable expert modifiers.

    Structure mirrors the critic side: core SAC loss + layered expert additions.
    1. Core: forward pass → sample → Q eval → α·log π - Q
    2. Modifier: policy_action_transform (residual RL before Q eval)
    3. Modifier: extension_stack.actor_loss (e.g. OnlineBC term)
    4. Diagnostics: expert Q gap, L2 distance
    """
    _raw_obs = (
        raw_observations if raw_observations is not None else observations[..., :-1]
    )

    # 1. Core forward pass + sample (``carries`` for sequence replay).
    pi = actor_dist(actor_state, actor_params, observations, carries)
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = pi.sample_and_log_prob(seed=sample_key)
    log_probs = log_probs.sum(-1, keepdims=True)

    policy_std = (
        pi.unsquashed_stddev().mean()
        if isinstance(pi, SquashedNormal)
        else pi.stddev().mean()
    )

    # 2. Action transform modifier (residual RL)
    q_input_actions = (
        policy_action_transform(actions, _raw_obs, a_expert_precomputed)
        if policy_action_transform is not None
        else actions
    )

    # Core Q evaluation and SAC loss. In recurrent mode the critic carry
    # was burned in on buffer actions and evaluates fresh policy actions
    # (standard burned-state approximation); gradients flow to the actor
    # through the actions.
    q_preds = q_values(
        critic_states, critic_states.params, observations, q_input_actions, carries
    )
    q_min = jnp.min(q_preds, axis=0)
    loss_actor = alpha * log_probs - q_min

    # 3. Expert diagnostics and the actor-loss extensions' terms, which
    # get the expert action (OnlineBC's BC target) once phi* exists.
    needs_expert = expert_policy is not None and use_expert_guidance
    needs_bc = (
        extension_stack is not None
        and "actor_loss" in extension_stack.implemented_phases()
        and expert_policy is not None
        and expert_critic_params is not None
    )

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

    # Additive actor-loss terms. Each extension reads what it needs out of
    # the ``batch`` dict (OnlineBC: pi_loc, a_expert, train_frac,
    # critic_state, expert_critic_params, expert_v_min/v_max) and returns
    # 0.0 when an operand is missing (no phi* before MC pretraining).
    if extension_stack:
        ext_batch = {
            "pi_loc": pi.distribution.loc,
            "a_expert": a_expert,
            "observations": observations,
            "train_frac": train_frac,
            "critic_state": critic_states,
            "expert_critic_params": expert_critic_params,
            "expert_v_min": expert_v_min,
            "expert_v_max": expert_v_max,
        }
        bc_term = jnp.asarray(
            extension_stack.fold_actor_loss(
                agent_state,
                ext_batch,
                agent_state.collector_state.timestep,
                rng,
                total_timesteps,
            )
        )
    else:
        bc_term = jnp.zeros(())

    total_loss = loss_actor.mean() + bc_term

    return total_loss, PolicyAuxiliaries(
        policy_loss=total_loss,
        log_pi=log_probs.mean(),
        policy_std=policy_std,
        q_min=q_min.mean(),
        q_expert=q_expert_logged,
        l2_expert=l2_expert_logged,
        above_expert_frac=above_expert_frac,
        raw_loss=loss_actor.mean(),
        bc_term=bc_term,
    )


def update_policy(
    agent_state: SACState,
    observations: jax.Array,
    raw_observations: jax.Array,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    a_expert_precomputed: Optional[jax.Array] = None,
    train_frac: Optional[jax.Array] = None,
    policy_action_transform: Optional[Callable] = None,
    extension_stack: Optional[ExtensionStack] = None,
    total_timesteps: int = 1,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[SACState, PolicyAuxiliaries, jax.Array]:
    """Returns (new_state, aux, log_probs) — log_probs reused by update_temperature
    to avoid a redundant actor forward pass."""
    rng, policy_key = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])

    (loss, aux), grads = jax.value_and_grad(
        policy_loss_function, has_aux=True, argnums=0
    )(
        agent_state.actor_state.params,
        agent_state.actor_state,
        agent_state.critic_state,
        observations,
        alpha,
        policy_key,
        raw_observations=raw_observations,
        expert_policy=expert_policy,
        use_expert_guidance=use_expert_guidance,
        a_expert_precomputed=a_expert_precomputed,
        train_frac=train_frac,
        expert_critic_params=agent_state.expert_critic_params,
        expert_v_min=agent_state.expert_v_min,
        expert_v_max=agent_state.expert_v_max,
        policy_action_transform=policy_action_transform,
        carries=carries,
        extension_stack=extension_stack,
        agent_state=agent_state,
        total_timesteps=total_timesteps,
    )

    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)

    # Recompute log_probs from updated actor for temperature update reuse
    temp_rng, temp_sample_key = jax.random.split(rng)
    pi = actor_dist(
        updated_actor_state, updated_actor_state.params, observations, carries
    )
    _, log_probs = pi.sample_and_log_prob(seed=temp_sample_key)
    return (
        agent_state.replace(rng=temp_rng, actor_state=updated_actor_state),
        aux,
        jax.lax.stop_gradient(log_probs),
    )


# ---------------------------------------------------------------------------
# Agent update (one gradient step)
# ---------------------------------------------------------------------------


def update_agent(
    agent_state: SACState,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    target_entropy: float,
    tau: float,
    num_critic_updates: int = 1,
    reward_scale: float = 1.0,
    expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    fixed_alpha: bool = False,
    expert_mix_fraction: float = 0.1,
    augment_obs_with_expert_action: bool = False,
    total_timesteps: int = 1,
    target_entropy_initial: Optional[float] = None,
    target_entropy_ramp_frac: float = 0.5,
    extension_stack: Optional[ExtensionStack] = None,
    policy_action_transform: Optional[Callable] = None,
    burn_in: int = 8,
    stored_state: bool = False,
) -> Tuple[SACState, AuxiliaryLogs]:
    sample_key, expert_sample_key, rng = jax.random.split(agent_state.rng, 3)
    agent_state = agent_state.replace(rng=rng)

    # --- Sample from buffer ---
    carries = None
    if recurrent:
        # Sequence replay with burned-in carries (R2D2-style); expert
        # features are rejected upstream in make_train, so the expert
        # blocks below are all trace-time no-ops.
        transition, carries = sample_and_burnin_sequences(
            agent_state, buffer, sample_key, burn_in, stored_state=stored_state
        )
        expert_frac_in_buffer = jnp.zeros(())
    else:
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
            a_expert_buf, next_a_expert_buf = get_expert_fields_from_buffer(
                buffer, agent_state.collector_state.buffer_state, sample_key
            )
        else:
            a_expert_buf, next_a_expert_buf = None, None
        expert_frac_in_buffer = is_expert.mean()
        transition = Transition(
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

    dones = bootstrap_cuts(transition, carries)
    # The env observations without train_frac, which is what expert_policy
    # expects.
    _raw = (
        transition.raw_obs
        if transition.raw_obs is not None
        else transition.obs[..., :-1]
    )

    # --- Obs augmentation: append a_expert to obs and next_obs ---
    # Must happen before any network call (critic, actor, policy loss).
    # For next_obs we strip the last dim (train_frac).
    if augment_obs_with_expert_action and expert_policy is not None:
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
    # The actor-loss extensions (OnlineBC) take the expert action from
    # here, phi* or not.
    has_actor_loss = (
        extension_stack is not None
        and "actor_loss" in extension_stack.implemented_phases()
    )
    needs_expert = expert_policy is not None and (
        use_expert_guidance or policy_action_transform is not None or has_actor_loss
    )
    if needs_expert:
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
    _next_a_expert_for_target = (
        transition.next_a_expert
        if transition.next_a_expert is not None and policy_action_transform is not None
        else None
    )

    # See ajax.perf_utils.final_aux_scan: carry-only scan that exposes
    # last-step aux without materialising the full ys axis.
    def critic_update_step(state, _):
        return update_value_functions(
            observations=transition.obs,
            actions=transition.action,
            next_observations=transition.next_obs,
            rewards=transition.reward,
            dones=dones,
            agent_state=state,
            recurrent=recurrent,
            gamma=gamma,
            reward_scale=reward_scale,
            expert_q=expert_q,
            extension_stack=extension_stack,
            total_timesteps=total_timesteps,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            next_action_transform=policy_action_transform,
            next_a_expert=_next_a_expert_for_target,
            carries=carries,
        )

    agent_state, aux_value = final_aux_scan(
        critic_update_step,
        agent_state,
        length=num_critic_updates,
    )

    # --- Policy update — returns log_probs for temperature reuse ---
    train_frac = agent_state.collector_state.timestep / total_timesteps
    new_agent_state, aux_policy, policy_log_probs = update_policy(
        observations=transition.obs,
        agent_state=agent_state,
        raw_observations=transition.raw_obs,
        expert_policy=expert_policy,
        use_expert_guidance=use_expert_guidance,
        a_expert_precomputed=a_expert_precomputed,
        train_frac=train_frac,
        policy_action_transform=policy_action_transform,
        extension_stack=extension_stack,
        total_timesteps=total_timesteps,
        carries=carries,
    )
    agent_state = jax.lax.cond(
        agent_state.collector_state.timestep >= policy_update_start,
        lambda: new_agent_state,
        lambda: agent_state,
    )

    if target_entropy_initial is not None:
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
    # ``fixed_alpha`` keeps alpha pinned at ``alpha_init`` for the whole run
    # (the temperature loss is still computed for its logged auxiliaries,
    # only the gradient step is dropped).
    new_agent_state_temp, aux_temperature = update_temperature(
        agent_state, policy_log_probs, effective_target_entropy
    )
    agent_state = jax.lax.cond(
        jnp.logical_and(
            not fixed_alpha,
            agent_state.collector_state.timestep >= alpha_update_start,
        ),
        lambda: new_agent_state_temp,
        lambda: agent_state,
    )

    agent_state = update_target_networks(agent_state, tau=tau)

    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=aux_value.replace(
            expert_frac_in_buffer=expert_frac_in_buffer,
            phi_star_q_gap_ood=phi_star_q_gap_ood,
        ),
    )
    return agent_state, aux


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
    start_timestep: int = 0,
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
    eval_expert_policy: Optional[Callable] = None,
    use_expert_guidance: bool = True,
    fixed_alpha: bool = False,
    num_critics: int = 2,
    expert_buffer_n_steps: int = 20_000,
    num_critic_updates: int = 1,
    expert_mix_fraction: float = 0.1,
    # MC sizing kwarg threaded into the inline Bellman-pretrain block
    # below (``use_bellman_critic_pretrain``). MCPretrain extensions own
    # the equivalent for the MC path inside ``MCPretrain(n_steps=...)``.
    mc_pretrain_n_steps: int = 5_000,
    # Bellman critic pretraining (legacy fallback, mutually exclusive with MC)
    use_bellman_critic_pretrain: bool = False,
    # Expert obs augmentation: changes init_SAC / collect_experience
    # network input dim.
    augment_obs_with_expert_action: bool = False,
    # Train-fraction conditioning: append timestep/total_timesteps to obs
    use_train_frac: bool = False,
    # Update start thresholds
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    expert_fraction: float = 0.7,
    target_entropy_initial: Optional[float] = None,
    target_entropy_ramp_frac: float = 0.5,
    augment_obs_with_expert_state: bool = False,
    expert_state_aug_dim: int = 0,
    normalize_obs_running: bool = False,
    store_policy_action: bool = False,
    # Residual RL (Johannink et al.): execute clip(a_expert + scale * a_pi, -1, 1).
    # Threads into ``make_action_pipeline``; the ResidualPolicy extension
    # owns the actor-loss / TD-target transform via :meth:`transform_action`.
    use_residual_rl: bool = False,
    residual_scale: float = 1.0,
    # True JSRL (Uchendu et al. 2023): per-episode curriculum handoff.
    # The flag gates the ``step_in_episode`` counter init in
    # :func:`init_SAC`; the curriculum math lives on
    # :class:`JSRLCurriculum`.action.
    jsrl_curriculum: bool = False,
    # PID policy: execute expert action directly (no actor used for env interaction)
    use_pid_policy: bool = False,
    # PID actor: actor network predicts PID gains instead of raw actions.
    pid_actor_config=None,
    # Gain-policy mode: actor output dim = len(expert.learnable_fields)
    action_dim_override: Optional[int] = None,
    # --- Extension framework (the research-features surface) ---
    extensions: Sequence = (),
):
    """SAC's train function on :meth:`TrainLoop.off_policy`.

    expert_policy:      used for training (warmup seeding, expert buffer
                        prefill, residual / JSRL collection-time
                        substitution). Pass None for true vanilla SAC.
    eval_expert_policy: used ONLY for eval logging (expert bias metric).
                        Defaults to ``expert_policy`` if unset.
    extensions:         composable research features (see
                        :mod:`ajax.extensions`), bound to SAC's context.

    The flags above thread into init, collection or the action pipeline at
    a level the extension framework does not reach yet; they mirror the
    matching extension's "static" flag where there is one.
    """
    recurrent = network_args.memory is not None
    # Bind the SAC-factory-only context onto every extension (see
    # :meth:`Extension.bind_to_agent`: each picks the kwargs it needs).
    _resolved_action_dim = (
        action_dim_override
        if action_dim_override is not None
        else get_action_dim(env_args.env, env_args.env_params)
    )
    stack = ExtensionStack(extensions).bind_to_agent(
        env_args=env_args,
        network_args=network_args,
        critic_optimizer_args=critic_optimizer_args,
        num_critics=num_critics,
        buffer=buffer,
        gamma=agent_config.gamma,
        reward_scale=agent_config.reward_scale,
        use_train_frac=use_train_frac,
        augment_obs_with_expert_action=augment_obs_with_expert_action,
        action_dim=_resolved_action_dim,
        extensions=tuple(extensions),
    )
    if recurrent:
        # Expert-guidance features (and their Extension replacements) are
        # orthogonal to memory and untested with sequence replay; fail
        # loudly instead of silently misbehaving.
        unsupported_recurrent_options(
            "SAC",
            expert_policy=expert_policy,
            pid_actor_config=pid_actor_config,
            extensions=(stack.extensions or None),
            # SAC always builds a default CloningConfig; only actual
            # pre-training (pre_train_n_steps > 0) conflicts with memory.
            cloning_pretrain=(
                cloning_args
                if cloning_args is not None and cloning_args.pre_train_n_steps > 0
                else None
            ),
        )
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        stack.extensions,
        start_timestep=start_timestep,
    )

    def init(key: jax.Array, pretrain_key: jax.Array) -> SACState:
        agent_state = init_SAC(
            key=key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            stored_state=agent_config.stored_state,
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
        )
        if expert_policy is not None and use_bellman_critic_pretrain:
            agent_state = pretrain_critic_bellman(
                agent_state=agent_state,
                recurrent=recurrent,
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
        return pretrain_on_expert(
            agent_state,
            pretrain_key,
            cloning_args,
            expert_policy,
            env_args,
            actor_optimizer_args,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            augment_obs_with_expert_state=augment_obs_with_expert_state,
        )

    # An extension with ``build_policy_transform`` (ResidualPolicy)
    # supplies the actor-loss / TD-target action transform
    # ``(actions, raw_obs, a_expert) -> actions`` and the eval-time one;
    # ``None`` ⇒ pure SAC actor loss. Gain mode (``use_pid_policy``)
    # handles its own eval transform in ``step_environment``.
    residual = next(
        (e for e in loop.stack.extensions if hasattr(e, "build_policy_transform")),
        None,
    )
    policy_action_transform = (
        residual.build_policy_transform(expert_policy) if residual else None
    )
    eval_action_transform = (
        None
        if (use_pid_policy or residual is None)
        else residual.build_eval_transform()
    )

    def update(agent_state: SACState, _transition: Transition) -> Any:
        # The step just collected is in the buffer: SAC samples it from there.
        return update_agent(
            agent_state,
            buffer=buffer,
            recurrent=recurrent,
            gamma=agent_config.gamma,
            target_entropy=agent_config.target_entropy,
            tau=agent_config.tau,
            reward_scale=agent_config.reward_scale,
            expert_policy=expert_policy,
            use_expert_guidance=use_expert_guidance,
            policy_update_start=policy_update_start,
            alpha_update_start=alpha_update_start,
            fixed_alpha=fixed_alpha,
            num_critic_updates=num_critic_updates,
            expert_mix_fraction=expert_mix_fraction,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            total_timesteps=total_timesteps,
            target_entropy_initial=target_entropy_initial,
            target_entropy_ramp_frac=target_entropy_ramp_frac,
            extension_stack=loop.stack,
            policy_action_transform=policy_action_transform,
            burn_in=agent_config.burn_in,
            stored_state=agent_config.stored_state,
        )

    # The pipeline carries the SAC-side bookkeeping (warmup mix,
    # ``is_expert_flag``, ``buffer_action``, expert-state threading) and
    # the gain-policy short-circuit; the EDGE / ValueBox / JSRL gates run
    # through the extension stack.
    action_pipeline = make_action_pipeline(
        expert_policy=expert_policy,
        recurrent=recurrent,
        env_args=env_args,
        extension_stack=loop.stack,
        expert_fraction=expert_fraction,
        use_residual_rl=use_residual_rl,
        residual_scale=residual_scale,
        use_pid_policy=use_pid_policy,
        augment_obs_with_expert_action=augment_obs_with_expert_action,
        store_policy_action=store_policy_action,
        total_timesteps=total_timesteps,
    )
    return loop.off_policy(
        init,
        update,
        AuxiliaryLogs,
        agent_config.learning_starts,
        recurrent=recurrent,
        collect_kwargs={
            "buffer": buffer,
            "action_pipeline": action_pipeline,
            "next_expert_fn": make_next_expert_fn(expert_policy),
            "store_hidden": recurrent and agent_config.stored_state,
        },
        eval_kwargs={
            # Evaluate only in the last 20% of training (HPO phases).
            "sweep": logging_config is not None and logging_config.sweep,
            "expert_policy": (
                eval_expert_policy if eval_expert_policy is not None else expert_policy
            ),
            "eval_action_transform": eval_action_transform,
            "pid_gain_policy": use_pid_policy,
            "augment_obs_with_expert_action": augment_obs_with_expert_action,
            "augment_obs_with_expert_state": augment_obs_with_expert_state,
        },
    )
