from collections.abc import Sequence
from math import floor
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial

from ajax.agents.cloning import CloningConfig, get_pre_trained_agent
from ajax.agents.recurrent import (
    RecurrentCarries,
    sample_and_burnin_sequences,
    stored_actor_carry_dim,
    unsupported_recurrent_options,
)
from ajax.agents.SAC import core
from ajax.agents.SAC.state import SACConfig, SACState
from ajax.agents.SAC.utils import SquashedNormal
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    collect_experience,
    get_pi,
    get_pi_sequence,
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
    make_next_expert_fn,
)
from ajax.extensions.base import ExtensionContext, ExtensionStack
from ajax.extensions.pretrain import PhiRefresh as _PhiRefresh
from ajax.log import compose_eval_metrics, evaluate_and_log
from ajax.logging.wandb_logging import (
    LoggingConfig,
    start_async_logging,
    vmap_log,
)
from ajax.modules.expert import augment_obs_if_needed, compute_expert_diagnostics
from ajax.modules.pretrain import (
    collect_and_store_expert_transitions,
    pretrain_critic_bellman,
)
from ajax.networks.networks import (
    get_initialized_actor_critic,
    predict_value,
    predict_value_sequence,
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
from ajax.utils import fill_with_nan

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
        actor_carry_dim=stored_actor_carry_dim(network_args.memory, stored_state),
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
    dones: Optional[jax.Array],
    recurrent: bool,
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
    ext_state: tuple = (),
    total_timesteps: int = 1,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """SAC actor loss with composable expert modifiers.

    Structure mirrors the critic side: core SAC loss + layered expert additions.
    1. Pre-process: extension_stack.on_obs (detach expert-action dims)
    2. Core: forward pass → sample → Q eval → α·log π - Q
    3. Modifier: policy_action_transform (residual RL before Q eval)
    4. Modifier: extension_stack.actor_loss (e.g. OnlineBC term)
    5. Diagnostics: expert Q gap, L2 distance
    """
    _raw_obs = (
        raw_observations if raw_observations is not None else observations[..., :-1]
    )

    # 1. Pre-process: the ExtensionStack's ``on_obs`` fold
    # (ExpertObsAugmentation detaches the expert-action dims). Empty
    # stack ⇒ identity.
    if extension_stack is not None and extension_stack.extensions:
        _on_obs_ctx = ExtensionContext(
            step=jnp.asarray(0),
            rng=rng,
            total_steps=total_timesteps,
        )
        obs_for_actor = extension_stack.on_obs(observations, ext_state, _on_obs_ctx)
    else:
        obs_for_actor = observations

    # 2. Core forward pass + sample
    if recurrent:
        assert carries is not None  # narrowed: set by the recurrent path
        pi, _ = get_pi_sequence(
            actor_state=actor_state,
            actor_params=actor_params,
            obs=obs_for_actor,
            resets=carries.resets,
            initial_hidden=carries.actor_hidden,
        )
    else:
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

    # Core Q evaluation and SAC loss. In recurrent mode the critic carry
    # was burned in on buffer actions and evaluates fresh policy actions
    # (standard burned-state approximation); gradients flow to the actor
    # through the actions.
    if recurrent:
        assert carries is not None  # narrowed: set by the recurrent path
        q_preds, _ = predict_value_sequence(
            critic_state=critic_states,
            critic_params=critic_states.params,
            x=jnp.concatenate([observations, q_input_actions], axis=-1),
            resets=carries.resets,
            initial_hidden=carries.critic_hidden,
        )
    else:
        q_preds = predict_value(
            critic_state=critic_states,
            critic_params=critic_states.params,
            x=jnp.concatenate([observations, q_input_actions], axis=-1),
        )
    q_min = jnp.min(q_preds, axis=0)
    loss_actor = alpha * log_probs - q_min

    # 4. Expert diagnostics and BC loss
    # OnlineBC is the only :class:`Extension` that currently contributes
    # an ``actor_loss`` term; trigger the precomputed-a_expert path
    # whenever that extension is present so the BC math finds its
    # operand (the legacy ``needs_bc`` gate keyed on a non-None
    # ``bc_loss_fn`` callable, which is gone now).
    needs_expert = expert_policy is not None and use_expert_guidance
    has_online_bc = extension_stack is not None and any(
        ext.name == "online_bc" for ext in extension_stack.extensions
    )
    needs_bc = has_online_bc and expert_critic_params is not None

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

    # Additive actor-loss terms — folded through ``stack.actor_loss``.
    # Each extension reads what it needs out of the ``batch`` dict
    # (OnlineBC: pi_loc, a_expert, train_frac, critic_state,
    # expert_critic_params, expert_v_min/v_max). When the relevant
    # operands are missing (e.g. ``expert_critic_params is None`` ⇒ MC
    # pre-training hasn't run) the extension's own gate returns 0.0,
    # so this path is a silent no-op in that case — matching the
    # pre-refactor ``bc_loss_fn`` builder, which simply returned
    # ``None``. Empty stack ⇒ 0.0 too.
    if extension_stack is not None and extension_stack.extensions:
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
        ext_ctx = ExtensionContext(
            step=jnp.asarray(0),
            rng=rng,
            total_steps=total_timesteps,
        )
        # The actor-loss extensions (OnlineBC) read every operand off
        # ``batch``; agent_state and ext_state are passed for API
        # symmetry. The ext_state tuple must match the stack's
        # ``init_states`` shape (one entry per extension) so the
        # ExtensionStack fold can index it.
        bc_term = jnp.asarray(
            extension_stack.actor_loss(None, ext_state, ext_batch, ext_ctx)
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
    done: Optional[jax.Array],
    recurrent: bool,
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
        done,
        recurrent,
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
        ext_state=agent_state.ext_state,
        total_timesteps=total_timesteps,
    )

    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)

    # Recompute log_probs from updated actor for temperature update reuse
    temp_rng, temp_sample_key = jax.random.split(rng)
    if recurrent:
        assert carries is not None  # narrowed: set by the recurrent path
        pi, _ = get_pi_sequence(
            actor_state=updated_actor_state,
            actor_params=updated_actor_state.params,
            obs=observations,
            resets=carries.resets,
            initial_hidden=carries.actor_hidden,
        )
    else:
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
            from ajax.buffers.utils import get_expert_fields_from_buffer

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
    # ``has_online_bc`` mirrors the legacy ``bc_loss_fn is not None``
    # gate now that the BC term lives on
    # :meth:`OnlineBC.actor_loss`. Triggering the precomputed-a_expert
    # path whenever the extension is present preserves the original
    # numerical path even when MC pretrain hasn't (yet) populated
    # ``expert_critic_params``.
    has_online_bc = extension_stack is not None and any(
        ext.name == "online_bc" for ext in extension_stack.extensions
    )
    needs_expert = expert_policy is not None and (
        use_expert_guidance or policy_action_transform is not None or has_online_bc
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
            next_action_transform=_next_action_transform,
            next_a_expert=_next_a_expert_for_target,
            carries=carries,
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
        agent_state,
        log_probs=policy_log_probs,
        effective_target_entropy=effective_target_entropy,
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
    total_timesteps: int,
    log_frequency: int = 1000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    expert_policy: Optional[Callable] = None,  # used for training
    eval_expert_policy: Optional[Callable] = None,  # used for eval logging only
    use_expert_guidance: bool = True,
    num_critic_updates: int = 1,
    expert_mix_fraction: float = 0.1,
    augment_obs_with_expert_action: bool = False,
    augment_obs_with_expert_state: bool = False,
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    fixed_alpha: bool = False,
    target_entropy_initial: Optional[float] = None,
    target_entropy_ramp_frac: float = 0.5,
    action_pipeline: Optional[Callable] = None,
    extension_stack: Optional[ExtensionStack] = None,
    policy_action_transform: Optional[Callable] = None,
    eval_action_transform: Optional[Callable] = None,
    extra_eval_metrics: Optional[Callable] = None,
    pid_gain_policy: bool = False,
    next_expert_fn: Optional[Callable] = None,
    # Eval-suppression mode: when True, evaluate_and_log only fires evals
    # in the last 20% of training. Use for HPO phases where the only
    # number that matters is the final-window IQM.
    sweep: bool = False,
) -> tuple[SACState, None]:
    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_config.learning_starts)

    collect_scan_fn = partial(
        collect_experience,
        store_hidden=recurrent and agent_config.stored_state,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        buffer=buffer,
        uniform=uniform,
        action_pipeline=action_pipeline,
        next_expert_fn=next_expert_fn,
    )

    agent_state, _transition = collect_scan_fn(agent_state, None)
    timestep = agent_state.collector_state.timestep

    def do_update(agent_state):
        # ExtensionStack.post_update — PhiRefresh owns the periodic
        # interval gate + self-consistent refresh of
        # ``agent_state.expert_critic_params``. Other extensions'
        # post_update defaults to identity; empty stack ⇒ no-op.
        if extension_stack is not None:
            agent_state = extension_stack.fold_post_update(
                agent_state,
                agent_state.collector_state.timestep,
                agent_state.rng,
                total_timesteps,
            )

        agent_state, aux = update_agent(
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
            extension_stack=extension_stack,
            policy_action_transform=policy_action_transform,
            burn_in=agent_config.burn_in,
            stored_state=agent_config.stored_state,
        )
        # One (1,)-shaped leaf per metric: the metric-flattening contract.
        return agent_state, jax.tree.map(lambda x: x.reshape((1,)), aux)

    def skip_update(agent_state):
        return agent_state, fill_with_nan(AuxiliaryLogs)

    agent_state, aux = jax.lax.cond(
        timestep >= agent_config.learning_starts,
        do_update,
        skip_update,
        operand=agent_state,
    )

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
        log,
        log_fn,
        log_frequency,
        total_timesteps,
        sweep=sweep,
        expert_policy=eval_expert_policy,
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
# Extension-stack context injection
# ---------------------------------------------------------------------------
#
# Phase 5 + backbone lift: the back-compat translation layer that turned
# legacy flag kwargs into auto-appended Extensions is gone, and the
# Phase-5-era SAC-side ``_inject_*`` / ``_build_residual_*`` helpers
# (~170 lines) have been deleted. Context that an extension needs at
# runtime is now populated by the extension itself via
# :meth:`Extension.bind_to_agent`, which the factory calls uniformly on
# every extension in the stack — see the ``bind_to_agent`` block in
# ``make_train`` below. ``extensions=`` is the sole surface for
# composing research features on SAC.


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
    fixed_alpha: bool = False,
    num_critics: int = 2,
    extra_critic_head_names: Tuple[str, ...] = (),
    extra_critic_head_dims: Tuple[int, ...] = (),
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
    # network input dim. The runtime stop-gradient on the augmented dims
    # lives on :meth:`ExpertObsAugmentation.on_obs`.
    augment_obs_with_expert_action: bool = False,
    # Train-fraction conditioning: append timestep/total_timesteps to obs
    use_train_frac: bool = False,
    # Update start thresholds
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    # Value-threshold box (v_min/v_max inferred from MC pretraining).
    # Threads into ``make_scan_fn`` to resolve _box_v_min/_box_v_max from
    # the agent state. The ValueBox.action math lives on the extension.
    use_box: bool = False,
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
    # Pre-collected MC data: (obs, action, mc_return) JAX arrays.
    # When provided, the in-run expert rollout + MC-return computation is skipped.
    mc_preloaded_data: Optional[Tuple] = None,
    # PID actor: actor network predicts PID gains instead of raw actions.
    pid_actor_config=None,
    # Gain-policy mode: actor output dim = len(expert.learnable_fields)
    action_dim_override: Optional[int] = None,
    # --- Extension framework (the research-features surface) ---
    extensions: Sequence = (),
):
    """SAC training factory.

    expert_policy:      used for training (warmup seeding, expert buffer
                        prefill, residual / JSRL collection-time
                        substitution). Pass None for true vanilla SAC.
    eval_expert_policy: used ONLY for eval logging (expert bias metric).
                        Defaults to ``expert_policy`` if unset.
    extensions:         composable research features. Each
                        :class:`~ajax.extensions.base.Extension`
                        instance owns its own math via the phase
                        methods (``on_target`` / ``actor_loss`` /
                        ``action`` / ``pretrain`` / ``post_update`` /
                        ``on_obs`` / ``init_state``). The SAC loop
                        folds the stack at each phase. See
                        :mod:`ajax.extensions` for the catalogue
                        (``IBRL``, ``LCBGatedBootstrap``, ``CriticBlend``,
                        ``MCVarianceCorrection``, ``ValueBox``,
                        ``OnlineBC``, ``ResidualPolicy``, ``PhiRefresh``,
                        ``EDGEExploration``, ``JSRLCurriculum``,
                        ``ExpertGuidance``, ``ExpertObsAugmentation``,
                        ``MCPretrain``, ``BellmanPretrain``).

    Several kwargs above are kept on the function signature because they
    thread into init / collection / pipeline at a level the extension
    framework doesn't reach yet (``use_residual_rl``, ``jsrl_curriculum``,
    ``use_box``, ``use_bellman_critic_pretrain``, ``use_pid_policy``,
    ``augment_obs_with_expert_action``, ``use_train_frac``,
    ``normalize_obs_running``, ``store_policy_action``,
    ``extra_critic_head_*``, etc.). They mirror the corresponding
    extension's "static" flag where applicable.
    """
    # Phase 5: the back-compat shim that turned legacy boolean kwargs
    # into auto-appended Extensions (``_resolve_extension_stack`` +
    # ``_auto_append_*`` + the ``_locals.get(...)`` rebinding block) was
    # stripped. ``extensions=`` is now the sole surface for composing
    # research features (target modifiers, online-BC, PhiRefresh, EDGE,
    # MCPretrain, …). What remains is context injection: filling
    # SAC-factory-only fields (env / network / critic-optimizer config,
    # resolved action_dim, the shared buffer / gamma / reward_scale)
    # onto a few Extension instances that can't know them at
    # construction time.

    # If no separate eval policy provided, fall back to the training policy
    # (which may be None for vanilla SAC — in that case no expert bias logged)
    _eval_expert_policy = (
        eval_expert_policy if eval_expert_policy is not None else expert_policy
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    # Bind the SAC-factory-only context onto every extension in the
    # stack (no-op for extensions that don't override ``bind_to_agent``).
    # PhiRefresh / MCPretrain / ExpertObsAugmentation / ResidualPolicy
    # each pick up the kwargs they need from the agent context and
    # return a frozen instance populated with the resolved values. SAC
    # does not need to know which extension consumes which kwarg —
    # see :meth:`Extension.bind_to_agent` for the per-class contract.
    _resolved_action_dim = (
        action_dim_override
        if action_dim_override is not None
        else get_action_dim(env_args.env, env_args.env_params)
    )
    _use_phi_refresh = any(isinstance(e, _PhiRefresh) for e in extensions)
    extensions = tuple(
        ext.bind_to_agent(
            env_args=env_args,
            network_args=network_args,
            critic_optimizer_args=critic_optimizer_args,
            num_critics=num_critics,
            buffer=buffer,
            mode=mode,
            gamma=agent_config.gamma,
            reward_scale=agent_config.reward_scale,
            total_timesteps=total_timesteps,
            use_train_frac=use_train_frac,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            use_phi_refresh=_use_phi_refresh,
            mc_preloaded_data=mc_preloaded_data,
            action_dim=_resolved_action_dim,
        )
        for ext in extensions
    )

    _recurrent = network_args.memory is not None
    if _recurrent:
        # Expert-guidance features (and their Extension replacements) are
        # orthogonal to memory and untested with sequence replay; fail
        # loudly instead of silently misbehaving.
        unsupported_recurrent_options(
            "SAC",
            expert_policy=expert_policy,
            pid_actor_config=pid_actor_config,
            extensions=(tuple(extensions) or None),
            # SAC always builds a default CloningConfig; only actual
            # pre-training (pre_train_n_steps > 0) conflicts with memory.
            cloning_pretrain=(
                cloning_args
                if cloning_args is not None and cloning_args.pre_train_n_steps > 0
                else None
            ),
        )

    if logging_config is not None:
        start_async_logging()

    pre_train_n_steps = cloning_args.pre_train_n_steps if cloning_args else 0
    num_updates = total_timesteps // env_args.n_envs

    # ------------------------------------------------------------------
    # Fresh-init path: build the agent state and run all one-shot
    # initialization (MC / Bellman critic pretraining,
    # behavioural-cloning pretraining). This is *only* invoked on a fresh
    # run; on resume the shared helper reuses ``initial_state`` directly
    # so none of this expensive one-shot work is re-run.
    # ------------------------------------------------------------------
    def init_fn(key, index):
        """Build a fresh SAC agent state with all one-shot pretraining."""
        # Three keys: the third once seeded a user init hook; splitting
        # three keeps init_key and expert_key, hence every run, unchanged.
        init_key, expert_key, _ = jax.random.split(key, 3)

        agent_state = init_SAC(
            key=init_key,
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
            extra_critic_head_names=extra_critic_head_names,
            extra_critic_head_dims=extra_critic_head_dims,
        )

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
            agent_state = _stack.fold_init_states(agent_state, _ext_key)

        # MC critic pre-training now lives on
        # :meth:`MCPretrain.pretrain`. The ExtensionStack fold is the
        # framework-standard wiring point: when an :class:`MCPretrain`
        # is in ``extensions`` it populates
        # ``agent_state.expert_critic_params`` + ``expert_v_min/v_max``
        # (and optionally ``expert_critic_state`` for PhiRefresh);
        # otherwise the fold is identity. Empty stack ⇒ no-op. The
        # ``expert_key`` here threads the same byte-identical RNG slot
        # the legacy inline block consumed for
        # ``get_initialized_critic``. ``ext_state`` was already
        # populated above and is overwritten with the fold's result so
        # any state changes a pretrain phase makes propagate.
        if extensions:
            _stack_for_pretrain = ExtensionStack(extensions)
            agent_state = _stack_for_pretrain.fold_pretrain(
                agent_state, jnp.asarray(0), expert_key, total_timesteps
            )
            # ``use_box`` value-box bounds == the MC-pretrain v_min/v_max,
            # which are persisted on ``agent_state`` above; the scan-fn
            # builder reads them back from there (see make_scan_fn).

        if expert_policy is not None and use_bellman_critic_pretrain:
            agent_state = pretrain_critic_bellman(
                agent_state=agent_state,
                recurrent=_recurrent,
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

        # The collection-time substitution extensions (EDGE / ValueBox /
        # JSRL) and the four target-mod extensions (IBRL /
        # LCBGatedBootstrap / CriticBlend / MCVarianceCorrection) now
        # fold through the ExtensionStack — collection-time via
        # ``stack.action(...)`` (the action pipeline dispatches in the
        # canonical legacy ordering), TD-target via ``stack.on_target``.
        _extension_stack = ExtensionStack(extensions)

        # The pipeline carries the SAC-side bookkeeping (warmup mix,
        # ``is_expert_flag``, ``buffer_action``, expert-state threading)
        # and the gain-policy short-circuit; the EDGE / ValueBox / JSRL
        # gates run through the extension stack passed in here.
        _action_pipeline = make_action_pipeline(
            expert_policy=expert_policy,
            recurrent=_recurrent,
            env_args=env_args,
            extension_stack=_extension_stack,
            box_v_min=_box_v_min,
            box_v_max=_box_v_max,
            expert_fraction=expert_fraction,
            use_residual_rl=use_residual_rl,
            residual_scale=residual_scale,
            use_pid_policy=use_pid_policy,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            store_policy_action=store_policy_action,
            total_timesteps=total_timesteps,
        )

        # ResidualPolicy owns the ``clip(a_expert + scale·a_pi, -1, 1)``
        # math via :meth:`ResidualPolicy.transform_action`. The actor-
        # loss / TD-target call sites still consume a thin callable
        # (signature ``(actions, raw_obs, a_expert_precomputed) ->
        # actions``); we build it here off the first
        # :class:`ResidualPolicy` in the stack so the math lives on the
        # Extension and the legacy ``make_policy_action_transform``
        # builder is gone. ``None`` ⇒ pure SAC actor loss.
        # ResidualPolicy owns both the actor-loss / TD-target residual
        # transform and the eval-time transform. The factory pulls them
        # off the first :class:`ResidualPolicy` in the stack via the
        # extension's :meth:`build_policy_transform` /
        # :meth:`build_eval_transform` methods — self-contained
        # replacements for the legacy SAC-side ``_build_residual_*``
        # helpers. ``None`` ⇒ pure SAC actor loss / default box-based
        # handover.
        from ajax.extensions.expert import ResidualPolicy, first_of_type

        _rp = first_of_type(extensions, ResidualPolicy)
        _policy_action_transform = (
            _rp.build_policy_transform(expert_policy) if _rp is not None else None
        )

        # Eval transform: ``None`` when ``use_pid_policy`` is set
        # (gain-mode handles its own eval transform in
        # ``step_environment``) or when no ResidualPolicy is present.
        _eval_action_transform = (
            None if (use_pid_policy or _rp is None) else _rp.build_eval_transform()
        )

        # Periodic φ* refresh now lives on
        # :meth:`PhiRefresh.post_update`; the SAC loop folds
        # ``stack.post_update(...)`` inside ``training_iteration`` at
        # the start of each ``do_update``. The legacy
        # ``runtime_maintenance`` builder is gone — the auto-append
        # shim below has already copied ``buffer`` /
        # ``agent_config.gamma`` / ``agent_config.reward_scale`` onto
        # any PhiRefresh instance in ``extensions``.

        training_iteration_scan_fn = partial(
            training_iteration,
            buffer=buffer,
            recurrent=_recurrent,
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
            expert_policy=expert_policy,
            eval_expert_policy=_eval_expert_policy,
            use_expert_guidance=use_expert_guidance,
            num_critic_updates=num_critic_updates,
            expert_mix_fraction=expert_mix_fraction,
            augment_obs_with_expert_action=augment_obs_with_expert_action,
            augment_obs_with_expert_state=augment_obs_with_expert_state,
            policy_update_start=policy_update_start,
            alpha_update_start=alpha_update_start,
            fixed_alpha=fixed_alpha,
            target_entropy_initial=target_entropy_initial,
            target_entropy_ramp_frac=target_entropy_ramp_frac,
            action_pipeline=_action_pipeline,
            extension_stack=_extension_stack,
            policy_action_transform=_policy_action_transform,
            eval_action_transform=_eval_action_transform,
            pid_gain_policy=use_pid_policy,
            next_expert_fn=make_next_expert_fn(expert_policy),
            extra_eval_metrics=compose_eval_metrics(
                None, _extension_stack, total_timesteps
            ),
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
