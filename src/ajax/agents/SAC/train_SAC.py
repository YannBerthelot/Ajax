from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents.loop import TrainLoop, actor_step, critic_step, when
from ajax.agents.recurrent import (
    RecurrentCarries,
    actor_dist,
    sample_replay,
    unsupported_recurrent_options,
)
from ajax.agents.SAC import core
from ajax.agents.SAC.action_pipeline import make_action_pipeline
from ajax.agents.SAC.core import (
    TemperatureAuxiliaries,
    update_target_networks,
    update_temperature,
)
from ajax.agents.SAC.expert import collect_and_store_expert_transitions
from ajax.agents.SAC.state import SACConfig, SACState
from ajax.buffers.dataset import TransitionDataset
from ajax.controller import executed_actions
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack, holds_expert, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.distributions import latent
from ajax.networks.networks import action_value, predict_value
from ajax.perf_utils import final_aux_scan
from ajax.schedule import Schedule, at, linear
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)
from ajax.types import BufferType


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array  # α·log π - Q, before the extensions' terms
    log_pi: jax.Array  # entropy proxy; tracks target_entropy
    policy_std: jax.Array  # mean unsquashed std; lower = more deterministic
    q_min: jax.Array  # Q(s, π(s)): what policy optimises
    # Before clipping; the loss builds these metrics before the gradient
    # exists, update_policy fills it in.
    grad_norm: jax.Array | float = jnp.nan


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    q_pred_min: jax.Array  # min over ensemble
    var_preds: jax.Array  # inter-critic variance
    expert_frac_in_buffer: jax.Array  # fraction of sampled batch flagged as expert
    grad_norm: jax.Array  # before clipping


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics


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
    num_critics: int = 2,
    expert_buffer_n_steps: int = 20_000,
    count_episode_steps: bool = False,
    store_raw_next_obs: bool = False,
    store_expert_data: bool = False,
    store_return_to_go: bool = False,
) -> SACState:
    rng, init_key, collector_key, expert_key, eval_rng = jax.random.split(key, 5)

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
        collector_extras={
            "expert_policy": expert_policy,
            "include_raw_next_obs": store_raw_next_obs,
            "include_return_to_go": store_return_to_go,
            "include_raw_obs": store_expert_data,
            "include_is_expert": store_expert_data,
        },
    )
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

    # A per-env step_in_episode counter for the extensions that read it;
    # left None otherwise, so the collector skips its reset/increment.
    if count_episode_steps:
        collector_state = collector_state.replace(
            step_in_episode=jnp.zeros((env_args.n_envs,), dtype=jnp.int32)
        )

    return SACState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=core.create_alpha_train_state(**to_state_dict(alpha_args)),
        collector_state=collector_state,
    )


def update_value_functions(
    agent_state: SACState,
    transition: Transition,
    recurrent: bool,
    gamma: float,
    extension_stack: ExtensionStack,
    reward_scale: float = 1.0,
    total_timesteps: int = 1,
    next_action_transform: Optional[Callable] = None,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[SACState, ValueAuxiliaries, dict]:
    """The critic step on SAC's soft target. ``next_action_transform``
    (residual RL) evaluates the bootstrap at the transformed next action,
    from the expert's stored action at s', so the critic is queried where
    it was trained."""
    agent_state, key = agent_state.next_keys()
    # A truncated row bootstraps on its final observation (bootstrap_obs).
    dones = transition.terminated
    target_q = core.compute_td_target(
        actor_state=agent_state.actor_state,
        critic_state=agent_state.critic_state,
        next_observations=transition.next_obs,
        dones=dones,
        rewards=transition.reward,
        gamma=gamma,
        alpha=jnp.exp(agent_state.alpha.params["log_alpha"]),
        rng=key,
        recurrent=recurrent,
        reward_scale=reward_scale,
        next_action_transform=next_action_transform,
        next_a_expert=(
            transition.next_a_expert if next_action_transform is not None else None
        ),
        carries=carries,
    )
    # The target extensions (MCVarianceCorrection's variance) read the
    # critics' predictions, from a pass of their own: the target comes
    # before the gradient-bearing pass.
    q_preds = None
    if "on_target" in extension_stack.implemented_phases():
        q_preds = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.params,
            x=jnp.concatenate(
                (transition.obs, jax.lax.stop_gradient(transition.action)), axis=-1
            ),
        )

    def value_loss(params: FrozenDict, target_q: jax.Array) -> Tuple[jax.Array, Any]:
        return core.critic_loss_fn(
            params,
            agent_state.critic_state,
            transition.obs,
            transition.action,
            target_q,
            carries=carries,
        )

    critic_state, core_aux, grad_norm, metrics = critic_step(
        agent_state,
        transition,
        target_q,
        value_loss,
        extension_stack,
        key,
        total_timesteps,
        rewards=transition.reward,
        dones=dones,
        gamma=gamma,
        reward_scale=reward_scale,
        # The target's key (IBRL and LCBGatedBootstrap redraw its
        # next-action sample), the critics' predictions and the memory flag.
        target_extras={"rng_key": key, "q_preds": q_preds, "recurrent": recurrent},
    )
    aux = ValueAuxiliaries(
        critic_loss=core_aux.critic_loss,
        q_pred_min=core_aux.q_pred_min,
        var_preds=core_aux.var_preds,
        expert_frac_in_buffer=jnp.zeros(1),
        grad_norm=grad_norm,
    )
    return agent_state.replace(critic_state=critic_state), aux, metrics


def update_policy(
    agent_state: SACState,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    expert_policy: Optional[Callable] = None,
    a_expert_precomputed: Optional[jax.Array] = None,
    policy_action_transform: Optional[Callable] = None,
    total_timesteps: int = 1,
    carries: Optional[RecurrentCarries] = None,
    replay_actions: Optional[jax.Array] = None,
) -> Tuple[SACState, PolicyAuxiliaries, jax.Array, dict]:
    """The actor step on SAC's soft policy loss (core.soft_policy_loss),
    the critics scoring ``policy_action_transform``'s executed actions with
    a residual policy. Returns the state, ``aux``, the log-probs of a fresh
    sample of the stepped actor for update_temperature (CleanRL's SAC order;
    cited from memory, unverified) and the extensions' metrics."""
    agent_state, key, sample_key = agent_state.next_keys(2)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    # The env's own observations, what the expert reads (a run with an
    # expert stores them).
    raw = raw_observations
    transform = None
    if policy_action_transform is not None:

        def transform(actions: jax.Array) -> jax.Array:
            return policy_action_transform(actions, raw, a_expert_precomputed)

    # The actor-loss extensions get the expert's action as collection stored
    # it, from its tracked state (the clone target of OnlineBC and
    # ImitationLoss).
    a_expert = None
    if (
        expert_policy is not None
        and "actor_loss" in extension_stack.implemented_phases()
    ):
        a_expert = a_expert_precomputed
        if a_expert is None:
            a_expert = jax.lax.stop_gradient(expert_policy(raw))

    def policy_loss(params: FrozenDict) -> Tuple[jax.Array, Any, jax.Array]:
        loss, (log_probs, q, pi) = core.soft_policy_loss(
            params,
            agent_state.actor_state,
            agent_state.critic_state,
            observations,
            alpha,
            key,
            carries,
            action_transform=transform,
        )
        aux = PolicyAuxiliaries(
            policy_loss=loss,
            log_pi=log_probs.mean(),
            policy_std=latent(pi).stddev().mean(),
            q_min=q.mean(),
        )
        return loss, aux, pi.mean()

    actor_state, aux, grad_norm, metrics = actor_step(
        agent_state,
        policy_loss,
        extension_stack,
        key,
        total_timesteps,
        observations=observations,
        # The replay keeps them only for a run with an expert: None
        # without one, never the observations passed off as raw.
        raw_observations=raw_observations,
        a_expert=a_expert,
        actions=replay_actions,
        critic_state=agent_state.critic_state,
    )
    pi = actor_dist(actor_state, actor_state.params, observations, carries)
    _, log_probs = pi.sample_and_log_prob(seed=sample_key)
    return (
        agent_state.replace(actor_state=actor_state),
        aux.replace(grad_norm=grad_norm),
        jax.lax.stop_gradient(log_probs),
        metrics,
    )


def update_agent(
    agent_state: SACState,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    target_entropy: Schedule,
    tau: float,
    extension_stack: ExtensionStack,
    num_critic_updates: int = 1,
    reward_scale: float = 1.0,
    expert_policy: Optional[Callable] = None,
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    fixed_alpha: bool = False,
    total_timesteps: int = 1,
    policy_action_transform: Optional[Callable] = None,
    burn_in: int = 8,
    stored_state: bool = False,
) -> Tuple[SACState, AuxiliaryLogs]:
    agent_state, sample_key = agent_state.next_keys()
    # Sequence replay burns the carries in (R2D2-style); expert features are
    # rejected with memory upstream (make_train), so the expert blocks below
    # are trace-time no-ops there.
    transition, carries = sample_replay(
        agent_state, buffer, sample_key, recurrent, burn_in, stored_state
    )
    # The share of the rows an expert acted in (no stored flags: none).
    expert_frac_in_buffer = (
        jnp.zeros(()) if transition.is_expert is None else transition.is_expert.mean()
    )

    a_expert_precomputed = None
    # The actor-loss extensions (OnlineBC) take the expert action from
    # here, phi* or not.
    has_actor_loss = "actor_loss" in extension_stack.implemented_phases()
    needs_expert = expert_policy is not None and (
        policy_action_transform is not None or has_actor_loss
    )
    if needs_expert:
        assert expert_policy is not None
        # Prefer the a_expert stored in the buffer at collection time
        # (computed with the correct stateful expert internal state).
        # Fall back to a fresh stateless expert call only if the buffer
        # transition predates the schema change (legacy run).
        if transition.a_expert is not None:
            a_expert_precomputed = jax.lax.stop_gradient(transition.a_expert)
        else:
            # The env's own observations, which a run with an expert stores.
            a_expert_precomputed = jax.lax.stop_gradient(
                expert_policy(transition.raw_obs)
            )

    # See ajax.perf_utils.final_aux_scan: carry-only scan that exposes
    # last-step aux without materialising the full ys axis.
    def critic_update_step(state, _):
        state, aux, metrics = update_value_functions(
            state,
            transition,
            recurrent,
            gamma,
            extension_stack,
            reward_scale=reward_scale,
            total_timesteps=total_timesteps,
            next_action_transform=policy_action_transform,
            carries=carries,
        )
        return state, (aux, metrics)

    agent_state, (aux_value, value_metrics) = final_aux_scan(
        critic_update_step,
        agent_state,
        length=num_critic_updates,
    )

    new_agent_state, aux_policy, policy_log_probs, policy_metrics = update_policy(
        observations=transition.obs,
        agent_state=agent_state,
        raw_observations=transition.raw_obs,
        expert_policy=expert_policy,
        a_expert_precomputed=a_expert_precomputed,
        policy_action_transform=policy_action_transform,
        extension_stack=extension_stack,
        total_timesteps=total_timesteps,
        carries=carries,
        replay_actions=transition.action,
    )
    # The policy step is computed from the start, its log-probs feeding the
    # temperature step, and kept from policy_update_start.
    timestep = agent_state.collector_state.timestep
    agent_state = when(
        timestep >= policy_update_start, lambda _: new_agent_state, agent_state
    )

    # The temperature loss is computed for its logged auxiliaries; its step
    # is kept from alpha_update_start, never with ``fixed_alpha`` (alpha stays
    # at ``alpha_init``).
    new_agent_state_temp, aux_temperature = update_temperature(
        agent_state, policy_log_probs, jnp.asarray(at(target_entropy, timestep))
    )
    alpha_ready = jnp.logical_and(not fixed_alpha, timestep >= alpha_update_start)
    agent_state = when(alpha_ready, lambda _: new_agent_state_temp, agent_state)

    agent_state = update_target_networks(agent_state, tau=tau)

    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=aux_value.replace(expert_frac_in_buffer=expert_frac_in_buffer),
        extensions=merge_metrics(value_metrics, policy_metrics),
    )
    return agent_state, aux


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
    end_timestep: Optional[int] = None,
    expert_policy: Optional[Callable] = None,
    eval_expert_policy: Optional[Callable] = None,
    fixed_alpha: bool = False,
    num_critics: int = 2,
    expert_buffer_n_steps: int = 20_000,
    num_critic_updates: int = 1,
    policy_update_start: int = 2_000,
    alpha_update_start: int = 2_000,
    expert_fraction: float = 0.7,
    target_entropy_initial: Optional[float] = None,
    target_entropy_ramp_frac: float = 0.5,
    store_policy_action: bool = False,
    offline_data: Optional[TransitionDataset] = None,
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
    agent_context = {
        "env_args": env_args,
        "actor_optimizer_args": actor_optimizer_args,
        "network_args": network_args,
        "critic_optimizer_args": critic_optimizer_args,
        "num_critics": num_critics,
        "buffer": buffer,
        "gamma": agent_config.gamma,
        "reward_scale": agent_config.reward_scale,
        "extensions": tuple(extensions),
        "expert_policy": expert_policy,
        "recurrent": recurrent,
        "update_value_fn": update_value_functions,
        "update_target_fn": update_target_networks,
    }
    stack = ExtensionStack(extensions).bind_to_agent(**agent_context)
    if recurrent:
        # Expert-guidance features (and their Extension replacements) are
        # orthogonal to memory and untested with sequence replay; fail
        # loudly instead of silently misbehaving.
        unsupported_recurrent_options(
            "SAC",
            expert_policy=expert_policy,
            extensions=(stack.extensions or None),
        )
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        stack.extensions,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        expert_policy=expert_policy,
        offline_data=offline_data,
    )

    def init(key: jax.Array) -> SACState:
        return init_SAC(
            key=key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            stored_state=agent_config.stored_state,
            alpha_args=alpha_args,
            buffer=buffer,
            expert_policy=expert_policy,
            num_critics=num_critics,
            expert_buffer_n_steps=(
                expert_buffer_n_steps if expert_policy is not None else 0
            ),
            count_episode_steps=any(e.reads_step_in_episode for e in stack),
            # Kept only for an extension that declares it reads it.
            store_raw_next_obs="raw_next_obs" in stack.buffer_fields(),
            store_expert_data=holds_expert(expert_policy, stack),
            store_return_to_go="return_to_go" in stack.buffer_fields(),
        )

    # A residual policy (``transform_action``) makes the executed action
    # residual: in collection (the action pipeline), the actor loss and TD
    # target (the critics score executed actions) and evaluation (the loop's
    # controller). Gain mode (``actuate``) acts through the default
    # pipeline, which the shared collector actuates.
    residual: Any = next(
        (
            t
            for t in env_args.transforms
            if "transform_action" in t.implemented_phases()
        ),
        None,
    )
    policy_action_transform = None
    if residual is not None:

        def policy_action_transform(actions, raw_obs, a_expert):
            del raw_obs
            return executed_actions(env_args, actions, a_expert)

    # A ramp from target_entropy_initial (low: a BC-warmed policy mean rolls
    # out near-deterministically and stays alive on brittle envs) to the
    # standard target over the first target_entropy_ramp_frac of training.
    target_entropy: Schedule = (
        agent_config.target_entropy
        if target_entropy_initial is None
        else linear(
            target_entropy_initial,
            agent_config.target_entropy,
            target_entropy_ramp_frac * total_timesteps,
        )
    )

    def update(agent_state: SACState, _transition: Transition) -> Any:
        # The step just collected is in the buffer: SAC samples it from there.
        return update_agent(
            agent_state,
            buffer=buffer,
            recurrent=recurrent,
            gamma=agent_config.gamma,
            target_entropy=target_entropy,
            tau=agent_config.tau,
            reward_scale=agent_config.reward_scale,
            expert_policy=expert_policy,
            policy_update_start=policy_update_start,
            alpha_update_start=alpha_update_start,
            fixed_alpha=fixed_alpha,
            num_critic_updates=num_critic_updates,
            total_timesteps=total_timesteps,
            extension_stack=loop.stack,
            policy_action_transform=policy_action_transform,
            burn_in=agent_config.burn_in,
            stored_state=agent_config.stored_state,
        )

    # The pipeline carries the SAC-side bookkeeping (warmup mix,
    # ``is_expert_flag``, ``buffer_action``, expert-state threading) and
    # the gain-policy short-circuit; the EDGE / ValueBox / JSRL gates run
    # through the extension stack.
    gain_mode = env_args.policy_action_dim is not None
    action_pipeline = make_action_pipeline(
        expert_policy=None if gain_mode else expert_policy,
        recurrent=recurrent,
        env_args=env_args,
        extension_stack=loop.stack,
        expert_fraction=expert_fraction,
        residual=residual,
        store_policy_action=store_policy_action,
        total_timesteps=total_timesteps,
    )
    return loop.off_policy(
        init,
        update,
        agent_config.learning_starts,
        recurrent=recurrent,
        collect_kwargs={
            "buffer": buffer,
            "action_pipeline": action_pipeline,
            "store_hidden": recurrent and agent_config.stored_state,
        },
        eval_kwargs={
            # Evaluate only in the last 20% of training (HPO phases).
            "sweep": logging_config is not None and logging_config.sweep,
            # The training expert feeds the observation features, the
            # residual's base and the PID gains; eval_expert_policy, when
            # given, only the expert's baseline return.
            "expert_policy": expert_policy,
            "eval_expert_policy": eval_expert_policy,
            "value_estimate": ValueEstimate(
                action_value, agent_config.gamma, agent_config.reward_scale
            ),
        },
    )
