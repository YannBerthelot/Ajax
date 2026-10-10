"""PPO (Schulman et al., 2017) with brax's minibatch geometries.

Each iteration collects an ``(n_steps, n_envs)`` rollout, then runs
``n_epochs`` passes of a critic and an actor step on every minibatch of it
(the clipped surrogate: :mod:`ajax.agents.PPO.core`). The rollout's
:class:`Geometry` decides how it is minibatched and where GAE is computed:

* ``time``: fragments of ``length`` steps (whole rollouts per env unless
  ``unroll_length``) shuffled into minibatches, GAE recomputed in every
  minibatch with the current critic (brax's ``compute_ppo_loss``);
* ``flat``: GAE once on the whole rollout, then a flat shuffle of the
  samples (when the fragments do not split evenly, e.g. one env);
* ``recurrent``: GAE once on the whole rollout, then a shuffle of whole
  sequences of ``length`` steps, each replayed from the carry it started
  from.
"""

import dataclasses
from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.tree_util import Partial as partial

from ajax.agents.PPO.core import (
    AuxiliaryLogs,
    PolicyAuxiliaries,
    ValueAuxiliaries,
    clipped_surrogate,
    policy_entropy,
    recompute_log_prob,
    resolve_clip_coef,
    resolve_num_minibatches,
    rollout_actions,
    run_epochs,
)
from ajax.agents.PPO.state import PPOConfig, PPOState
from ajax.agents.PPO.utils import (
    _compute_gae,
    get_minibatches_from_batch,
    get_minibatches_preserving_time,
    split_fragments,
)
from ajax.environments.interaction import (
    collect_experience,
    get_pi,
    get_pi_sequence,
    init_collector_state,
    preallocate_last_rollout,
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
    predict_value_sequence,
)
from ajax.perf_utils import build_resumable_train
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    Transition,
)


def init_PPO(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    pid_actor_config: Optional[PIDActorConfig] = None,
    normalize_obs_running: bool = False,
) -> PPOState:
    """The initial actor, critic and collector; with
    ``normalize_obs_running``, the agent-side observation statistics."""
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


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    value_targets: jax.Array,
    recurrent: bool,
    vf_coef: float = 1.0,
    extra_loss_fn: Optional[Callable] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """``vf_coef * 0.5 * MSE(V(obs), targets)`` plus ``extra_loss_fn``'s
    term. A recurrent critic replays the sequence from ``initial_hidden``,
    ``resets`` flagging the observations that start an episode."""
    if recurrent:
        v_preds, _ = predict_value_sequence(
            critic_state=critic_states,
            critic_params=critic_params,
            x=observations,
            resets=resets,
            initial_hidden=initial_hidden,
        )
    else:
        v_preds = predict_value(
            critic_state=critic_states,
            critic_params=critic_params,
            x=observations,
        )
    v_preds = v_preds.squeeze(0)  # the single critic keeps the ensemble axis

    loss = vf_coef * 0.5 * jnp.mean((v_preds - value_targets) ** 2)
    if extra_loss_fn is not None:
        loss = loss + extra_loss_fn(
            critic_params, critic_states, observations, value_targets
        )

    return loss, ValueAuxiliaries(
        critic_loss=loss,
        predictions=v_preds.mean().flatten(),
        targets=value_targets.mean().flatten(),
    )


def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    log_probs: jax.Array,
    gae: jax.Array,
    recurrent: bool,
    clip_coef: float,
    ent_coef: float,
    advantage_normalization: bool,
    extra_loss_fn: Optional[Callable] = None,
    raw_actions: Optional[jax.Array] = None,
    entropy_rng: Optional[jax.Array] = None,
    resets: Optional[jax.Array] = None,
    initial_hidden: Optional[Any] = None,
) -> Tuple[jax.Array, PolicyAuxiliaries]:
    """The clipped surrogate minus ``ent_coef`` times the entropy bonus,
    plus ``extra_loss_fn``'s term; with ``advantage_normalization`` the
    advantages are normalised per minibatch. A recurrent actor replays the
    sequence from ``initial_hidden`` (the live carry on ``actor_state`` has
    advanced past this rollout), ``resets`` flagging episode starts."""
    if recurrent:
        pi, _ = get_pi_sequence(
            actor_state=actor_state,
            actor_params=actor_params,
            obs=observations,
            resets=resets,
            initial_hidden=initial_hidden,
        )
    else:
        pi, _ = get_pi(actor_state, actor_params, observations)

    new_log_probs = recompute_log_prob(pi, actions, raw_actions)
    ratio = jnp.exp(new_log_probs - log_probs)

    if advantage_normalization:
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    surrogate, clip_fraction = clipped_surrogate(ratio, gae, clip_coef)
    loss_actor = surrogate.mean()
    entropy = policy_entropy(pi, entropy_rng).mean()

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


# ---------------------------------------------------------------------------
# Extension-stack loss terms
# ---------------------------------------------------------------------------
def _stack_critic_loss(
    extension_stack: Optional[ExtensionStack],
    agent_state: PPOState,
    total_timesteps: int,
) -> Optional[Callable]:
    """The stack's additive critic-loss term as a ``value_loss_function``
    ``extra_loss_fn``: ``(critic_params, critic_states, observations,
    value_targets) -> scalar``; ``None`` for an empty stack.

    ``agent_state`` (the iteration-time state, captured by closure) is
    what the extensions read ``ext_state`` from.
    """
    if not extension_stack:
        return None

    def critic_loss(critic_params, critic_states, observations, value_targets):
        _batch = {
            "observations": observations,
            "targets": value_targets,
            "critic_params": critic_params,
            "critic_state": critic_states,
        }
        return extension_stack.fold_critic_loss(
            agent_state,
            _batch,
            agent_state.collector_state.timestep,
            agent_state.rng,
            total_timesteps,
        )

    return critic_loss


def _stack_actor_loss(
    extension_stack: Optional[ExtensionStack],
    agent_state: PPOState,
    total_timesteps: int,
) -> Optional[Callable]:
    """The stack's additive actor-loss term as a ``policy_loss_function``
    ``extra_loss_fn``: ``(actor_params, actor_state) -> scalar``; ``None``
    for an empty stack."""
    if not extension_stack:
        return None

    def actor_loss(actor_params, actor_state):
        _batch = {
            "actor_params": actor_params,
            "actor_state": actor_state,
        }
        return extension_stack.fold_actor_loss(
            agent_state,
            _batch,
            agent_state.collector_state.timestep,
            agent_state.rng,
            total_timesteps,
        )

    return actor_loss


# ---------------------------------------------------------------------------
# Minibatch geometry
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Geometry:
    """How an update minibatches its rollout (the module docstring)."""

    kind: str  # "time", "flat" or "recurrent"
    num_minibatches: int
    length: int  # steps per fragment ("time") or sequence ("recurrent")


def minibatch_geometry(
    agent_config: PPOConfig, n_steps: int, n_envs: int, recurrent: bool
) -> Geometry:
    """The rollout's :class:`Geometry`.

    ``time`` needs fragments that split evenly into more than one
    minibatch: recomputing GAE on a single minibatch, the whole rollout,
    with a critic that just stepped only makes the value targets oscillate
    (the flake of the one-env probing fixture). A recurrent rollout is a
    pool of sequences of ``bptt_length`` steps (whole rollouts by default),
    in one minibatch when the pool does not split evenly (one env).
    """
    num_minibatches = resolve_num_minibatches(agent_config)
    if recurrent:
        length = agent_config.bptt_length or n_steps
        if n_steps % length:
            raise ValueError(f"bptt_length ({length}) must divide n_steps ({n_steps}).")
        if num_minibatches > 1 and n_steps // length * n_envs % num_minibatches == 0:
            return Geometry("recurrent", num_minibatches, length)
        return Geometry("recurrent", 1, length)
    unroll = agent_config.unroll_length
    if num_minibatches > 1:
        if (
            unroll is not None
            and n_steps % unroll == 0
            and n_steps // unroll * n_envs % num_minibatches == 0
        ):
            return Geometry("time", num_minibatches, unroll)
        if n_envs % num_minibatches == 0:
            return Geometry("time", num_minibatches, n_steps)
    return Geometry("flat", num_minibatches, n_steps)


def _recurrent_values(
    agent_state: PPOState, rollout: Transition, start: PPOState, length: int
) -> tuple[PPOState, jax.Array, jax.Array, jax.Array, tuple]:
    """A recurrent rollout's values, next values and reset flags, and the
    actor's and critic's carries at the start of each sequence of
    ``length`` steps.

    One critic pass gives the values and advances the critic carry, which
    collection never steps: it ends as the next rollout's start carry. Run
    as a scan over the sequences, it also yields their start carries. The
    actor's are recomputed sequence by sequence with the current params (one
    more pass, skipped for whole-rollout sequences, whose start carry is
    exact): a sequence never starts from a zero carry mid-episode, the
    variant that makes truncated BPTT worse than none. ``V(s_{t+1})`` is the
    shifted sequence plus one step on the collector's last observation: at a
    truncation, the reset observation (the usual recurrent-PPO
    approximation; GAE masks terminal steps).

    Returns the state (critic carry advanced), the values, the next values,
    the reset flags (``resets[t]``: ``obs[t]`` starts an episode) and the
    ``(actor, critic)`` start carries.
    """
    n_steps, n_envs = rollout.obs.shape[:2]
    n_sequences = n_steps // length
    initial_done = jnp.logical_or(
        start.collector_state.last_terminated, start.collector_state.last_truncated
    ).astype(bool)
    dones_seq = jnp.logical_or(rollout.terminated, rollout.truncated).squeeze(-1)
    resets = jnp.concatenate(
        [initial_done[None], dones_seq[:-1].astype(bool)], axis=0
    )  # (T, B)
    obs_seqs = rollout.obs.reshape(n_sequences, length, *rollout.obs.shape[1:])
    resets_seqs = resets.reshape(n_sequences, length, n_envs)
    critic, actor = agent_state.critic_state, agent_state.actor_state

    def critic_chunk(carry: Any, seq: tuple) -> tuple[Any, tuple]:
        obs, seq_resets = seq
        values, carry_next = predict_value_sequence(
            critic_state=critic,
            critic_params=critic.params,
            x=obs,
            resets=seq_resets,
            initial_hidden=carry,
        )
        return carry_next, (values, carry)

    critic_end, (seq_values, critic_starts) = jax.lax.scan(
        critic_chunk,
        jax.lax.stop_gradient(start.critic_state.hidden_state),
        (obs_seqs, resets_seqs),
    )
    # (S, num, L, B, 1) -> (num, T, B, 1) -> (T, B, 1)
    values = jnp.moveaxis(seq_values, 0, 1).reshape(
        seq_values.shape[1], n_steps, *seq_values.shape[3:]
    )
    values = values.squeeze(0)

    initial_actor_hidden = jax.lax.stop_gradient(start.actor_state.hidden_state)
    if n_sequences > 1:

        def actor_chunk(carry: Any, seq: tuple) -> tuple[Any, Any]:
            obs, seq_resets = seq
            _, carry_next = get_pi_sequence(actor, actor.params, obs, seq_resets, carry)
            return carry_next, carry

        _, actor_starts = jax.lax.scan(
            actor_chunk, initial_actor_hidden, (obs_seqs, resets_seqs)
        )
    else:
        actor_starts = jax.tree.map(lambda x: x[None], initial_actor_hidden)

    v_last, _ = predict_value_sequence(
        critic_state=critic,
        critic_params=critic.params,
        x=agent_state.collector_state.last_obs[None],
        resets=dones_seq[-1:].astype(bool),
        initial_hidden=critic_end,
    )
    next_values = jnp.concatenate([values[1:], v_last.squeeze(0)], axis=0)
    agent_state = agent_state.replace(
        critic_state=critic.replace(hidden_state=jax.lax.stop_gradient(critic_end))
    )
    carries = jax.lax.stop_gradient((actor_starts, critic_starts))
    return agent_state, values, next_values, resets, carries


def _split_carries(x: jax.Array, perm: jax.Array, num_minibatches: int) -> jax.Array:
    """``(S, B, ...)`` start carries to ``(k, S * B // k, ...)``, sequence
    ``s * B + b`` in the order of ``perm`` (as :func:`split_fragments`)."""
    x = x.reshape(-1, *x.shape[2:])
    x = jnp.take(x, perm, axis=0)
    return x.reshape(num_minibatches, -1, *x.shape[1:])


def _sequence_minibatches(
    batch: dict, carries: tuple, rng: jax.Array, num_minibatches: int, length: int
) -> dict:
    """A recurrent rollout's minibatches of whole sequences, each with the
    actor's and the critic's carry at its start (the critic's ensemble axis
    second)."""
    n_steps, n_envs = batch["obs"].shape[:2]
    perm = jax.random.permutation(rng, n_steps // length * n_envs)

    def split(x: jax.Array) -> jax.Array:
        return _split_carries(x, perm, num_minibatches)

    actor_starts, critic_starts = carries
    return split_fragments(batch, perm, num_minibatches, length) | {
        "actor_hidden": jax.tree.map(split, actor_starts),
        "critic_hidden": jax.tree.map(
            jax.vmap(split, in_axes=1, out_axes=1), critic_starts
        ),
    }


def _joint_clip(actor_grads: Any, critic_grads: Any) -> tuple[Any, Any]:
    """brax's global-norm clip at 1.0, over both networks' gradients."""
    actor_sq = sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(actor_grads))
    critic_sq = sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(critic_grads))
    scale = jnp.minimum(1.0, 1.0 / jnp.sqrt(actor_sq + critic_sq + 1e-12))
    return jax.tree.map(lambda g: g * scale, (actor_grads, critic_grads))


def _force_reset(
    agent_state: PPOState, env_args: EnvironmentConfig, mode: str
) -> PPOState:
    """Reset every env on a fresh key (brax's ``num_resets_per_eval``).

    The env wrapper's reset re-initialises its observation normaliser: the
    running statistics are kept, the fresh observation re-normalised with
    them.
    """
    reset_key, new_rng = jax.random.split(agent_state.rng)
    saved_norm = None
    if mode == "brax" and "normalization_info" in (
        agent_state.collector_state.env_state.info or {}
    ):
        saved_norm = agent_state.collector_state.env_state.info["normalization_info"]
    reset_keys = (
        jax.random.split(reset_key, env_args.n_envs) if mode == "gymnax" else reset_key
    )
    new_obs, new_env_state = reset(reset_keys, env_args.env, mode, env_args.env_params)
    if saved_norm is not None:
        fresh_obs_info = new_env_state.info["normalization_info"].obs
        saved_obs_info = saved_norm.obs
        # Undo fresh normalisation, re-apply saved-stats normalisation.
        # Match online_normalize: it uses ``mean(clipped_std, axis=0)``
        # to broadcast across envs (all rows of the batched stat are
        # identical post-Welford). Apply the same clip + mean here so
        # the recovered raw_obs is bit-for-bit what the env produced.
        fresh_std = jnp.clip(jnp.sqrt(fresh_obs_info.var + 1e-8), 1e-6, 1e6).mean(
            axis=0
        )
        fresh_mean = fresh_obs_info.mean.mean(axis=0)
        raw_obs = new_obs * fresh_std + fresh_mean
        saved_std = jnp.clip(jnp.sqrt(saved_obs_info.var + 1e-8), 1e-6, 1e6).mean(
            axis=0
        )
        saved_mean = saved_obs_info.mean.mean(axis=0)
        new_obs = (raw_obs - saved_mean) / saved_std
        new_env_state.info["normalization_info"] = saved_norm
        new_env_state = new_env_state.replace(obs=new_obs)
    new_collector = agent_state.collector_state.replace(
        _env_state=new_env_state, last_obs=new_obs
    )
    return agent_state.replace(collector_state=new_collector, rng=new_rng)


# ---------------------------------------------------------------------------
# The update
# ---------------------------------------------------------------------------
def update_agent(
    agent_state: PPOState,
    rollout: Transition,
    start: PPOState,
    agent_config: PPOConfig,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    total_n_updates: int,
    reward_shaping_fn: Optional[Callable] = None,
) -> tuple[PPOState, AuxiliaryLogs]:
    """One PPO update on the ``(n_steps, n_envs)`` ``rollout`` collected
    from ``start``: ``n_epochs`` passes over its minibatches, then brax's
    periodic forced reset."""
    rewards = rollout.reward
    if reward_shaping_fn is not None:
        rewards = rewards + reward_shaping_fn(agent_state, rollout)
    geometry = minibatch_geometry(agent_config, *rollout.obs.shape[:2], recurrent)
    actions, log_probs = rollout_actions(rollout)
    batch = {
        "obs": rollout.obs,
        "actions": actions,
        "log_probs": log_probs,
        "raw_actions": rollout.raw_action,
    }

    if geometry.kind == "time":
        # GAE in each minibatch, with the critic of that step.
        batch |= {
            "next_obs": rollout.next_obs,
            "terminated": rollout.terminated,
            "truncated": rollout.truncated,
            "rewards": rewards,
        }
    else:
        if recurrent:
            agent_state, values, next_values, resets, carries = _recurrent_values(
                agent_state, rollout, start, geometry.length
            )
            batch["resets"] = resets
        else:
            values = predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.critic_state.params,
                x=rollout.obs,
            ).squeeze(0)
            next_values = predict_value(
                critic_state=agent_state.critic_state,
                critic_params=agent_state.critic_state.params,
                x=rollout.next_obs,
            ).squeeze(0)
        gae, value_targets = _compute_gae(
            values=values,
            next_values=next_values,
            rewards=rewards,
            terminateds=rollout.terminated,
            truncateds=rollout.truncated,
            gamma=agent_config.gamma,
            gae_lambda=agent_config.gae_lambda,
        )
        if extension_stack:
            target_key, rng = jax.random.split(agent_state.rng)
            agent_state = agent_state.replace(rng=rng)
            target_batch = {
                "observations": rollout.obs,
                "next_observations": rollout.next_obs,
                "rewards": rewards,
                "terminated": rollout.terminated,
                "truncated": rollout.truncated,
                "gae": gae,
                "values": values,
                "next_values": next_values,
                "gamma": agent_config.gamma,
            }
            value_targets = extension_stack.fold_on_target(
                agent_state,
                target_batch,
                value_targets,
                agent_state.collector_state.timestep,
                target_key,
                total_timesteps,
            )
        batch |= {"targets": value_targets, "gae": gae}

    shuffle_key, rng = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)
    k, length = geometry.num_minibatches, geometry.length
    if geometry.kind == "time":
        minibatches = get_minibatches_preserving_time(batch, shuffle_key, k, length)
    elif geometry.kind == "recurrent":
        minibatches = _sequence_minibatches(batch, carries, shuffle_key, k, length)
    else:
        minibatches = get_minibatches_from_batch(batch, shuffle_key, k)

    # The extensions' loss terms read the state the epochs start from.
    critic_extra = _stack_critic_loss(extension_stack, agent_state, total_timesteps)
    actor_extra = _stack_actor_loss(extension_stack, agent_state, total_timesteps)

    def minibatch_step(agent_state: PPOState, mb: dict) -> tuple[PPOState, Any]:
        """A critic and an actor step, both from the state before either."""
        ent_rng, on_target_rng, new_rng = jax.random.split(agent_state.rng, 3)
        agent_state = agent_state.replace(rng=new_rng)
        critic, actor = agent_state.critic_state, agent_state.actor_state

        if geometry.kind == "time":
            # brax's compute_ppo_loss: GAE from the current critic, held
            # constant (no gradient reaches the critic through it).
            values = jax.lax.stop_gradient(
                predict_value(
                    critic_state=critic, critic_params=critic.params, x=mb["obs"]
                ).squeeze(0)
            )
            next_values = jax.lax.stop_gradient(
                predict_value(
                    critic_state=critic,
                    critic_params=critic.params,
                    x=mb["next_obs"],
                ).squeeze(0)
            )
            gae, value_targets = _compute_gae(
                values=values,
                next_values=next_values,
                rewards=mb["rewards"],
                terminateds=mb["terminated"],
                truncateds=mb["truncated"],
                gamma=agent_config.gamma,
                gae_lambda=agent_config.gae_lambda,
            )
            gae = jax.lax.stop_gradient(gae)
            value_targets = jax.lax.stop_gradient(value_targets)
            if extension_stack:
                target_batch = {
                    "observations": mb["obs"],
                    "next_observations": mb["next_obs"],
                    "rewards": mb["rewards"],
                    "terminated": mb["terminated"],
                    "truncated": mb["truncated"],
                    "gae": gae,
                    "values": values,
                    "next_values": next_values,
                    "gamma": agent_config.gamma,
                }
                value_targets = extension_stack.fold_on_target(
                    agent_state,
                    target_batch,
                    value_targets,
                    agent_state.collector_state.timestep,
                    on_target_rng,
                    total_timesteps,
                )
        else:
            gae, value_targets = mb["gae"], mb["targets"]

        resets = mb.get("resets")
        (_, v_aux), v_grads = jax.value_and_grad(value_loss_function, has_aux=True)(
            critic.params,
            critic,
            mb["obs"],
            value_targets,
            recurrent,
            vf_coef=agent_config.vf_coef,
            extra_loss_fn=critic_extra,
            resets=resets,
            initial_hidden=mb.get("critic_hidden"),
        )
        (_, p_aux), p_grads = jax.value_and_grad(policy_loss_function, has_aux=True)(
            actor.params,
            actor,
            mb["obs"],
            mb["actions"],
            mb["log_probs"],
            gae,
            recurrent,
            resolve_clip_coef(agent_config, agent_state.collector_state.timestep),
            agent_config.ent_coef,
            agent_config.normalize_advantage,
            extra_loss_fn=actor_extra,
            raw_actions=mb["raw_actions"],
            entropy_rng=ent_rng,
            resets=resets,
            initial_hidden=mb.get("actor_hidden"),
        )
        if agent_config.fused_grad_clip:
            p_grads, v_grads = _joint_clip(p_grads, v_grads)
        agent_state = agent_state.replace(
            critic_state=critic.apply_gradients(grads=v_grads),
            actor_state=actor.apply_gradients(grads=p_grads),
        )
        return agent_state, AuxiliaryLogs(policy=p_aux, value=v_aux)

    agent_state, aux = run_epochs(
        agent_state, minibatches, minibatch_step, agent_config.n_epochs
    )
    agent_state = agent_state.replace(n_updates=agent_state.n_updates + 1)

    # brax's num_resets_per_eval: every env reset every reset_every updates.
    # Playground envs draw a fresh initial state at every episode end by
    # default (build_env_from_id(fresh_reset=True)); with fresh_reset=False
    # their auto-reset replays the first reset state, and without these
    # resets a run sees only n_envs initial conditions.
    if agent_config.num_resets_per_eval > 0:
        num_evals_after_init = max(int(agent_config.num_evals) - 1, 1)
        reset_every = jnp.maximum(
            1,
            jnp.ceil(
                total_n_updates
                / (num_evals_after_init * agent_config.num_resets_per_eval)
            ).astype(jnp.int32),
        )
        should_reset = (agent_state.n_updates % reset_every == 0) & (
            agent_state.n_updates > 0
        )
        agent_state = jax.lax.cond(
            should_reset,
            lambda s: _force_reset(s, env_args, mode),
            lambda s: s,
            agent_state,
        )
    return agent_state, aux


def training_iteration(
    agent_state: PPOState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    agent_config: PPOConfig,
    total_timesteps: int,
    total_n_updates: int,
    log_frequency: Optional[int] = 1000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    reward_shaping_fn: Optional[Callable] = None,
    extension_stack: Optional[ExtensionStack] = None,
) -> tuple[PPOState, dict]:
    """Collect a rollout, update, fold the extensions' ``post_update``,
    then evaluate and log."""
    extension_stack = extension_stack or ExtensionStack()
    start = agent_state
    collect_scan_fn = partial(
        collect_experience, recurrent=recurrent, mode=mode, env_args=env_args
    )
    agent_state, transition = jax.lax.scan(
        collect_scan_fn, agent_state, xs=None, length=agent_config.n_steps
    )
    if agent_config.expose_recent_rollout:
        agent_state = agent_state.replace(last_rollout=transition)

    agent_state, aux = update_agent(
        agent_state,
        transition,
        start,
        agent_config,
        env_args,
        mode,
        recurrent,
        extension_stack,
        total_timesteps,
        total_n_updates,
        reward_shaping_fn,
    )

    # Extension post_update — folded after the per-iteration update loop.
    # Empty stack ⇒ identity.
    if extension_stack:
        _pu_rng, _pu_rng2 = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=_pu_rng2)
        agent_state = extension_stack.fold_post_update(
            agent_state,
            agent_state.collector_state.timestep,
            _pu_rng,
            total_timesteps,
        )

    _extra_eval = compose_eval_metrics(None, extension_stack, total_timesteps)
    agent_state, metrics_to_log = evaluate_and_log(
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
        extra_eval_metrics=_extra_eval,
    )
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
    reward_shaping_fn: Optional[Callable] = None,
    extensions: Sequence = (),
    normalize_obs_running: bool = False,
):
    """PPO's train function: an ``n_steps`` rollout per env, then one
    update, per iteration."""
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids)

    _recurrent = network_args.memory is not None
    if _recurrent and extensions:
        raise NotImplementedError("Recurrent PPO does not support extensions yet.")

    # The logging worker runs only for a backend: without one PPO still
    # evaluates and returns the metrics.
    if logging_config is not None and logging_config.backend:
        start_async_logging()

    num_updates = (total_timesteps // (env_args.n_envs * agent_config.n_steps)) + 1

    extension_stack = ExtensionStack(extensions)

    def init_fn(key, _index):
        init_key, pretrain_key = jax.random.split(key)
        agent_state = init_PPO(
            key=init_key,
            env_args=env_args,
            actor_optimizer_args=actor_optimizer_args,
            critic_optimizer_args=critic_optimizer_args,
            network_args=network_args,
            pid_actor_config=pid_actor_config,
            normalize_obs_running=normalize_obs_running,
        )
        agent_state = extension_stack.fold_init(
            agent_state, pretrain_key, total_timesteps
        )
        if agent_config.expose_recent_rollout:
            agent_state = preallocate_last_rollout(
                agent_state,
                agent_config.n_steps,
                recurrent=_recurrent,
                mode=mode,
                env_args=env_args,
            )
        return agent_state

    def make_scan_fn(_agent_state, _resume_from_state, _key, index):
        return partial(
            training_iteration,
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
            total_n_updates=num_updates,
            reward_shaping_fn=reward_shaping_fn,
            extension_stack=extension_stack,
        )

    return build_resumable_train(
        init_fn=init_fn,
        make_scan_fn=make_scan_fn,
        num_updates=num_updates,
    )
