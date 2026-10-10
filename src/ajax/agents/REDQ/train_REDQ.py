"""REDQ (Chen et al., 2021): Randomized Ensembled Double Q-learning.

SAC with an ensemble of ``num_critics`` critics updated
``num_critic_updates`` times per environment step, each on a fresh replay
minibatch with a target the min over a random subset of ``subset_size``
target critics, and an actor that
maximises the ensemble's mean Q. ``repulsion_coef`` adds a function-space
kernel repulsion between the critics (off by default).
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

from ajax.agents.cloning import CloningConfig, pretrain_on_expert
from ajax.agents.loop import TrainLoop, critic_step
from ajax.agents.recurrent import (
    RecurrentCarries,
    q_values,
    sample_replay,
    unsupported_recurrent_options,
)
from ajax.agents.REDQ.state import REDQConfig, REDQState
from ajax.agents.SAC import core
from ajax.agents.SAC.core import TemperatureAuxiliaries
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.modules.pid_actor import PIDActorConfig
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


@struct.dataclass
class PolicyAuxiliaries:
    policy_loss: jax.Array
    log_pi: jax.Array
    q_mean: jax.Array


@struct.dataclass
class ValueAuxiliaries:
    critic_loss: jax.Array
    target_q: jax.Array
    log_probs: jax.Array
    repulsion_loss: jax.Array
    # Scale-free ensemble divergence diagnostics (stop_gradient'd). Unlike
    # repulsion_loss (self-normalised by the median-heuristic bandwidth, so
    # ~O(1) regardless of actual spread), these reveal whether the ensemble
    # genuinely diversified in function space.
    ensemble_q_std: jax.Array
    mean_pairwise_q_dist: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries


def q_ensemble_divergence(q_preds: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Scale-free diagnostics for how spread out the critic ensemble is.

    Returns ``(ensemble_q_std, mean_pairwise_q_dist)``:
      * ensemble_q_std: std across the critic axis of the per-(s,a) Q
        prediction, averaged over the batch.
      * mean_pairwise_q_dist: mean L2 distance between the flattened
        per-critic Q-vectors over all off-diagonal pairs.

    Both are computed on stop_gradient'd predictions — they are telemetry
    only and must not contribute to the critic gradient.
    """
    q = jax.lax.stop_gradient(q_preds)
    n = q.shape[0]
    q_std = jnp.std(q, axis=0).mean()

    feats = q.reshape(n, -1)
    diffs = feats[:, None, :] - feats[None, :, :]
    dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-12)
    off_diag_sum = dists.sum() - jnp.trace(dists)
    mean_pairwise = off_diag_sum / (n * (n - 1))
    return q_std, mean_pairwise


def q_kernel_repulsion(q_preds: jax.Array) -> jax.Array:
    """Function-space SVGD-style RBF kernel repulsion penalty.

    q_preds has shape ``(num_critics, batch, ...)``. Each ensemble member's
    output is flattened to a feature vector; pairwise squared distances feed
    an RBF kernel with the median-heuristic bandwidth (Liu & Wang 2017).
    The returned scalar is the mean kernel value over all pairs — minimising
    it pushes members apart in function space.

    The bandwidth `h` is stop_gradient'd so the kernel adapts to the current
    spread of predictions without contributing a confounding gradient term.
    """
    n = q_preds.shape[0]
    feats = q_preds.reshape(n, -1)
    diffs = feats[:, None, :] - feats[None, :, :]
    sq_dists = jnp.sum(diffs**2, axis=-1)
    # Median heuristic over off-diagonal pairs. n*(n-1) off-diagonal entries;
    # `jnp.median` over the full matrix is fine because diagonal zeros are
    # only n out of n^2 and the median is dominated by the off-diagonal mass
    # for n >= 4. The +1e-8 floor avoids divide-by-zero at init when all
    # critics happen to predict identical values.
    h = jax.lax.stop_gradient(
        jnp.median(sq_dists) / (jnp.log(jnp.asarray(n, dtype=feats.dtype)) + 1e-8)
        + 1e-8
    )
    kernel = jnp.exp(-sq_dists / h)
    return kernel.mean()


def init_REDQ(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    buffer: BufferType,
    number_of_critics: int,
    window_size: int = 10,
    stored_state: bool = False,
    pid_actor_config: Optional[PIDActorConfig] = None,
) -> REDQState:
    rng, init_key, collector_key = jax.random.split(key, num=3)
    actor_state, critic_state, collector_state = core.init_soft_actor_critic(
        init_key,
        collector_key,
        env_args,
        actor_optimizer_args,
        critic_optimizer_args,
        network_args,
        buffer,
        num_critics=number_of_critics,
        window_size=window_size,
        stored_state=stored_state,
        pid_actor_config=pid_actor_config,
    )
    return REDQState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=core.create_alpha_train_state(**to_state_dict(alpha_args)),
        collector_state=collector_state,
    )


def compute_redq_td_target(
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    subset_size: int,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, jax.Array]:
    """REDQ bellman target: min over a random subset of target critics.

    Returns the stop_gradient'd target and the next actions' log-probs.
    """
    rewards = rewards * reward_scale
    sample_key, idx_sample_key = jax.random.split(rng)
    next_actions, log_probs = core.sample_next_actions(
        actor_state, next_observations, sample_key, carries
    )
    # The ensemble axis stays leading in sequence mode too, so the subset
    # sampling is the same.
    q_targets = q_values(
        critic_states,
        critic_states.target_params,
        next_observations,
        next_actions,
        carries,
        bootstrap=True,
    )
    sampled_indexes = jax.random.choice(
        idx_sample_key, q_targets.shape[0], shape=(subset_size,), replace=False
    )
    min_q_target = jnp.min(q_targets[sampled_indexes], axis=0)

    target = rewards + gamma * (1.0 - dones) * (min_q_target - alpha * log_probs)
    return jax.lax.stop_gradient(target), log_probs


def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    actions: jax.Array,
    target_q: jax.Array,
    next_log_probs: jax.Array,
    repulsion_coef: float = 0.0,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """Every critic regresses on the one target, plus the kernel repulsion."""
    q_preds = q_values(critic_states, critic_params, observations, actions, carries)
    bellman_loss = jnp.sum(
        jnp.mean((q_preds - target_q) ** 2, axis=tuple(range(1, q_preds.ndim)))
        / q_preds.ndim
    )
    repulsion = q_kernel_repulsion(q_preds)
    total_loss = bellman_loss + repulsion_coef * repulsion
    q_std, mean_pairwise_q_dist = q_ensemble_divergence(q_preds)
    return total_loss, ValueAuxiliaries(
        critic_loss=total_loss,
        target_q=target_q.mean().flatten(),
        log_probs=next_log_probs.mean().flatten(),
        repulsion_loss=repulsion.flatten(),
        ensemble_q_std=q_std.flatten(),
        mean_pairwise_q_dist=mean_pairwise_q_dist.flatten(),
    )


def update_value_functions(
    agent_state: REDQState,
    batch: Transition,
    agent_config: REDQConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[REDQState, ValueAuxiliaries]:
    """One critic step of the ensemble on the random-subset target."""
    key, rng = jax.random.split(agent_state.rng)
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    dones = jnp.logical_or(batch.terminated, batch.truncated)
    target_q, next_log_probs = compute_redq_td_target(
        agent_state.actor_state,
        agent_state.critic_state,
        key,
        batch.next_obs,
        dones,
        batch.reward,
        agent_config.gamma,
        alpha,
        agent_config.subset_size,
        agent_config.reward_scale,
        carries,
    )

    def value_loss(params: FrozenDict, target_q: jax.Array) -> Tuple[jax.Array, Any]:
        return value_loss_function(
            params,
            agent_state.critic_state,
            batch.obs,
            batch.action,
            target_q,
            next_log_probs,
            agent_config.repulsion_coef,
            carries,
        )

    critic_state, aux = critic_step(
        agent_state,
        batch,
        target_q,
        value_loss,
        extension_stack,
        key,
        total_timesteps,
        rewards=batch.reward,
        gamma=agent_config.gamma,
        reward_scale=agent_config.reward_scale,
    )
    return agent_state.replace(rng=rng, critic_state=critic_state), aux


def update_policy(
    agent_state: REDQState,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[REDQState, PolicyAuxiliaries, jax.Array]:
    """The actor step on ``alpha log pi(a|s) - mean_i Q_i(s, a)``; also
    returns its samples' log-probs (temperature)."""
    agent_state, (loss, log_probs, q_mean) = core.soft_actor_step(
        agent_state,
        observations,
        raw_observations,
        extension_stack,
        total_timesteps,
        carries,
        q_reduce=jnp.mean,
    )
    aux = PolicyAuxiliaries(
        policy_loss=loss, log_pi=log_probs.mean(), q_mean=q_mean.mean()
    )
    return agent_state, aux, log_probs


def update_agent(
    agent_state: REDQState,
    buffer: BufferType,
    recurrent: bool,
    agent_config: REDQConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[REDQState, AuxiliaryLogs]:
    """One update: ``num_critic_updates`` critic and target steps, each on a
    fresh replay batch, then one actor step and one temperature step on the
    last batch (Chen et al., 2021, Algorithm 1)."""

    def critic_update_step(agent_state: REDQState, _: Any) -> Tuple[REDQState, Any]:
        sample_key, rng = jax.random.split(agent_state.rng)
        batch, carries = sample_replay(
            agent_state,
            buffer,
            sample_key,
            recurrent,
            agent_config.burn_in,
            agent_config.stored_state,
        )
        agent_state, aux_value = update_value_functions(
            agent_state.replace(rng=rng),
            batch,
            agent_config,
            extension_stack,
            total_timesteps,
            carries,
        )
        agent_state = core.update_target_networks(agent_state, agent_config.tau)
        return agent_state, (aux_value, batch, carries)

    # Carry-only scan: only the last critic step's metrics and batch are kept.
    agent_state, (aux_value, batch, carries) = final_aux_scan(
        critic_update_step, agent_state, length=agent_config.num_critic_updates
    )
    agent_state, aux_policy, log_probs = update_policy(
        agent_state,
        batch.obs,
        batch.raw_obs,
        extension_stack,
        total_timesteps,
        carries,
    )
    agent_state, aux_temperature = core.update_temperature(
        agent_state, log_probs, jnp.asarray(agent_config.target_entropy)
    )
    aux = AuxiliaryLogs(temperature=aux_temperature, policy=aux_policy, value=aux_value)
    return agent_state, aux


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: REDQConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    cloning_args: Optional[CloningConfig] = None,
    expert_policy: Optional[Callable] = None,
    pid_actor_config: Optional[PIDActorConfig] = None,
    extensions: Sequence = (),
):
    """REDQ's train function: one step per env, then (from
    ``learning_starts``) one update, per iteration."""
    recurrent = network_args.memory is not None
    if recurrent:
        unsupported_recurrent_options(
            "REDQ",
            expert_policy=expert_policy,
            extensions=(tuple(extensions) or None),
            # REDQ always builds a default CloningConfig; only actual
            # pre-training (pre_train_n_steps > 0) conflicts with memory.
            cloning_pretrain=(
                cloning_args
                if cloning_args is not None and cloning_args.pre_train_n_steps > 0
                else None
            ),
            pid_actor_config=pid_actor_config,
        )
    loop = TrainLoop.create(
        env_args, total_timesteps, num_episode_test, run_ids, logging_config, extensions
    )

    def init(key: jax.Array, pretrain_key: jax.Array) -> REDQState:
        agent_state = init_REDQ(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            alpha_args,
            buffer,
            number_of_critics=agent_config.num_critics,
            stored_state=agent_config.stored_state,
            pid_actor_config=pid_actor_config,
        )
        return pretrain_on_expert(
            agent_state,
            pretrain_key,
            cloning_args,
            expert_policy,
            env_args,
            agent_config,
            actor_optimizer_args,
            critic_optimizer_args,
        )

    def update(agent_state: REDQState, _transition: Transition) -> Any:
        # The step just collected is in the buffer: REDQ samples it from there.
        return update_agent(
            agent_state, buffer, recurrent, agent_config, loop.stack, total_timesteps
        )

    return loop.off_policy(
        init,
        update,
        AuxiliaryLogs,
        agent_config.learning_starts,
        recurrent=recurrent,
        collect_kwargs={
            "buffer": buffer,
            "store_hidden": recurrent and agent_config.stored_state,
        },
        eval_kwargs={"expert_policy": expert_policy},
    )
