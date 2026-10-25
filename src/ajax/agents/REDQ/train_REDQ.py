"""REDQ (Chen et al., 2021): Randomized Ensembled Double Q-learning.

SAC with an ensemble of ``num_critics`` critics updated
``num_critic_updates`` times per environment step, each on a fresh replay
minibatch with a target the min over a random subset of ``subset_size``
target critics, and an actor that
maximises the ensemble's mean Q. The SVGD variant's kernel repulsion
between the critics is the KernelRepulsion extension.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict

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
from ajax.agents.update_steps import update_target_networks
from ajax.buffers.dataset import TransitionDataset
from ajax.controller import critic_action_units, executed_fn
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack, holds_expert, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import action_value
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
    # Scale-free ensemble divergence diagnostics (stop_gradient'd): whether
    # the ensemble diversified in function space.
    ensemble_q_std: jax.Array
    mean_pairwise_q_dist: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics


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
    store_expert_data: bool = False,
    expert_policy: Optional[Callable] = None,
    store_raw_next_obs: bool = False,
    store_return_to_go: bool = False,
    critic_action_units: Optional[tuple] = None,
) -> REDQState:
    rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
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
        collector_extras={
            "include_raw_obs": store_expert_data,
            "include_raw_next_obs": store_raw_next_obs,
            "include_return_to_go": store_return_to_go,
            "include_is_expert": store_expert_data,
            "expert_policy": expert_policy,
        },
    )
    return REDQState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=critic_state.replace(action_units=critic_action_units),
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
    executed: Optional[Callable] = None,
    next_a_expert: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """REDQ bellman target: min over a random subset of target critics, at
    the next actions executed (``executed``, a residual policy's).

    Returns the stop_gradient'd target and the next actions' log-probs.
    """
    rewards = rewards * reward_scale
    sample_key, idx_sample_key = jax.random.split(rng)
    next_actions, log_probs = core.sample_next_actions(
        actor_state, next_observations, sample_key, carries, executed, next_a_expert
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
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, ValueAuxiliaries]:
    """Every critic regresses on the one target: the sum over the critics
    of each one's mean squared error, as REDQ's official
    ``mse_criterion(q_prediction_cat, y_q) * self.num_Q``
    (watchernyu/REDQ@7b5d1bf, ``redq/algos/redq_sac.py:207``), whose
    per-critic Adam optimizers (``:49-51``) each see their critic's own
    error. A recurrent critic's error is its mean over time and batch."""
    q_preds = q_values(critic_states, critic_params, observations, actions, carries)
    loss = jnp.sum(
        jnp.mean((q_preds - target_q) ** 2, axis=tuple(range(1, q_preds.ndim)))
    )
    q_std, mean_pairwise_q_dist = q_ensemble_divergence(q_preds)
    return loss, ValueAuxiliaries(
        critic_loss=loss,
        target_q=target_q.mean().flatten(),
        log_probs=next_log_probs.mean().flatten(),
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
    executed: Optional[Callable] = None,
) -> Tuple[REDQState, ValueAuxiliaries, dict]:
    """One critic step of the ensemble on the random-subset target."""
    agent_state, key = agent_state.next_keys()
    alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
    # A truncated row bootstraps on its final observation (bootstrap_obs).
    dones = batch.terminated
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
        executed,
        batch.next_a_expert,
    )

    def value_loss(params: FrozenDict, target_q: jax.Array) -> Tuple[jax.Array, Any]:
        return value_loss_function(
            params,
            agent_state.critic_state,
            batch.obs,
            batch.action,
            target_q,
            next_log_probs,
            carries,
        )

    critic_state, aux, _, metrics = critic_step(
        agent_state,
        batch,
        target_q,
        value_loss,
        extension_stack,
        key,
        total_timesteps,
        rewards=batch.reward,
        dones=dones,
        gamma=agent_config.gamma,
        reward_scale=agent_config.reward_scale,
    )
    return agent_state.replace(critic_state=critic_state), aux, metrics


def update_policy(
    agent_state: REDQState,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
    a_expert: Optional[jax.Array] = None,
    actions: Optional[jax.Array] = None,
    executed: Optional[Callable] = None,
) -> Tuple[REDQState, PolicyAuxiliaries, jax.Array, dict]:
    """The actor step on ``alpha log pi(a|s) - mean_i Q_i(s, a)``; also
    returns its samples' log-probs (temperature) and the extensions'
    metrics."""
    agent_state, (loss, log_probs, q_mean), metrics = core.soft_actor_step(
        agent_state,
        observations,
        raw_observations,
        extension_stack,
        total_timesteps,
        carries,
        q_reduce=jnp.mean,
        a_expert=a_expert,
        actions=actions,
        executed=executed,
    )
    aux = PolicyAuxiliaries(
        policy_loss=loss, log_pi=log_probs.mean(), q_mean=q_mean.mean()
    )
    return agent_state, aux, log_probs, metrics


#: The temperature REDQ's code reads until its first temperature step: ``__init__`` sets
#: ``self.alpha = exp(log_alpha)``, then ``self.alpha = alpha``, the constructor's 0.2
#: (watchernyu/REDQ@7b5d1bf, ``redq/algos/redq_sac.py:61``, ``:80``), likely a slip, kept as the
#: original runs.
FIRST_ALPHA = 0.2


def read_temperature(agent_state: REDQState) -> REDQState:
    """``agent_state`` with the temperature REDQ's code reads: :data:`FIRST_ALPHA` before the first
    temperature step, ``exp(log_alpha)`` from it on."""
    log_alpha = agent_state.alpha.params["log_alpha"]
    read = jnp.where(agent_state.alpha.step == 0, jnp.log(FIRST_ALPHA), log_alpha)
    alpha = agent_state.alpha.replace(params=FrozenDict({"log_alpha": read}))
    return agent_state.replace(alpha=alpha)


def update_agent(
    agent_state: REDQState,
    buffer: BufferType,
    recurrent: bool,
    agent_config: REDQConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    executed: Optional[Callable] = None,
) -> Tuple[REDQState, AuxiliaryLogs]:
    """One update: ``num_critic_updates`` critic and target steps, each on a
    fresh replay batch, then one actor step and one temperature step on the
    last batch (Chen et al., 2021, Algorithm 1).

    The actor step reads the critics from before the last critic step, as
    REDQ's official code, which computes the actor loss
    (watchernyu/REDQ@7b5d1bf, ``redq/algos/redq_sac.py:220``) before the
    critics' optimizers step (``:242``); Algorithm 1 reads them after. The
    critic and actor steps read :func:`read_temperature`'s temperature; the
    temperature step updates the learned one."""
    learned = agent_state.alpha

    def critic_update_step(agent_state: REDQState, _: Any) -> Tuple[REDQState, Any]:
        agent_state, sample_key = agent_state.next_keys()
        critics_before = agent_state.critic_state.params
        batch, carries = sample_replay(
            agent_state,
            buffer,
            sample_key,
            recurrent,
            agent_config.burn_in,
            agent_config.stored_state,
        )
        agent_state, aux_value, value_metrics = update_value_functions(
            agent_state,
            batch,
            agent_config,
            extension_stack,
            total_timesteps,
            carries,
            executed,
        )
        agent_state = update_target_networks(agent_state, agent_config.tau)
        return agent_state, (aux_value, value_metrics, batch, carries, critics_before)

    # Carry-only scan: only the last critic step's metrics, batch and
    # critics before it are kept.
    agent_state, last = final_aux_scan(
        critic_update_step,
        read_temperature(agent_state),
        length=agent_config.num_critic_updates,
    )
    aux_value, value_metrics, batch, carries, critics_before = last
    critic_state = agent_state.critic_state
    agent_state, aux_policy, log_probs, policy_metrics = update_policy(
        agent_state.replace(critic_state=critic_state.replace(params=critics_before)),
        batch.obs,
        batch.raw_obs,
        extension_stack,
        total_timesteps,
        carries,
        a_expert=batch.a_expert,
        actions=batch.action,
        executed=executed,
    )
    agent_state = agent_state.replace(critic_state=critic_state, alpha=learned)
    agent_state, aux_temperature = core.update_temperature(
        agent_state, log_probs, jnp.asarray(agent_config.target_entropy)
    )
    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=aux_value,
        extensions=merge_metrics(value_metrics, policy_metrics),
    )
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
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    expert_policy: Optional[Callable] = None,
    offline_data: Optional[TransitionDataset] = None,
    extensions: Sequence = (),
):
    """REDQ's train function: one step per env, then (from
    ``learning_starts``) one update, per iteration."""
    agent_context = {
        "env_args": env_args,
        "actor_optimizer_args": actor_optimizer_args,
        "expert_policy": expert_policy,
        "buffer": buffer,
    }
    extensions = ExtensionStack(extensions).bind_to_agent(**agent_context).extensions
    recurrent = network_args.memory is not None
    if recurrent:
        unsupported_recurrent_options(
            "REDQ",
            expert_policy=expert_policy,
            extensions=(tuple(extensions) or None),
        )
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
        expert_policy=expert_policy,
        offline_data=offline_data,
    )

    # REDQ's official critics read act_limit * tanh, the env's units
    # (watchernyu/REDQ@7b5d1bf, core.py:238): so do Ajax's.
    units = critic_action_units(env_args)

    def init(key: jax.Array) -> REDQState:
        return init_REDQ(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            alpha_args,
            buffer,
            number_of_critics=agent_config.num_critics,
            stored_state=agent_config.stored_state,
            store_expert_data=holds_expert(expert_policy, loop.extensions),
            expert_policy=expert_policy,
            store_raw_next_obs="raw_next_obs" in loop.extensions.buffer_fields(),
            store_return_to_go="return_to_go" in loop.extensions.buffer_fields(),
            critic_action_units=units,
        )

    def update(agent_state: REDQState, _transition: Transition) -> Any:
        # The step just collected is in the buffer: REDQ samples it from there.
        return update_agent(
            agent_state,
            buffer,
            recurrent,
            agent_config,
            loop.extensions,
            total_timesteps,
            executed_fn(env_args),
        )

    return loop.off_policy(
        init,
        update,
        agent_config.learning_starts,
        recurrent=recurrent,
        collect_kwargs={
            "buffer": buffer,
            "store_hidden": recurrent and agent_config.stored_state,
        },
        eval_kwargs={
            "expert_policy": expert_policy,
            "value_estimate": ValueEstimate(
                action_value, agent_config.gamma, agent_config.reward_scale
            ),
        },
    )
