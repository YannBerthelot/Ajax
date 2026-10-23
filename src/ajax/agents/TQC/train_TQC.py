"""TQC (Kuznetsov, Shvechikov, Grishin and Vetrov, ICML 2020): Truncated
Quantile Critics.

SAC with ``num_critics`` distributional critics, each predicting
``n_quantiles`` quantiles of the return. The target pools every target
critic's quantiles at ``(s', a')``, drops the ``top_quantiles_to_drop_per_net
* num_critics`` largest, and every critic regresses its quantiles on the
rest with the quantile Huber loss; the actor maximises the mean of all the
critics' quantiles. The reference is the authors' code, bayesgroup/tqc_pytorch
(Software Heritage swh:1:rev:65114eba5d868e9f0355e33b96acab7530191f16):
``tqc/trainer.py`` and ``tqc/functions.py``.
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
from ajax.agents.SAC import core
from ajax.agents.SAC.core import TemperatureAuxiliaries
from ajax.agents.TQC.state import TQCConfig, TQCState
from ajax.buffers.dataset import TransitionDataset
from ajax.evaluate import ValueEstimate
from ajax.extensions.base import ExtensionStack, holds_expert, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import action_value
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
    target_q: jax.Array  # the kept target quantiles' mean
    log_probs: jax.Array
    # The kept target quantiles' spread, max - min per row: the return
    # distribution's width the critics learn.
    target_spread: jax.Array


@struct.dataclass
class AuxiliaryLogs:
    temperature: TemperatureAuxiliaries
    policy: PolicyAuxiliaries
    value: ValueAuxiliaries
    extensions: dict  # the extensions' training metrics


def init_TQC(
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
) -> TQCState:
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
    return TQCState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=core.create_alpha_train_state(**to_state_dict(alpha_args)),
        collector_state=collector_state,
    )


def truncated_quantiles(quantiles: jax.Array, n_drop: int) -> jax.Array:
    """Every critic's quantiles (the ensemble axis first, the quantiles
    last) pooled per row and sorted, without the ``n_drop`` largest
    (trainer.py:48-50)."""
    pooled = jnp.moveaxis(quantiles, 0, -2)
    pooled = pooled.reshape(pooled.shape[:-2] + (-1,))
    return jnp.sort(pooled, axis=-1)[..., : pooled.shape[-1] - n_drop]


def quantile_huber_loss(quantiles: jax.Array, targets: jax.Array) -> jax.Array:
    """Every critic's quantiles ``(num_critics, ..., M)`` against every
    target atom ``(..., K)``: the quantile Huber loss, threshold 1, at the
    quantile fractions ``(2m + 1) / 2M``, averaged over critics, rows,
    quantiles and atoms (functions.py:22-32)."""
    delta = targets[None, ..., None, :] - quantiles[..., :, None]
    abs_delta = jnp.abs(delta)
    huber = jnp.where(abs_delta > 1.0, abs_delta - 0.5, 0.5 * delta**2)
    n = quantiles.shape[-1]
    tau = jnp.arange(n, dtype=jnp.float32) / n + 1 / 2 / n
    return jnp.mean(jnp.abs(tau[:, None] - (delta < 0)) * huber)


def compute_tqc_td_target(
    actor_state: LoadedTrainState,
    critic_state: LoadedTrainState,
    rng: jax.Array,
    next_observations: jax.Array,
    dones: jax.Array,
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    n_drop: int,
    reward_scale: float,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[jax.Array, jax.Array]:
    """``r + gamma (1 - d)(z - alpha log pi(a'|s'))`` for every kept target
    quantile ``z`` at ``a' ~ pi(.|s')`` (trainer.py:45-53). Returns the
    held target atoms and the next actions' log-probs."""
    next_actions, log_probs = core.sample_next_actions(
        actor_state, next_observations, rng, carries
    )
    next_quantiles = q_values(
        critic_state,
        critic_state.target_params,
        next_observations,
        next_actions,
        carries,
        bootstrap=True,
        quantiles=True,
    )
    target = core.soft_bellman_target(
        truncated_quantiles(next_quantiles, n_drop),
        log_probs,
        rewards,
        dones,
        gamma,
        alpha,
        reward_scale,
    )
    return jax.lax.stop_gradient(target), log_probs


def update_value_functions(
    agent_state: TQCState,
    batch: Transition,
    agent_config: TQCConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
) -> Tuple[TQCState, ValueAuxiliaries, dict]:
    """The quantile critics' step on the truncated target."""
    agent_state, key = agent_state.next_keys()
    # A truncated row bootstraps on its final observation (bootstrap_obs).
    dones = batch.terminated
    target_q, next_log_probs = compute_tqc_td_target(
        agent_state.actor_state,
        agent_state.critic_state,
        key,
        batch.next_obs,
        dones,
        batch.reward,
        agent_config.gamma,
        jnp.exp(agent_state.alpha.params["log_alpha"]),
        agent_config.top_quantiles_to_drop_per_net * agent_config.num_critics,
        agent_config.reward_scale,
        carries,
    )

    def value_loss(params: FrozenDict, target_q: jax.Array) -> Tuple[jax.Array, Any]:
        quantiles = q_values(
            agent_state.critic_state,
            params,
            batch.obs,
            batch.action,
            carries,
            quantiles=True,
        )
        loss = quantile_huber_loss(quantiles, target_q)
        spread = jnp.max(target_q, axis=-1) - jnp.min(target_q, axis=-1)
        return loss, ValueAuxiliaries(
            critic_loss=loss,
            target_q=target_q.mean().flatten(),
            log_probs=next_log_probs.mean().flatten(),
            target_spread=spread.mean().flatten(),
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
    agent_state: TQCState,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
    a_expert: Optional[jax.Array] = None,
    actions: Optional[jax.Array] = None,
) -> Tuple[TQCState, PolicyAuxiliaries, jax.Array, dict]:
    """The actor step on ``alpha log pi(a|s) - mean Z(s, a)``, the mean over
    every critic's quantiles (trainer.py:61); also returns its samples'
    log-probs (temperature) and the extensions' metrics."""
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
    )
    aux = PolicyAuxiliaries(
        policy_loss=loss, log_pi=log_probs.mean(), q_mean=q_mean.mean()
    )
    return agent_state, aux, log_probs, metrics


def update_agent(
    agent_state: TQCState,
    buffer: BufferType,
    recurrent: bool,
    agent_config: TQCConfig,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> Tuple[TQCState, AuxiliaryLogs]:
    """One update on a replay batch: the critics' step and the target
    update, then the actor's and the temperature's steps.

    The actor step reads the critics from before their step, as the
    authors' code builds its actor loss (trainer.py:59-61) before the
    critics' optimizer steps (:64-66). It differentiates that loss after
    them; the torch it pins (1.3.0) steps parameters in place through
    ``.data``, which autograd does not track, so that backward pass would
    read the stepped critics' weights (from memory, unverified). Ajax
    differentiates the loss it computed, at the critics before their step:
    the order is logged as an upstream bug, which Ajax does not follow
    (owner, 2026-10-23), and the conformance check runs their update with
    the actor's backward before the critics' step.
    """
    agent_state, sample_key = agent_state.next_keys()
    batch, carries = sample_replay(
        agent_state,
        buffer,
        sample_key,
        recurrent,
        agent_config.burn_in,
        agent_config.stored_state,
    )
    critics_before = agent_state.critic_state.params
    agent_state, aux_value, value_metrics = update_value_functions(
        agent_state, batch, agent_config, extension_stack, total_timesteps, carries
    )
    agent_state = core.update_target_networks(agent_state, agent_config.tau)
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
    )
    agent_state = agent_state.replace(critic_state=critic_state)
    agent_state, aux_temperature = core.update_temperature(
        agent_state, log_probs, jnp.asarray(agent_config.target_entropy)
    )
    return agent_state, AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=aux_value,
        extensions=merge_metrics(value_metrics, policy_metrics),
    )


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: TQCConfig,
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
    """TQC's train function: one step per env, then (from
    ``learning_starts``) one update, per iteration; the policy acts from the
    first step, as the authors' code (main.py:54-66)."""
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
            "TQC",
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

    def init(key: jax.Array) -> TQCState:
        return init_TQC(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            alpha_args,
            buffer,
            number_of_critics=agent_config.num_critics,
            stored_state=agent_config.stored_state,
            store_expert_data=holds_expert(expert_policy, loop.stack),
            expert_policy=expert_policy,
            store_raw_next_obs="raw_next_obs" in loop.stack.buffer_fields(),
            store_return_to_go="return_to_go" in loop.stack.buffer_fields(),
        )

    def update(agent_state: TQCState, _transition: Transition) -> Any:
        # The step just collected is in the buffer: TQC samples it from there.
        return update_agent(
            agent_state, buffer, recurrent, agent_config, loop.stack, total_timesteps
        )

    return loop.off_policy(
        init,
        update,
        agent_config.learning_starts,
        uniform_steps=0,
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
