"""DAgger (Ross, Gordon and Bagnell, AISTATS 2011, Alg. 3.1): imitation by
dataset aggregation::

    D = {}
    for i = 1, 2, ...:
        roll out pi_i = beta_i pi* + (1 - beta_i) pi_hat_i
        D <- D + {(s, pi*(s)) for every state s pi_i visits}
        pi_hat_{i+1} = the policy fitted to D (supervised)

On the shared off-policy loop (:meth:`TrainLoop.off_policy`), at each step
every env executes the expert's action with probability ``beta_i``, else the
actor's deterministic one (:func:`make_action_pipeline`; the per-env,
per-step mixing of HumanCompatibleAI/imitation@e5ef188
``algorithms/dagger.py``, ``InteractiveTrajectoryCollector``). The replay,
which keeps every visited state with the expert's action there, is D. The
fit is online, one gradient step on a replay batch per tick once the replay
holds a batch, rather than a fit to convergence after each iteration: the
paper's guarantee holds for any no-regret online learner (Sec. 4). The loss
is the squared error to the expert's action for continuous actions and its
cross-entropy for discrete ones. Neither a reward nor a critic is used.

Deviations: the paper returns the best ``pi_hat_i`` on validation, Ajax the
last one, with every evaluation logged; the replay drops its oldest rows
past ``buffer_size``, where D keeps them all.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents.DAgger.state import DAggerConfig, DAggerState
from ajax.agents.loop import TrainLoop, actor_step
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    ActionPipelineResult,
    expert_step,
    get_pi,
    init_collector_state,
)
from ajax.environments.utils import flatten_action_column
from ajax.extensions.base import ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.deterministic import get_initialized_td3_actor_critic
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import BufferType


@struct.dataclass
class PolicyAuxiliaries:
    imitation_loss: jax.Array
    grad_norm: jax.Array | float = jnp.nan  # before clipping


@struct.dataclass
class DAggerAuxiliaries:
    beta: jax.Array  # the iteration's beta_i
    expert_frac: float | jax.Array  # the share of envs the expert drove last step


@struct.dataclass
class AuxiliaryLogs:
    policy: PolicyAuxiliaries
    dagger: DAggerAuxiliaries
    extensions: dict  # the extensions' training metrics


def beta(timestep: int | jax.Array, config: DAggerConfig) -> jax.Array:
    """``beta_i = beta_decay ** (i - 1)`` at iteration ``i = timestep //
    iteration_steps + 1``; ``0 ** 0 = 1``."""
    i = jnp.asarray(timestep // config.iteration_steps, jnp.float32)
    return jnp.power(jnp.float32(config.beta_decay), i)


def learner_action(pi: Any, continuous: bool) -> jax.Array:
    """The actor's deterministic action: the squashed mean, or the most
    likely action as an int ``(n, 1)`` column, as a discrete replay keeps it."""
    return pi.mean() if continuous else pi.mode()[..., None].astype(jnp.int32)


def make_action_pipeline(
    expert_policy: Any, config: DAggerConfig, continuous: bool, n_envs: int
) -> Callable:
    """Collection's pipeline: per env, the expert's action with probability
    ``beta_i``, else the actor's; the expert's action at every state is
    stored (``a_expert``), from its state along the trajectory."""

    def pipeline(agent_state, acting, raw_obs, rng, uniform, mix_key, action_key):
        del uniform, action_key
        actor = agent_state.actor_state
        own = learner_action(get_pi(actor, actor.params, acting.obs)[0], continuous)
        a_expert, expert_state = expert_step(expert_policy, acting, raw_obs, n_envs)
        a_expert = a_expert.reshape(own.shape).astype(own.dtype)
        draw = jax.random.uniform(mix_key, (n_envs, 1))
        use_expert = draw < beta(acting.timestep, config)
        action = jnp.where(use_expert, a_expert, own)
        zeros = jnp.zeros((n_envs, 1))
        return ActionPipelineResult(
            env_action=flatten_action_column(action, continuous),
            policy_action=own,
            log_probs=zeros,
            is_expert_flag=use_expert.astype(jnp.float32),
            in_value_box=zeros,
            entry_bonus=zeros,
            rng=rng,
            new_expert_state=expert_state,
            buffer_action=action,
            a_expert=a_expert,
        )

    return pipeline


def init_DAgger(
    key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    expert_policy: Any,
) -> DAggerState:
    """A deterministic actor (TD3's, tanh-squashed) for continuous actions,
    a categorical one for discrete; the critic built beside it is unused."""
    rng, init_key, collector_key, eval_rng = jax.random.split(key, 4)
    if env_args.continuous:
        actor_state, critic_state = get_initialized_td3_actor_critic(
            init_key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            num_critics=1,
        )
    else:
        actor_state, critic_state = get_initialized_actor_critic(
            init_key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            continuous=False,
        )
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=env_args.mode,
        buffer=buffer,
        include_raw_obs=True,
        include_is_expert=True,
        expert_policy=expert_policy,
    )
    return DAggerState(
        rng=rng,
        eval_rng=eval_rng,
        actor_state=actor_state,
        critic_state=critic_state,
        collector_state=collector_state,
    )


def update_agent(
    agent_state: DAggerState,
    buffer: BufferType,
    config: DAggerConfig,
    continuous: bool,
    extension_stack: ExtensionStack,
    total_timesteps: int,
) -> tuple[DAggerState, AuxiliaryLogs]:
    """One supervised step towards the expert's actions on a replay batch."""
    agent_state, sample_key, actor_key = agent_state.next_keys(2)
    batch = get_batch_from_buffer(
        buffer, agent_state.collector_state.buffer_state, sample_key
    )

    def imitation_loss(params: Any) -> tuple[jax.Array, jax.Array, Any]:
        pi, _ = get_pi(agent_state.actor_state, params, batch.obs)
        if continuous:
            pi_mean = pi.mean()
            loss = jnp.mean(jnp.square(pi_mean - batch.a_expert))
        else:
            pi_mean = None
            loss = -jnp.mean(pi.log_prob(batch.a_expert[..., 0]))
        return loss, loss, pi_mean

    actor_state, loss, grad_norm, metrics = actor_step(
        agent_state,
        imitation_loss,
        extension_stack,
        actor_key,
        total_timesteps,
        observations=batch.obs,
        raw_observations=batch.raw_obs,
        a_expert=batch.a_expert,
    )
    collector = agent_state.collector_state
    agent_state = agent_state.replace(
        actor_state=actor_state, n_updates=agent_state.n_updates + 1
    )
    return agent_state, AuxiliaryLogs(
        policy=PolicyAuxiliaries(imitation_loss=loss, grad_norm=grad_norm),
        dagger=DAggerAuxiliaries(
            beta=beta(collector.timestep, config),
            expert_frac=collector.last_expert_frac,
        ),
        extensions=metrics,
    )


def make_train(
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_config: DAggerConfig,
    total_timesteps: int,
    num_episode_test: int,
    batch_size: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
    start_timestep: int = 0,
    end_timestep: Optional[int] = None,
    expert_policy: Any = None,
    extensions: Sequence = (),
) -> Callable:
    """DAgger's train function: one step per env, then, once the replay
    holds a batch, one update, per iteration."""
    agent_context = {
        "env_args": env_args,
        "actor_optimizer_args": actor_optimizer_args,
        "expert_policy": expert_policy,
        "buffer": buffer,
    }
    extensions = ExtensionStack(extensions).bind_to_agent(**agent_context).extensions
    continuous = env_args.continuous
    # The pipeline tracks the expert and stores its action itself. The loop
    # gets no expert_policy, so the expert's next action (next_a_expert),
    # which only the critic modules DAgger refuses read, stays zeros.
    loop = TrainLoop.create(
        env_args,
        total_timesteps,
        num_episode_test,
        run_ids,
        logging_config,
        extensions,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
    )

    def init(key: jax.Array) -> DAggerState:
        return init_DAgger(
            key,
            env_args,
            actor_optimizer_args,
            critic_optimizer_args,
            network_args,
            buffer,
            expert_policy,
        )

    def update(agent_state: DAggerState, _transition: Any) -> Any:
        return update_agent(
            agent_state,
            buffer,
            agent_config,
            continuous,
            loop.extensions,
            total_timesteps,
        )

    pipeline = make_action_pipeline(
        expert_policy, agent_config, continuous, env_args.n_envs
    )
    return loop.off_policy(
        init,
        update,
        batch_size,
        collect_kwargs={"buffer": buffer, "action_pipeline": pipeline},
        eval_kwargs={"expert_policy": expert_policy},
    )
