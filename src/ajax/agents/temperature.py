"""The entropy temperature alpha of a soft actor-critic (Haarnoja et al.,
2018b), shared by every soft actor-critic: its state, its learning step
towards a target entropy, the actor's step whose log-probabilities it reads
(update_soft_actor), the two steps kept from their starts
(keep_actor_then_update_temperature), and its logs. The temperature's and
the actor's losses are each agent's equations (SAC's, CrossQ's), which the
steps take as arguments."""

from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from ajax.agents import update_steps
from ajax.agents.update_steps import compute_gradients
from ajax.state import BaseAgentState


@struct.dataclass(kw_only=True)
class SoftActorCriticState(BaseAgentState):
    """A soft actor-critic's state: the base one plus the temperature."""

    alpha: TrainState  # log-temperature, params["log_alpha"]


AgentState = TypeVar("AgentState", bound=SoftActorCriticState)


@struct.dataclass
class TemperatureAuxiliaries:
    alpha: jax.Array
    log_alpha: jax.Array
    effective_target_entropy: jax.Array | float


#: loss(log_alpha_params, log_probs, target_entropy) -> (loss, aux): a
#: temperature's loss, the log-probabilities summed over the action
#: dimensions (SAC's compute_temperature_loss).
TemperatureLoss = Callable[..., tuple[jax.Array, TemperatureAuxiliaries]]
#: loss(actor_params, actor_state, critic_state, observations, alpha, key,
#: carries, reduce_critics, map_actions) -> (loss, (log_probs, critic_values,
#: policy)): a soft actor's loss (SAC's compute_soft_policy_loss).
SoftPolicyLoss = Callable[..., tuple[jax.Array, tuple[jax.Array, jax.Array, Any]]]


def create_alpha_train_state(
    learning_rate: float = 3e-4,
    alpha_init: float = 1.0,
) -> TrainState:
    log_alpha = jnp.log(alpha_init)
    params: FrozenDict[str, Any] = FrozenDict({"log_alpha": log_alpha})
    # No gradient clipping on the dual variable. log_alpha's gradient is
    # the scalar (H_target - H_pi); clipping it to a constant norm erases
    # the magnitude (and hence the target) from the update whenever the
    # policy entropy is more than the clip away from the target, turning
    # the step into pure sign-descent at learning_rate per update.
    # Reference SAC uses unclipped Adam here. Nor does the agent's
    # weight_decay reach it: decaying log_alpha towards 0 would pull the
    # temperature towards 1, a bias no entropy target asks for.
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=lambda params: jnp.exp(params["log_alpha"]),
        params=params,
        tx=tx,
    )


def get_alpha(agent_state: SoftActorCriticState) -> jax.Array:
    """The temperature alpha, from its log the state learns."""
    return jnp.exp(agent_state.alpha.params["log_alpha"])


def update_temperature(
    agent_state: AgentState,
    log_probs: jax.Array,
    target_entropy: jax.Array,
    loss: TemperatureLoss,
) -> tuple[AgentState, TemperatureAuxiliaries]:
    """One dual-gradient step of log_alpha towards target_entropy down the
    agent's temperature loss, log_probs per action dimension (summed here);
    returns the state and the loss's logs, no gradient flowing back through
    them."""

    summed_log_probs = log_probs.sum(-1)

    def temperature_loss(params: FrozenDict) -> tuple[jax.Array, Any]:
        return loss(params, summed_log_probs, target_entropy)

    grads, aux = compute_gradients(temperature_loss, agent_state.alpha.params)
    new_alpha_state = agent_state.alpha.apply_gradients(grads=grads)
    return agent_state.replace(alpha=new_alpha_state), jax.lax.stop_gradient(aux)


def update_soft_actor(
    agent_state: AgentState,
    batch: Any,
    carries: Any,
    context: Any,
    loss: SoftPolicyLoss,
    reduce_critics: Callable[..., jax.Array],
) -> tuple[AgentState, jax.Array, jax.Array, jax.Array, dict]:
    """The actor's step down the agent's soft policy loss on the batch (a
    Transition; carries, sequence replay's or None), the critics reduced over
    the ensemble by reduce_critics (jnp.min or jnp.mean) and scoring the
    actions they score (a residual policy's executed ones), plus the
    extensions' actor-loss terms; context is the run's TrainContext.

    Returns the state with the stepped actor, the loss before those terms,
    the sample's log-probabilities, which the temperature's step reads, the
    reduced critic values and the extensions' metrics. SAC's own step
    differs: its temperature reads a fresh sample of the stepped actor."""
    actor_key = update_steps.derive_actor_key(agent_state)
    alpha = get_alpha(agent_state)
    critic_state = agent_state.critic_state
    scored = context.make_scored_actions(batch)

    def actor_loss(params: FrozenDict) -> tuple[jax.Array, Any, jax.Array]:
        value, (log_probs, critic_values, policy) = loss(
            params,
            agent_state.actor_state,
            critic_state,
            batch.obs,
            alpha,
            actor_key,
            carries,
            reduce_critics,
            scored.map_actions,
        )
        return value, (value, log_probs, critic_values), policy.mean()

    actor_state, (value, log_probs, critic_values), _, metrics = (
        update_steps.actor_step(
            agent_state,
            actor_loss,
            context.extensions,
            actor_key,
            context.total_timesteps,
            batch,
            critic_state,
        )
    )
    stepped = agent_state.replace(actor_state=actor_state)
    return stepped, value, log_probs, critic_values, metrics


def update_temperature_when_due(
    agent: Any,
    agent_state: AgentState,
    log_probs: jax.Array,
    extensions: Any,
    loss: TemperatureLoss,
    fixed_alpha: bool = False,
) -> tuple[AgentState, TemperatureAuxiliaries]:
    """update_temperature towards the agent's target entropy at the
    timestep, kept from its alpha_update_start (temperature_update_due);
    agent is the soft actor-critic (EntropyTemperature). With fixed_alpha
    the loss is computed for its logs and the step dropped, so alpha stays at
    alpha_init; the flag is static, so that run compiles no gate."""
    timestep = agent_state.collector_state.timestep
    tempered, logs = update_temperature(
        agent_state, log_probs, agent.target_entropy_at(timestep), loss
    )
    if fixed_alpha:
        return agent_state, logs
    due = agent.temperature_update_due(timestep, extensions)
    return update_steps.run_if(due, lambda _: tempered, agent_state), logs


def keep_actor_then_update_temperature(
    agent: Any,
    agent_state: AgentState,
    stepped: AgentState,
    actor_logs: Any,
    log_probs: jax.Array,
    extensions: Any,
    loss: TemperatureLoss,
    fixed_alpha: bool = False,
) -> tuple[AgentState, Any, TemperatureAuxiliaries]:
    """A soft actor-critic's actor step, then its temperature's.

    stepped and actor_logs are the actor's step from agent_state. It is
    computed from the first update, since its sample's log_probs feed the
    temperature's step, and kept from the agent's policy_update_start
    (update_steps.run_actor_if); the temperature's step follows
    (update_temperature_when_due). Returns the state, the actor's logs and
    the temperature's."""
    timestep = agent_state.collector_state.timestep
    agent_state, kept_actor_logs = update_steps.run_actor_if(
        agent.policy_update_due(timestep),
        lambda _: (stepped, actor_logs),
        agent_state,
        keep=agent.keeps_actor_metrics(),
    )
    agent_state, temperature_logs = update_temperature_when_due(
        agent, agent_state, log_probs, extensions, loss, fixed_alpha
    )
    return agent_state, kept_actor_logs, temperature_logs
