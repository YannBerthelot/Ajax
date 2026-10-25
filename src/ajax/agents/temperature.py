"""The entropy temperature alpha of a soft actor-critic (Haarnoja et al.,
2018b): its state, its learning step towards a target entropy and its logs,
shared by every soft actor-critic (SAC, ASAC, AVG, CrossQ, REDQ, TQC). The
temperature's loss is each agent's equation (SAC's, CrossQ's), which the
step takes as an argument."""

from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

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
