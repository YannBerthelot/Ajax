"""The automatic reset cost of Hisaki & Ono (ICML 2024, Section 3.3): a way to
learn a continuing criterion on a task whose episodes terminate. A
termination is treated as a step into the reset state, its target
bootstrapping there, and is charged a cost c. c is the Lagrange multiplier of
the constraint "terminate at most at the rate eps" (their Eq. 16): a reset
critic learns the termination rate of the policy by RVI Q-learning on the
indicator of a termination (Eq. 22-23), and c rises while that rate is above
eps, falls while it is below, and never goes below 0 (Eq. 24).

As the authors' code (yhisaki/average-reward-drl b5b0d31, rvi_sac.py:64-74,
:91-93, :158-214; the same in their SAC and ARO-DDPG with reset): a 64-64
ReLU reset critic whose every layer starts orthogonal with gain sqrt(1/3)
and a zero bias, its own Adam and Polyak target, a mean squared error
without the 1/2, c from 0 stepped by Adam, then clamped at 0 (the Adam's
moments kept). RVI-SAC calls these functions in the authors' order; the
ResetCost extension (ajax.extensions.reset_cost) gives them to other agents.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from ajax.networks.utils import dense, parse_architecture

#: The authors' initialisation of every layer (modules.py:40-43, ortho_init).
ORTHOGONAL_THIRD = "orthogonal(np.sqrt(1 / 3))"
#: torch's Adam eps, which the authors' Adams keep.
TORCH_ADAM_EPS = 1e-8


@dataclass(frozen=True)
class ResetCostSettings:
    """The reset cost's settings, the authors' by default (their Table 1,
    rvi_sac.yaml): the termination rate eps it holds the policy to, the rate
    kappa and the gain at which xi_reset follows the reset critic's batch
    mean, the reset critic's architecture and the Adams' learning rate."""

    target_rate: float = 1e-3
    rate_tau: float = 0.005
    gain: float = 1.0
    architecture: tuple[str, ...] = ("64", "relu", "64", "relu")
    learning_rate: float = 3e-4


class ResetCritic(nn.Module):
    """Q_reset(s, a): the differential value of the termination indicator."""

    architecture: tuple[str, ...]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = nn.Sequential(parse_architecture(self.architecture, ORTHOGONAL_THIRD))(x)
        return dense(1, ORTHOGONAL_THIRD, None, nn.initializers.zeros)(h)


@struct.dataclass
class ResetCostState:
    """The reset critic and its target, the termination rate xi_reset its
    reference function follows, and the reset cost c (params["cost"])."""

    critic: TrainState
    target_params: Any
    rate: jax.Array
    cost: TrainState


def reset_critic_input(
    observations: jax.Array, actions: jax.Array, action_count: Optional[int] = None
) -> jax.Array:
    """The reset critic's input: the observation and the action, one-hot for
    a discrete one (action_count actions)."""
    if action_count is not None:
        actions = jax.nn.one_hot(actions.reshape(observations.shape[:-1]), action_count)
    return jnp.concatenate([observations, actions.astype(observations.dtype)], -1)


def create_reset_cost_state(
    key: jax.Array,
    input_width: int,
    architecture: tuple[str, ...] = ("64", "relu", "64", "relu"),
    learning_rate: float = 3e-4,
) -> ResetCostState:
    """The reset critic (its target a copy), the rate and the cost at 0."""
    network = ResetCritic(architecture)
    params = network.init(key, jnp.zeros((1, input_width)))
    critic = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.adam(learning_rate, eps=TORCH_ADAM_EPS),
    )
    cost = TrainState.create(
        apply_fn=None,
        params=FrozenDict({"cost": jnp.zeros(())}),
        tx=optax.adam(learning_rate, eps=TORCH_ADAM_EPS),
    )
    return ResetCostState(critic, params, jnp.zeros(()), cost)


def reset_charge(state: ResetCostState, terminated: jax.Array) -> jax.Array:
    """c on a terminated row, 0 elsewhere: what its reward is charged."""
    return state.cost.params["cost"] * terminated


def step_reset_critic(
    state: ResetCostState,
    inputs: jax.Array,
    next_inputs: jax.Array,
    terminated: jax.Array,
    rate_tau: float,
    gain: float = 1.0,
) -> tuple[ResetCostState, dict]:
    """One step of the reset critic on 1(term) - xi_reset + Q'_reset(s', a')
    (Eq. 22), a' the policy's next action, then xi_reset <- (1 - rate_tau)
    xi_reset + rate_tau gain mean Q'_reset(s', a') (Eq. 23); terminated is
    shaped like the critic's output."""
    apply = state.critic.apply_fn
    next_values = apply(state.target_params, next_inputs)
    target = jax.lax.stop_gradient(terminated - state.rate + next_values)

    def loss(params: FrozenDict) -> jax.Array:
        return jnp.mean((apply(params, inputs) - target) ** 2)

    value, grads = jax.value_and_grad(loss)(state.critic.params)
    rate = (1 - rate_tau) * state.rate + rate_tau * gain * jnp.mean(next_values)
    critic = state.critic.apply_gradients(grads=grads)
    return dataclasses.replace(state, critic=critic, rate=rate), {
        "reset_critic_loss": value
    }


def step_reset_cost(state: ResetCostState, target_rate: float) -> ResetCostState:
    """One Adam step of c down -c (xi_reset - eps) (Eq. 24), then c >= 0."""

    def loss(params: FrozenDict) -> jax.Array:
        return -params["cost"] * (state.rate - target_rate)

    grads = jax.grad(loss)(state.cost.params)
    cost = state.cost.apply_gradients(grads=grads)
    clamped: FrozenDict = FrozenDict({"cost": jnp.maximum(cost.params["cost"], 0.0)})
    return dataclasses.replace(state, cost=cost.replace(params=clamped))


def update_reset_target(state: ResetCostState, tau: float) -> ResetCostState:
    """Polyak-average the reset critic's target towards it."""
    target = optax.incremental_update(state.critic.params, state.target_params, tau)
    return dataclasses.replace(state, target_params=target)
