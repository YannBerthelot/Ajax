"""SAC's former core module. SAC's equations moved to
:mod:`ajax.agents.SAC.SAC`, renamed by the verb rule (compute_ for a loss or
a target), and the temperature to :mod:`ajax.agents.temperature`; their old
names resolve from here with a FutureWarning until 0.5.0. The functions
with no new home stay, as 0.4.0 released them, until then:
init_soft_actor_critic, soft_actor_step and update_temperature (SAC's loss
by default)."""

from typing import Any, Callable, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from ajax.agents import temperature
from ajax.agents.recurrent import RecurrentCarries, stored_actor_carry_dim
from ajax.agents.SAC.SAC import compute_soft_policy_loss, compute_temperature_loss
from ajax.agents.update_steps import derive_extensions_key, gradient_step
from ajax.controller import make_scored_action
from ajax.deprecation import moved_names
from ajax.environments.interaction import init_collector_state
from ajax.extensions.base import ExtensionStack
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import BufferType

S = TypeVar("S", bound=temperature.SoftActorCriticState)


def init_soft_actor_critic(
    init_key: jax.Array,
    collector_key: jax.Array,
    env_args: EnvironmentConfig,
    actor_optimizer_args: OptimizerConfig,
    critic_optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: Optional[BufferType] = None,
    *,
    num_critics: int,
    window_size: int = 10,
    stored_state: bool = False,
    collector_extras: Optional[dict] = None,
) -> Tuple[Any, Any, Any]:
    """A soft actor-critic's squashed-Gaussian actor, num_critics Q critics
    and collector (writing to buffer when given); the collector_extras reach
    the collector's builder."""
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
    )
    collector_state = init_collector_state(
        collector_key,
        env_args=env_args,
        mode=env_args.mode,
        buffer=buffer,
        window_size=window_size,
        actor_carry_dim=stored_actor_carry_dim(network_args.memory, stored_state),
        **(collector_extras or {}),
    )
    return actor_state, critic_state, collector_state


def soft_actor_step(
    agent_state: S,
    observations: jax.Array,
    raw_observations: Optional[jax.Array],
    extension_stack: ExtensionStack,
    total_timesteps: int,
    carries: Optional[RecurrentCarries] = None,
    q_reduce: Callable[..., jax.Array] = jnp.min,
    a_expert: Optional[jax.Array] = None,
    actions: Optional[jax.Array] = None,
    executed: Optional[Callable[[jax.Array, Any], jax.Array]] = None,
) -> Tuple[S, Tuple[jax.Array, jax.Array, jax.Array], dict]:
    """One actor step on SAC's soft policy loss plus the extensions'
    actor-loss terms, which also read a_expert, the expert's stored actions
    at the rows (None without an expert), and actions, the rows' own. With
    a residual policy the critics score executed(a, a_expert). Returns the
    state, (loss, log_probs, q), the SAC loss before the extensions' terms,
    and the extensions' metrics."""
    agent_state, policy_key = agent_state.next_keys()
    alpha = temperature.get_alpha(agent_state)
    actor_state = agent_state.actor_state
    transform = make_scored_action(executed, a_expert)

    def loss_fn(params: FrozenDict) -> Tuple[jax.Array, Any]:
        loss, (log_probs, q, pi) = compute_soft_policy_loss(
            params,
            actor_state,
            agent_state.critic_state,
            observations,
            alpha,
            policy_key,
            carries,
            q_reduce,
            transform,
        )
        actor_batch = {
            "observations": observations,
            "raw_observations": raw_observations,
            "pi_mean": pi.mean(),
            "actor_params": params,
            "actor_state": actor_state,
            "a_expert": a_expert,
            "actions": actions,
            "critic_state": agent_state.critic_state,
            "agent_loss": loss,
        }
        extra, metrics = extension_stack.compute_extensions_actor_loss(
            agent_state,
            actor_batch,
            agent_state.collector_state.timestep,
            derive_extensions_key(policy_key),
            total_timesteps,
        )
        return loss + extra, ((loss, log_probs, q), metrics)

    actor_state, (out, metrics), _ = gradient_step(actor_state, loss_fn)
    return agent_state.replace(actor_state=actor_state), out, metrics


def update_temperature(
    agent_state: S,
    log_probs: jax.Array,
    target_entropy: jax.Array,
    loss_fn: Callable[..., Tuple[jax.Array, temperature.TemperatureAuxiliaries]] = (
        compute_temperature_loss
    ),
) -> Tuple[S, temperature.TemperatureAuxiliaries]:
    """The 0.4.0 form of ajax.agents.temperature.update_temperature, SAC's
    temperature loss by default (0.5.0 removes it)."""
    return temperature.update_temperature(
        agent_state, log_probs, target_entropy, loss_fn
    )


_SAC = "ajax.agents.SAC.SAC"
_new_homes = {
    "CoreCriticAux": "ajax.agents.critics",
    "TemperatureAuxiliaries": "ajax.agents.temperature",
    "create_alpha_train_state": "ajax.agents.temperature",
    "critic_loss_fn": "ajax.agents.critics:compute_mean_critic_loss",
    "update_target_networks": "ajax.agents.update_steps",
    "sample_next_actions": _SAC,
    "compute_td_target": _SAC,
    "soft_bellman_target": f"{_SAC}:compute_soft_bellman_target",
    "soft_policy_loss": f"{_SAC}:compute_soft_policy_loss",
    "soft_policy_objective": f"{_SAC}:compute_soft_policy_objective",
    "temperature_loss_fn": f"{_SAC}:compute_temperature_loss",
    "alpha_temperature_loss_fn": f"{_SAC}:compute_alpha_temperature_loss",
}
__getattr__ = moved_names(__name__, _new_homes)
