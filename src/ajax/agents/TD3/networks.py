"""TD3 deterministic actor and Deterministic distribution wrapper.

Matches the original TD3 architecture (Fujimoto et al., 2018, Table 1):
state -> Dense -> ReLU -> Dense -> ReLU -> Dense(action_dim) -> tanh.

No LayerNorm, no log_std head. The layers' initialisers come from the
network config (TD3 sets torch's nn.Linear default, its reference code's).
"""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct

from ajax.environments.utils import get_action_dim
from ajax.modules.pid_actor import PIDLaw, PIDPolicy, check_pid_actor
from ajax.networks.memory import MemoryCell, MemoryConfig
from ajax.networks.networks import get_initialized_actor_critic
from ajax.networks.utils import dense, parse_architecture, uniform_init
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import ActivationFunction, InitializationFunction


@struct.dataclass
class Deterministic:
    """Tiny distrax-compatible wrapper for deterministic policies.

    Exposes `.mean()` so existing code (the policy loss, the target action
    computation, the ImitationLoss extension) keeps working without branches.
    A pytree of its action, as distrax distributions are of their
    parameters (a replayed bootstrap gathers its steps, recurrent.actor_dist).
    """

    _action: jax.Array

    def mean(self) -> jax.Array:
        return self._action

    def sample(self, seed=None, sample_shape=()) -> jax.Array:
        del seed, sample_shape
        return self._action

    def sample_and_log_prob(self, seed=None) -> Tuple[jax.Array, jax.Array]:
        del seed
        return self._action, jnp.zeros(self._action.shape[:-1] + (1,))

    def entropy(self) -> jax.Array:
        return jnp.zeros(self._action.shape[:-1])

    def mode(self) -> jax.Array:
        return self._action

    def stddev(self) -> jax.Array:
        return jnp.zeros_like(self._action)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """For BC pretraining: a delta distribution's log-density is +∞ at
        x=action, -∞ elsewhere. We use the negative-MSE proxy so that
        maximizing log_prob is equivalent to MSE-minimizing the actor toward
        the target action. Shape: (..., action_dim) -> (..., action_dim)
        (per-dim, summed by caller)."""
        return -((x - self._action) ** 2)


class DeterministicActor(nn.Module):
    """TD3 deterministic actor: hidden_arch + Dense(action_dim) + tanh.

    With ``memory`` set, a memory block sits between the hidden stack and
    the head (see ajax.networks.memory); the call then takes time-major
    (T, B, obs) plus (hidden_state, done) and returns
    (Deterministic, new_hidden_state).
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int
    memory: Optional[MemoryConfig] = None
    # The hidden layers' and the head's kernel inits (as the network
    # config's encoder_kernel_init and mean_kernel_init); None keeps
    # parse_layer's default and, for the head, DDPG's uniform(+-3e-3), the
    # head recurrent TD3 keeps until Ni et al.'s initialisation is audited.
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    head_kernel_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
        self.hidden = nn.Sequential(
            parse_architecture(self.input_architecture, self.kernel_init)
        )
        if self.memory is not None:
            self.memory_cell = MemoryCell(self.memory)
        self.head = dense(
            self.action_dim,
            self.head_kernel_init,
            None,
            uniform_init(3e-3),
            name="head",
        )

    def __call__(self, obs: jax.Array, hidden_state=None, done=None):
        h = self.hidden(obs)
        if self.memory is not None:
            if hidden_state is None or done is None:
                raise ValueError(
                    "Recurrent DeterministicActor requires hidden_state and done flags."
                )
            hidden_state, h = self.memory_cell(hidden_state, h, done)
            return Deterministic(jnp.tanh(self.head(h))), hidden_state
        return Deterministic(jnp.tanh(self.head(h)))


class DeterministicPIDActor(nn.Module):
    """TD3's actor as a PID: ``tanh`` of the PID output, as
    :class:`DeterministicActor` squashes its head (see
    :mod:`ajax.modules.pid_actor`)."""

    policy: PIDPolicy
    action_dim: int

    def setup(self):
        self.pid = PIDLaw(self.policy, self.action_dim)

    def gains(self, obs: jax.Array) -> jax.Array:
        return self.pid.gains(obs)

    def __call__(self, obs: jax.Array) -> Deterministic:
        return Deterministic(jnp.tanh(self.pid(obs)))


def get_initialized_td3_actor_critic(
    key: jax.Array,
    env_config: EnvironmentConfig,
    actor_optimizer_config: OptimizerConfig,
    critic_optimizer_config: OptimizerConfig,
    network_config: NetworkConfig,
    num_critics: int = 2,
) -> Tuple[LoadedTrainState, LoadedTrainState]:
    """TD3-specific init: deterministic actor + standard MultiCritic ensemble
    (a plain MLP, as Fujimoto's critic -- sfujim/TD3, TD3.py, Critic: Linear
    -> ReLU -> Linear -> ReLU -> Linear -- unless ``network_config`` opts
    into the encoder's LayerNorm). A :class:`PIDPolicy` architecture makes
    the actor a :class:`DeterministicPIDActor`. The critic and both
    initialisations are :func:`get_initialized_actor_critic`'s.
    """
    action_dim = get_action_dim(env_config.env, env_config.env_params)
    actor: nn.Module
    if isinstance(network_config.actor_architecture, PIDPolicy):
        check_pid_actor(True, network_config.memory)
        actor = DeterministicPIDActor(network_config.actor_architecture, action_dim)
    else:
        actor = DeterministicActor(
            input_architecture=network_config.actor_architecture,
            action_dim=action_dim,
            memory=network_config.memory,
            kernel_init=network_config.encoder_kernel_init,
            head_kernel_init=network_config.mean_kernel_init,
        )
    return get_initialized_actor_critic(
        key,
        env_config,
        actor_optimizer_config,
        critic_optimizer_config,
        network_config,
        continuous=True,
        action_value=True,
        num_critics=num_critics,
        actor=actor,
    )
