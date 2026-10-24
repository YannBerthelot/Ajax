"""Deterministic actors and their actor-critic builder: TD3's (Fujimoto et
al., 2018, Table 1), shared by TD3 and DAgger, which builds its actor with
TD3's builder.

state -> Dense -> ReLU -> Dense -> ReLU -> Dense(action_dim) -> tanh, the
tanh :class:`~ajax.networks.networks.Squashed`'s. No LayerNorm, no log_std
head. The layers' initialisers come from the network config (TD3 sets
torch's nn.Linear default, its reference code's).
"""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax

from ajax.controller import policy_action_dim
from ajax.extensions.observe import pid_terms_at
from ajax.modules.pid_actor import PIDLaw, PIDPolicy, check_pid_actor
from ajax.networks.distributions import Deterministic
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


class DeterministicActor(nn.Module):
    """TD3 deterministic actor: hidden_arch + Dense(action_dim), squashed
    by :func:`get_initialized_td3_actor_critic`.

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
            return Deterministic(self.head(h)), hidden_state
        return Deterministic(self.head(h))


class DeterministicPIDActor(nn.Module):
    """TD3's actor as a PID: the PID output, squashed as
    :class:`DeterministicActor`'s head is (see :mod:`ajax.modules.pid_actor`)."""

    policy: PIDPolicy
    action_dim: int
    terms_at: int = 0  # PIDLaw's

    def setup(self):
        self.pid = PIDLaw(self.policy, self.action_dim, self.terms_at)

    def gains(self, obs: jax.Array) -> jax.Array:
        return self.pid.gains(obs)

    def __call__(self, obs: jax.Array) -> Deterministic:
        return Deterministic(self.pid(obs))


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
    the actor a :class:`DeterministicPIDActor`. The critic, the ``tanh``
    on the actor and both initialisations are
    :func:`get_initialized_actor_critic`'s.
    """
    action_dim = policy_action_dim(env_config)  # gain mode's gains
    actor: nn.Module
    if isinstance(network_config.actor_architecture, PIDPolicy):
        check_pid_actor(True, network_config.memory)
        policy = network_config.actor_architecture
        at = pid_terms_at(env_config, policy.n_errors)
        actor = DeterministicPIDActor(policy, action_dim, at)
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
        squash=True,
        num_critics=num_critics,
        actor=actor,
    )
