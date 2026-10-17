"""TD3 deterministic actor.

Matches the original TD3 architecture (Fujimoto et al., 2018, Table 1):
state -> Dense -> ReLU -> Dense -> ReLU -> Dense(action_dim) -> tanh, the
tanh :class:`~ajax.networks.networks.Squashed`'s.

No LayerNorm, no log_std head. Final layer uses uniform(-3e-3, 3e-3) init
per the paper / original DDPG init recipe.
"""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
from flax.linen.initializers import constant

from ajax.environments.utils import get_action_dim
from ajax.modules.pid_actor import PIDLaw, PIDPolicy, check_pid_actor
from ajax.networks.distributions import Deterministic
from ajax.networks.memory import MemoryCell, MemoryConfig
from ajax.networks.networks import get_initialized_actor_critic
from ajax.networks.utils import (
    parse_architecture,
    uniform_init,
)
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import ActivationFunction


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

    def setup(self):
        self.hidden = nn.Sequential(parse_architecture(self.input_architecture))
        if self.memory is not None:
            self.memory_cell = MemoryCell(self.memory)
        # Final layer init: uniform(-3e-3, 3e-3) -- standard DDPG/TD3 final-layer init.
        self.head = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(3e-3),
            bias_init=constant(0.0),
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

    def setup(self):
        self.pid = PIDLaw(self.policy, self.action_dim)

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
