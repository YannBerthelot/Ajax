"""DQN Q-network: maps a state to one Q-value per discrete action.

Unlike the SAC/TD3 ``Critic`` (a state-action value ``Q(s, a) -> scalar``
that takes ``concat(obs, action)`` as input), DQN's network is a
state-value-vector ``Q(s) -> R^{n_actions}``. The greedy policy is the
argmax over that vector; epsilon-greedy exploration is layered on at
collection time via a custom ``action_pipeline``.

``QNetwork.__call__`` returns a :class:`GreedyQPolicy` -- a tiny
distrax-compatible wrapper (mirroring TD3's ``Deterministic``) so the
shared evaluation loop (``ajax.evaluate.evaluate`` -> ``get_pi``) can call
``.mode()`` / ``.entropy()`` on it unchanged, while DQN's own loss reads
the raw Q-vector via ``.q_values``.
"""

from collections.abc import Sequence
from typing import Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from flax.serialization import to_state_dict

from ajax.environments.utils import get_state_action_shapes
from ajax.networks.networks import Encoder, init_network_state
from ajax.networks.utils import get_adam_tx, parse_initialization
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import ActivationFunction, InitializationFunction


class GreedyQPolicy:
    """Distrax-compatible wrapper around a Q-value vector.

    Mirrors TD3's ``Deterministic`` wrapper: it exposes the minimal
    distribution interface the shared eval / interaction code expects
    (``.mode()``, ``.entropy()``, ``.sample_and_log_prob()``) so DQN can
    reuse that infrastructure without branches, while keeping the raw
    Q-values reachable via ``.q_values`` for the TD loss and the
    epsilon-greedy action pipeline.

    The implied policy is greedy: ``mode == sample == argmax_a Q(s, a)``.
    Exploration is *not* baked in here -- it is applied at collection
    time by the epsilon-greedy ``action_pipeline``.
    """

    def __init__(self, q_values: jax.Array):
        self.q_values = q_values

    def mode(self) -> jax.Array:
        return jnp.argmax(self.q_values, axis=-1)

    def mean(self) -> jax.Array:
        # No meaningful mean for a discrete greedy policy; eval uses
        # ``mode`` for discrete action spaces. Kept for interface parity.
        return jnp.argmax(self.q_values, axis=-1)

    def sample(self, seed=None, sample_shape=()) -> jax.Array:
        del seed, sample_shape
        return jnp.argmax(self.q_values, axis=-1)

    def sample_and_log_prob(self, seed=None):
        del seed
        action = jnp.argmax(self.q_values, axis=-1)
        return action, jnp.zeros(action.shape, dtype=jnp.float32)

    def entropy(self) -> jax.Array:
        """Entropy of ``softmax(Q)`` -- a Boltzmann-policy diagnostic.

        Not used by the algorithm; logged at eval as a cheap proxy for
        how peaked the Q-values are.
        """
        log_p = jax.nn.log_softmax(self.q_values, axis=-1)
        return -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)

    def log_prob(self, value: jax.Array) -> jax.Array:
        log_p = jax.nn.log_softmax(self.q_values, axis=-1)
        return jnp.take_along_axis(log_p, value[..., None], axis=-1)[..., 0]


class QNetwork(nn.Module):
    """Encoder + linear head producing one Q-value per discrete action."""

    input_architecture: Sequence[Union[str, ActivationFunction]]
    n_actions: int
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
        self.encoder = Encoder(
            input_architecture=self.input_architecture,
            penultimate_normalization=self.penultimate_normalization,
        )
        kernel_init = (
            orthogonal(1.0)
            if self.kernel_init is None
            else parse_initialization(self.kernel_init)
        )
        bias_init = (
            constant(0.0)
            if self.bias_init is None
            else parse_initialization(self.bias_init)
        )
        self.head = nn.Dense(self.n_actions, kernel_init=kernel_init, bias_init=bias_init)

    def __call__(self, obs: jax.Array, raw_obs=None) -> GreedyQPolicy:
        del raw_obs
        return GreedyQPolicy(self.head(self.encoder(obs)))


class DuelingQNetwork(nn.Module):
    """Dueling DQN architecture (Wang et al., 2016).

    A shared encoder feeds two heads -- a scalar state-value ``V(s)`` and a
    per-action advantage ``A(s, a)`` -- recombined as

        Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

    The mean-subtraction fixes the unidentifiability of the V/A split.
    Drop-in compatible with :class:`QNetwork`: same constructor kwargs,
    same ``GreedyQPolicy`` return type. Select it via ``DQN(...,
    q_network_cls=DuelingQNetwork)``.
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    n_actions: int
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
        self.encoder = Encoder(
            input_architecture=self.input_architecture,
            penultimate_normalization=self.penultimate_normalization,
        )
        kernel_init = (
            orthogonal(1.0)
            if self.kernel_init is None
            else parse_initialization(self.kernel_init)
        )
        bias_init = (
            constant(0.0)
            if self.bias_init is None
            else parse_initialization(self.bias_init)
        )
        self.value_head = nn.Dense(1, kernel_init=kernel_init, bias_init=bias_init)
        self.advantage_head = nn.Dense(
            self.n_actions, kernel_init=kernel_init, bias_init=bias_init
        )

    def __call__(self, obs: jax.Array, raw_obs=None) -> GreedyQPolicy:
        del raw_obs
        features = self.encoder(obs)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        q_values = value + advantage - jnp.mean(advantage, axis=-1, keepdims=True)
        return GreedyQPolicy(q_values)


def predict_q(
    q_state: LoadedTrainState,
    q_params,
    obs: jax.Array,
) -> jax.Array:
    """Return the raw Q-value vector ``Q(obs)`` of shape ``(..., n_actions)``."""
    return q_state.apply_fn(q_params, obs).q_values


def get_initialized_q_network(
    key: jax.Array,
    env_config: EnvironmentConfig,
    optimizer_config: OptimizerConfig,
    network_config: NetworkConfig,
    n_actions: int,
    q_network_cls: Optional[type] = None,
) -> LoadedTrainState:
    """Build the DQN Q-network train state (online params == target params).

    ``q_network_cls`` selects the network module (defaults to
    :class:`QNetwork`); pass :class:`DuelingQNetwork` for the dueling
    architecture. Both share the same constructor signature.
    """
    cls = q_network_cls if q_network_cls is not None else QNetwork
    network = cls(
        input_architecture=network_config.critic_architecture,
        n_actions=n_actions,
        penultimate_normalization=network_config.penultimate_normalization,
    )
    tx = get_adam_tx(**to_state_dict(optimizer_config))
    observation_shape, _ = get_state_action_shapes(env_config.env)
    init_obs = jnp.zeros((env_config.n_envs, *observation_shape))

    return init_network_state(
        init_x=init_obs,
        network=network,
        key=key,
        tx=tx,
        recurrent=False,
        lstm_hidden_size=None,
        n_envs=env_config.n_envs,
        lr_schedule=optimizer_config.learning_rate,
    )
