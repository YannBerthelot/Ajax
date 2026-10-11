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
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from flax.serialization import to_state_dict

from ajax.environments.utils import get_state_action_shapes
from ajax.networks.networks import Encoder, build_cnn_encoder, init_network_state
from ajax.networks.utils import get_adam_tx
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import ActivationFunction


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


def _build_q_encoder(module: nn.Module) -> nn.Module:
    """Encoder for a Q-network: a `CNNEncoder` over flat image obs when
    `cnn_image_shape` is set (architecture from `cnn_spec`), else the
    legacy MLP `Encoder`."""
    if module.cnn_image_shape is not None:
        return build_cnn_encoder(
            module.cnn_image_shape, module.cnn_extra_obs_dim, module.cnn_spec
        )
    return Encoder(
        input_architecture=module.input_architecture,
        penultimate_normalization=module.penultimate_normalization,
        layer_norm=module.encoder_layer_norm,
    )


class QNetwork(nn.Module):
    """Encoder + linear head producing one Q-value per discrete action."""

    input_architecture: Sequence[Union[str, ActivationFunction]]
    n_actions: int
    penultimate_normalization: bool = False
    # Opt-in LayerNorm on the encoder's output; Mnih et al.'s network has none.
    encoder_layer_norm: bool = False
    # Optional CNN encoder for image obs -- see `NetworkConfig.cnn_image_shape`.
    cnn_image_shape: Optional[Tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
    cnn_spec: Optional[tuple] = None

    def setup(self):
        self.encoder = _build_q_encoder(self)
        self.head = nn.Dense(
            self.n_actions, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    def __call__(self, obs: jax.Array) -> GreedyQPolicy:
        features = self.encoder(obs)
        # Penultimate encoder features, exposed for plasticity/conditioning
        # probes. `sow` is a no-op unless the caller marks "intermediates"
        # mutable, so it costs nothing during training.
        self.sow("intermediates", "encoder_features", features)
        return GreedyQPolicy(self.head(features))


class DuelingQNetwork(QNetwork):
    """Dueling DQN architecture (Wang et al., 2016).

    A shared encoder feeds two heads -- a scalar state-value ``V(s)`` and a
    per-action advantage ``A(s, a)`` -- recombined as

        Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

    The mean-subtraction fixes the unidentifiability of the V/A split.
    Select it via ``DQN(..., q_network_cls=DuelingQNetwork)``.
    """

    def setup(self):
        self.encoder = _build_q_encoder(self)
        init = {"kernel_init": orthogonal(1.0), "bias_init": constant(0.0)}
        self.value_head = nn.Dense(1, **init)
        self.advantage_head = nn.Dense(self.n_actions, **init)

    def __call__(self, obs: jax.Array) -> GreedyQPolicy:
        features = self.encoder(obs)
        # See QNetwork.__call__: zero-cost probe seam for conditioning metrics.
        self.sow("intermediates", "encoder_features", features)
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
    :class:`QNetwork`; :class:`DuelingQNetwork`, PQN's network), built with
    QNetwork's constructor arguments.
    """
    cls = q_network_cls if q_network_cls is not None else QNetwork
    network = cls(
        input_architecture=network_config.critic_architecture,
        n_actions=n_actions,
        penultimate_normalization=network_config.penultimate_normalization,
        encoder_layer_norm=network_config.encoder_layer_norm,
        cnn_image_shape=network_config.cnn_image_shape,
        cnn_extra_obs_dim=network_config.cnn_extra_obs_dim,
        cnn_spec=network_config.cnn_spec,
    )
    tx = get_adam_tx(**to_state_dict(optimizer_config))
    observation_shape, _ = get_state_action_shapes(env_config.env)
    # One observation: the parameters do not depend on the batch.
    init_obs = jnp.zeros((1, *observation_shape))
    return init_network_state(init_x=init_obs, network=network, key=key, tx=tx)
