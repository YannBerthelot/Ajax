"""Q-networks mapping a state to one Q-value per discrete action, DQN's and
its dueling variant: shared by DQN, PQN (the greedy policy, predict_q,
q_network_value) and the instrumentation extension (predict_q).

Unlike the SAC/TD3 ``Critic`` (a state-action value ``Q(s, a) -> scalar``
that takes ``concat(obs, action)`` as input), DQN's network is a
state-value-vector ``Q(s) -> R^{n_actions}``. The greedy policy is the
argmax over that vector; epsilon-greedy exploration is layered on at
collection time via a custom ``action_pipeline``.
"""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import orthogonal
from flax.serialization import to_state_dict

from ajax.controller import feature_width
from ajax.environments.obs_norm import apply_obs_norm
from ajax.environments.utils import get_state_action_shapes
from ajax.networks.networks import CNNEncoder, CNNSpec, Encoder, init_network_state
from ajax.networks.utils import dense, get_adam_tx
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    with_batch_stats,
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


def _build_q_encoder(module: nn.Module) -> nn.Module:
    """Encoder for a Q-network: a `CNNEncoder` over flat image obs when
    `cnn_image_shape` is set (architecture from `cnn_spec`), else the
    MLP `Encoder`."""
    if module.cnn_image_shape is not None:
        return CNNEncoder(
            module.cnn_image_shape, module.cnn_extra_obs_dim, module.cnn_spec
        )
    return Encoder(
        input_architecture=module.input_architecture,
        penultimate_normalization=module.penultimate_normalization,
        kernel_init=module.kernel_init,
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
    cnn_spec: Optional[CNNSpec] = None
    # The MLP encoder's and the Q head's kernel inits (the network config's
    # encoder_kernel_init and value_kernel_init); None keeps parse_layer's
    # default and orthogonal(1.0).
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    head_kernel_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
        self.encoder = _build_q_encoder(self)
        self.head = dense(self.n_actions, self.head_kernel_init, None, orthogonal(1.0))

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
        self.value_head = dense(1, self.head_kernel_init, None, orthogonal(1.0))
        self.advantage_head = dense(
            self.n_actions, self.head_kernel_init, None, orthogonal(1.0)
        )

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
    """Return the raw Q-value vector ``Q(obs)`` of shape ``(..., n_actions)``,
    the observations normalised first when the agent normalises them
    (``q_state.obs_norm_info``)."""
    obs = apply_obs_norm(obs, getattr(q_state, "obs_norm_info", None))
    return q_state.apply_fn(with_batch_stats(q_state, q_params), obs).q_values


def q_network_value(agent_state, obs: jax.Array, action: jax.Array) -> jax.Array:
    """``Q(obs, action)`` of the agent's Q-network (its ``actor_state``): an
    evaluation's value estimate (:class:`ajax.evaluate.ValueEstimate`)."""
    q_state = agent_state.actor_state
    q_values = predict_q(q_state, q_state.params, obs)
    index = action.astype(jnp.int32).reshape(*q_values.shape[:-1], 1)
    return jnp.take_along_axis(q_values, index, axis=-1).squeeze(-1)


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
    # The observation features (env_config.observers) follow the env's
    # columns; a CNN passes them on after its convolutions.
    features = feature_width(env_config)
    network = cls(
        input_architecture=network_config.critic_architecture,
        n_actions=n_actions,
        penultimate_normalization=network_config.penultimate_normalization,
        encoder_layer_norm=network_config.encoder_layer_norm,
        cnn_image_shape=network_config.cnn_image_shape,
        cnn_extra_obs_dim=network_config.cnn_extra_obs_dim + features,
        cnn_spec=network_config.cnn_spec,
        kernel_init=network_config.encoder_kernel_init,
        head_kernel_init=network_config.value_kernel_init,
    )
    tx = get_adam_tx(**to_state_dict(optimizer_config))
    shape, _ = get_state_action_shapes(env_config.env)
    # One observation: the parameters do not depend on the batch.
    init_obs = jnp.zeros((1, *shape[:-1], shape[-1] + features))
    return init_network_state(init_x=init_obs, network=network, key=key, tx=tx)
