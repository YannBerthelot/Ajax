"""PQN Q-network: a Q(s) -> R^{n_actions} network with per-layer LayerNorm.

LayerNorm after every hidden Dense layer is the stabilising ingredient
that lets PQN (Gallici et al., 2024, "Simplifying Deep Temporal
Difference Learning") regress TD targets without a replay buffer or a
target network. The network reuses DQN's :class:`GreedyQPolicy` wrapper
so the shared evaluation loop and the epsilon-greedy action pipeline work
unchanged.
"""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
from flax.linen.initializers import lecun_normal

from ajax.agents.DQN.networks import GreedyQPolicy
from ajax.networks.networks import CNNEncoder, CNNSpec
from ajax.networks.utils import dense, parse_activation
from ajax.types import ActivationFunction, InitializationFunction


class PQNNetwork(nn.Module):
    """MLP Q-network with LayerNorm after each hidden Dense layer.

    The architecture tuple is parsed like Ajax's other networks: numeric
    entries become ``Dense`` layers, string entries become activations.
    Each ``Dense`` is immediately followed by ``LayerNorm`` (so the order
    is Dense -> LayerNorm -> activation), which is PQN's defining trait.

    Returns a :class:`GreedyQPolicy` so it is a drop-in for DQN's
    ``QNetwork`` -- same constructor signature, same return type.
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    n_actions: int
    # Accepted for constructor parity with QNetwork (get_initialized_q_network
    # passes them); PQN always applies LayerNorm, so these flags are unused.
    penultimate_normalization: bool = False
    encoder_layer_norm: bool = False
    # Optional CNN front-end for image obs (see `NetworkConfig.cnn_image_shape`).
    # When set, a `CNNEncoder` maps the flat image obs to features, and the
    # LayerNorm-MLP stack below runs on those features.
    cnn_image_shape: Optional[Tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
    cnn_spec: Optional[CNNSpec] = None
    # The hidden layers' and the Q head's kernel inits, as QNetwork's; None
    # keeps the reference code's below.
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    head_kernel_init: Optional[Union[str, InitializationFunction]] = None

    @nn.compact
    def __call__(self, obs: jax.Array) -> GreedyQPolicy:
        if self.cnn_image_shape is not None:
            x = CNNEncoder(self.cnn_image_shape, self.cnn_extra_obs_dim, self.cnn_spec)(
                obs
            )
        else:
            x = obs
        # Every Dense starts as in PQN's reference code, flax's Dense default:
        # mttga/purejaxql @47af6d7 (purejaxql/pqn_gymnax.py:52, :56) sets no
        # initializer, so kernels are lecun_normal and biases zero.
        for layer in self.input_architecture:
            if str(layer).isnumeric():
                x = dense(int(layer), self.kernel_init, None, lecun_normal())(x)
                x = nn.LayerNorm()(x)
            else:
                x = parse_activation(layer)(x)
        # Penultimate (post-activation) features, exposed for
        # plasticity/conditioning probes. `sow` is a no-op unless the
        # caller marks "intermediates" mutable -- zero training cost.
        self.sow("intermediates", "encoder_features", x)
        q_values = dense(self.n_actions, self.head_kernel_init, None, lecun_normal())(x)
        return GreedyQPolicy(q_values)
