"""PQN Q-network: a Q(s) -> R^{n_actions} network with per-layer LayerNorm.

LayerNorm after every hidden Dense layer is the stabilising ingredient
that lets PQN (Gallici et al., 2024, "Simplifying Deep Temporal
Difference Learning") regress TD targets without a replay buffer or a
target network. The network reuses DQN's :class:`GreedyQPolicy` wrapper
so the shared evaluation loop and the epsilon-greedy action pipeline work
unchanged.
"""

from collections.abc import Sequence
from typing import Optional, Union

import flax.linen as nn
import jax
from flax.linen.initializers import constant, orthogonal

from ajax.agents.DQN.networks import GreedyQPolicy
from ajax.networks.utils import parse_activation, parse_initialization
from ajax.types import ActivationFunction, InitializationFunction


def _resolve_init(
    spec: Optional[Union[str, InitializationFunction]],
    default: InitializationFunction,
) -> InitializationFunction:
    """Resolve an init spec to a callable.

    ``None`` -> ``default``; a string -> looked up via ``parse_initialization``;
    an already-callable initializer is returned unchanged.
    """
    if spec is None:
        return default
    if isinstance(spec, str):
        return parse_initialization(spec)
    return spec


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
    # passes it); PQN always applies LayerNorm, so this flag is unused.
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None

    @nn.compact
    def __call__(self, obs: jax.Array, raw_obs=None) -> GreedyQPolicy:
        del raw_obs
        kernel_init = _resolve_init(self.kernel_init, orthogonal(2.0**0.5))
        bias_init = _resolve_init(self.bias_init, constant(0.0))
        x = obs
        for layer in self.input_architecture:
            if str(layer).isnumeric():
                x = nn.Dense(int(layer), kernel_init=kernel_init, bias_init=bias_init)(
                    x
                )
                x = nn.LayerNorm()(x)
            else:
                x = parse_activation(layer)(x)
        q_values = nn.Dense(
            self.n_actions, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(x)
        return GreedyQPolicy(q_values)
