"""Helpfer functions for Networks initialization"""

import functools
from collections.abc import Sequence
from typing import Any, Optional, TypeAlias, Union, cast, get_args

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

ActivationFunction: TypeAlias = Any  # Union[
#     jax._src.custom_derivatives.custom_jvp,
#     jaxlib.xla_extension.PjitFunction,
# ]


def check_architecture(actor: bool, num_of_actions: Optional[int]) -> None:
    """Make sure the architecture is valid w.r.t actor and number of actions"""
    if actor and num_of_actions is None:
        raise ValueError("Actor mode is selected but no num_of_actions provided")
    if actor and not isinstance(num_of_actions, int):
        raise ValueError(
            f"Got unexpected num of actions : {num_of_actions} {type(num_of_actions)}",
        )


def reset_hidden_state_where_episode_finished(resets, hidden_state, reset_hidden_state):
    """Reset the hidden states for the environments that have terminated"""
    return jnp.where(
        resets[:, np.newaxis],
        reset_hidden_state,
        hidden_state,
    )


class ScannedRNN(nn.Module):
    """Scan-compatible version of a jax RNN cell"""

    features: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, hidden_state, obs_and_resets):
        """Applies the module."""
        obs, resets = obs_and_resets
        reset_hidden_state = self.initialize_carry(jax.random.PRNGKey(0), obs.shape[0])
        hidden_state = reset_hidden_state_where_episode_finished(
            resets,
            hidden_state,
            reset_hidden_state,
        )

        new_hidden_state, embedding = nn.GRUCell(self.features)(hidden_state, obs)
        return new_hidden_state, embedding

    def initialize_carry(self, key, batch_size):
        """Initialize the hidden_state of the RNN layer"""
        return nn.GRUCell(self.features, parent=None).initialize_carry(
            key,
            (batch_size, self.features),
        )


def parse_activation(activation: Union[str, ActivationFunction]) -> ActivationFunction:  # type: ignore[return]
    """Parse string representing activation or jax activation function towards\
        jax activation function
    """
    activation_matching = {"relu": nn.relu, "tanh": nn.tanh}

    match activation:
        case str():
            if activation in activation_matching:
                return activation_matching[activation]
            raise ValueError(
                (
                    f"Unrecognized activation name {activation}, acceptable activations"
                    f" names are : {activation_matching.keys()}"
                ),
            )
        case activation if isinstance(activation, get_args(ActivationFunction)):
            return activation
        case _:
            raise ValueError(f"Unrecognized activation {activation}")


def parse_layer(
    layer: Union[str, ActivationFunction],
) -> Union[nn.Dense, ActivationFunction]:
    """Parse a layer representation into either a Dense or an activation function"""
    if str(layer).isnumeric():
        return nn.Dense(
            features=int(cast("str", layer)),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )
    return parse_activation(activation=layer)


def parse_architecture(
    architecture: Sequence[Union[str, ActivationFunction]],
) -> Sequence[Union[nn.Dense, ActivationFunction]]:
    """Parse a list of string/module architecture into a list of jax modules"""
    layers = []
    for layer in architecture:
        layers.append(parse_layer(layer))
        # if "LSTM" in layer:
        #     if i > 0:
        #         raise ValueError("Only one LSTM layer is allowed")
        #     i += 1
    return layers
    # return [parse_layer(layer, i) for i, layer in enumerate(architecture)]
