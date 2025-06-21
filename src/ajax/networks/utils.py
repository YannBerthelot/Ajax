import math
import re
from collections.abc import Sequence
from typing import Callable, Optional, Union, cast, get_args

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen import initializers as flax_initializers
from flax.linen.initializers import (
    constant,
    orthogonal,
)
from optax import GradientTransformationExtraArgs

from ajax.types import ActivationFunction, InitializationFunction


def get_adam_tx(
    learning_rate: Union[float, Callable[[int], float]] = 1e-3,
    max_grad_norm: Optional[float] = 0.5,
    eps: float = 1e-5,
    clipped=True,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
) -> GradientTransformationExtraArgs:
    """Return an Adam optimizer with optional gradient clipping.

    Args:
        learning_rate (Union[float, Callable[[int], float]]): Learning rate for the optimizer.
        max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
        eps (float): Epsilon value for numerical stability.
        clipped (bool): Whether to apply gradient clipping.

    Returns:
        GradientTransformationExtraArgs: The configured optimizer.

    """
    # # if not isinstance(learning_rate, float):
    # learning_rate = (
    #     1.0  # deactivate learning_rate here, to handle it custom in the training loop
    # )
    if clipped:
        if max_grad_norm is None:
            raise ValueError("Gradient clipping requested but no norm provided.")
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=eps, b1=beta_1, b2=beta_2),
        )
    return optax.adam(learning_rate=learning_rate, eps=eps, b1=beta_1, b2=beta_2)


def parse_activation(activation: Union[str, ActivationFunction]) -> ActivationFunction:  # type: ignore[return]
    """Parse string representing activation or jax activation function towards\
        jax activation function
    """
    activation_matching = {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "leaky_relu": nn.leaky_relu,
    }

    match activation:
        case str():
            if activation in activation_matching:
                return cast("ActivationFunction", activation_matching[activation])
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


def parse_function_string(s, context=None):
    if context is None:
        context = {"np": np, "math": math, "jnp": jnp}

    match = re.match(r"(\w+)(?:\((.*)\))?", s)
    if match:
        name, expr = match.groups()
        if expr:
            try:
                # Evaluate the expression in a limited context
                value = eval(expr, {"__builtins__": {}}, context)
            except Exception as e:
                raise ValueError(f"Error evaluating expression '{expr}': {e}") from e
            return name, value
        else:
            return name, None
    else:
        # Try to parse it as a raw value
        try:
            value = eval(s, {"__builtins__": {}}, context)
            return None, value
        except Exception as e:
            raise ValueError(f"Invalid input: {s}") from e


def get_initializer(name):
    try:
        return getattr(flax_initializers, name)
    except AttributeError as e:
        raise ValueError(
            f"Initializer '{name}' not found in flax.linen.initializers"
        ) from e


def parse_initialization(initialization: str) -> InitializationFunction:  # type: ignore[return]
    """Parse string representing activation or jax activation function towards\
        jax activation function
    """

    initialization_name, number = parse_function_string(initialization)
    # initialization_matching = {
    #     "constant": constant,
    #     "orthogonal": orthogonal,
    #     "xavier_uniform": xavier_uniform,
    #     "he_normal": he_normal,
    #     "he_uniform": he_uniform,
    # }
    if number is not None and initialization_name is None:
        return constant(float(number))
    init_fn = get_initializer(initialization_name)
    if init_fn is not None:
        return init_fn() if number is None else init_fn(number)
    # if initialization_name in initialization_matching:
    #     init_fn = initialization_matching[initialization_name]
    #     return init_fn() if number is None else init_fn(number)

    # raise ValueError(
    #     (
    #         f"Unrecognized initialization name {initialization}, acceptable"
    #         f" initializations names are : {initialization_matching.keys()}"
    #     ),
    # )


def parse_layer(
    layer: Union[str, ActivationFunction],
    kernel_init: Optional[Union[str, InitializationFunction]] = None,
    bias_init: Optional[Union[str, InitializationFunction]] = None,
) -> Union[nn.Dense, ActivationFunction]:
    """Parse a layer representation into either a Dense or an activation function"""
    if kernel_init is None:
        kernel_init = orthogonal(1.0)
    elif isinstance(kernel_init, str):
        kernel_init = parse_initialization(kernel_init)
    if bias_init is None:
        bias_init = constant(0)
    elif isinstance(bias_init, str):
        bias_init = parse_initialization(bias_init)

    if str(layer).isnumeric():
        return nn.Dense(
            int(cast("str", layer)),
            kernel_init=kernel_init,
            bias_init=bias_init,
        )
    return parse_activation(activation=layer)


def parse_architecture(
    architecture: Sequence[Union[str, ActivationFunction]],
    kernel_init: Optional[Union[str, InitializationFunction]] = None,
    bias_init: Optional[Union[str, InitializationFunction]] = None,
) -> Sequence[Union[nn.Dense, ActivationFunction]]:
    """Parse a list of string/module architecture into a list of jax modules"""
    return [parse_layer(layer, kernel_init, bias_init) for layer in architecture]


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(
            key,
            shape=shape,
            minval=-bound,
            maxval=bound,
            dtype=dtype,
        )

    return _init
