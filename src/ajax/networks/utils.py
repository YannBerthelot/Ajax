import math
import re
from collections.abc import Sequence
from typing import Callable, Optional, Union, cast

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
    max_grad_norm: Optional[float] = None,
    eps: float = 1e-5,
    clipped: bool = False,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    weight_decay: float = 0.0,
) -> GradientTransformationExtraArgs:
    """Return an Adam(W) optimizer with optional global-norm gradient clipping.

    Unclipped by default. Global-norm clipping is a PPO-family convention
    (PPO / APO pass ``max_grad_norm=0.5``); the off-policy agents and the
    supervised ones run plain Adam, as in their reference implementations.

    Args:
        learning_rate (Union[float, Callable[[int], float]]): Learning rate for the optimizer.
        max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
        eps (float): Epsilon value for numerical stability.
        clipped (bool): Whether to apply gradient clipping.
        weight_decay (float): Decoupled weight decay; ``> 0`` selects
            ``optax.adamw`` (used by the APG family), ``0`` keeps plain Adam.

    Returns:
        GradientTransformationExtraArgs: The configured optimizer.

    """
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if weight_decay > 0:
        adam = optax.adamw(
            learning_rate=learning_rate,
            eps=eps,
            b1=beta_1,
            b2=beta_2,
            weight_decay=weight_decay,
        )
    else:
        adam = optax.adam(learning_rate=learning_rate, eps=eps, b1=beta_1, b2=beta_2)
    if clipped:
        if max_grad_norm is None:
            raise ValueError("Gradient clipping requested but no norm provided.")
        return optax.chain(optax.clip_by_global_norm(max_grad_norm), adam)
    return adam


def parse_activation(activation: Union[str, ActivationFunction]) -> ActivationFunction:
    """Parse an activation name, or pass an activation callable through.

    ``"silu"`` is ``x * sigmoid(x)`` (DreamerV3's activation, Table 4 of
    arXiv:2301.04104v2). ``"mish"`` is ``x * tanh(softplus(x))`` (TD-MPC2's
    activation, Sec. 3.1 of arXiv:2310.16828v2; torch ``nn.Mish``).
    """
    activation_matching = {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "leaky_relu": nn.leaky_relu,
        "swish": nn.swish,
        "silu": nn.silu,
        "mish": jax.nn.mish,
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
        case _ if callable(activation):
            return activation
        case _:
            raise ValueError(f"Unrecognized activation {activation}")


def parse_function_string(s, context=None):
    """Split ``"name"`` / ``"name(expr)"`` into ``(name, value of expr)``.

    A string that does not start with an identifier (a letter or ``_``),
    such as ``"0.5"`` or ``"-1.0"``, is evaluated as a whole and returned
    as ``(None, value)``.
    """
    if context is None:
        context = {"np": np, "math": math, "jnp": jnp}

    match = re.match(r"([A-Za-z_]\w*)(?:\((.*)\))?", s)
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


def trunc_normal_fan_in(scale: float = 1.0) -> InitializationFunction:
    """Fan-in truncated normal: DreamerV3's weight initializer.

    DreamerV3 (arXiv:2301.04104v2; ``danijar/dreamerv3@2411f7d``,
    ``dreamerv3/nets.py:842-848, :870``, fans ``:873-884``; dreamerv3_spec
    1.5) initialises every Linear, BlockLinear and Conv kernel as::

        W = scale * 1.1368 * sqrt(1 / fan_in) * TruncatedNormal(-2, 2)

    with a unit-sigma normal truncated at +-2 and ``scale`` the layer's
    ``outscale``. This is ``variance_scaling(scale**2, "fan_in",
    "truncated_normal")``: flax divides by the exact standard deviation of
    the truncated unit normal, 0.87962566, where DreamerV3 multiplies by the
    rounded 1.1368. Drawn with the same key the two kernels differ by the
    constant factor 1.13684723 / 1.1368, a residual relative difference of
    4.16e-5 (tested; deviation D23 in ``docs/world_models/deviations.md``).
    Both give ``Std[W] = scale / sqrt(fan_in)``, and flax's
    fan-in matches the reference for every kernel rank >= 2:
    ``shape[-2] * prod(shape[:-2])``, so a BlockLinear kernel
    ``(g, I/g, U/g)`` gets the full input width ``I``.

    ``scale = 0`` gives an all-zero kernel (DreamerV3's reward and value
    heads).
    """
    if scale < 0:
        raise ValueError(f"trunc_normal_fan_in scale must be >= 0, got {scale}")
    return flax_initializers.variance_scaling(scale**2, "fan_in", "truncated_normal")


# Initializers selectable by name besides the factories of
# ``flax.linen.initializers``. Each value is a factory: called with no
# argument for ``"name"``, with the parsed number for ``"name(number)"``.
_AJAX_INITIALIZERS: dict[str, Callable[..., InitializationFunction]] = {
    # flax's ``zeros`` is an initializer, not a factory (``zeros_init`` is).
    "zeros": lambda: flax_initializers.zeros,
    "trunc_normal_fan_in": trunc_normal_fan_in,
}


def get_initializer(name: str) -> Callable[..., InitializationFunction]:
    """Factory for the initializer called ``name``.

    Looks in Ajax's own registry first, then in ``flax.linen.initializers``.
    """
    if name in _AJAX_INITIALIZERS:
        return _AJAX_INITIALIZERS[name]
    try:
        return getattr(flax_initializers, name)
    except AttributeError as e:
        raise ValueError(
            f"Initializer '{name}' not found in Ajax's registry"
            f" ({sorted(_AJAX_INITIALIZERS)}) or in flax.linen.initializers"
        ) from e


def parse_initialization(
    initialization: Union[str, InitializationFunction],
) -> InitializationFunction:
    """Parse an initializer string such as ``"orthogonal(1.0)"``, or pass a
    callable initializer through.

    ``"name"`` calls the factory ``name`` (see :func:`get_initializer`)
    with no argument and ``"name(expr)"`` with the value of ``expr``. A
    string that does not start with a name, such as ``"0.5"`` or
    ``"-1.0"``, gives a constant initializer.
    """
    if callable(initialization):
        return initialization
    initialization_name, number = parse_function_string(initialization)
    if number is not None and initialization_name is None:
        return constant(float(number))
    init_fn = get_initializer(initialization_name)
    return init_fn() if number is None else init_fn(number)


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
