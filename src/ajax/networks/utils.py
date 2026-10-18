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
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    weight_decay: float = 0.0,
) -> GradientTransformationExtraArgs:
    """Return an Adam(W) optimizer, clipping the gradients' global norm to
    ``max_grad_norm`` when one is given.

    Unclipped by default. Global-norm clipping is a PPO-family convention
    (PPO passes ``max_grad_norm=0.5``; APO clips both its networks'
    gradients at once, ``train_APO.update_networks``); the off-policy agents
    and the supervised ones run plain Adam, as in their reference
    implementations.

    Args:
        weight_decay: Decoupled weight decay, multiplied by the learning
            rate (``OptimizerConfig.weight_decay``); ``> 0`` selects
            ``optax.adamw``, ``0`` keeps plain Adam.
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
    if max_grad_norm is not None:
        return optax.chain(optax.clip_by_global_norm(max_grad_norm), adam)
    return adam


_ACTIVATIONS: dict[str, ActivationFunction] = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "leaky_relu": nn.leaky_relu,
    "swish": nn.swish,
    "silu": nn.silu,
    "mish": jax.nn.mish,
}


def parse_activation(activation: Union[str, ActivationFunction]) -> ActivationFunction:
    """Parse an activation name, or pass an activation callable through.

    ``"silu"`` is ``x * sigmoid(x)`` (DreamerV3's activation, Table 4 of
    arXiv:2301.04104v2). ``"mish"`` is ``x * tanh(softplus(x))`` (TD-MPC2's
    activation, Sec. 3.1 of arXiv:2310.16828v2; torch ``nn.Mish``).
    """
    if callable(activation):
        return activation
    if isinstance(activation, str) and activation in _ACTIVATIONS:
        return _ACTIVATIONS[activation]
    raise ValueError(
        f"Unrecognized activation {activation!r}; the names are {sorted(_ACTIVATIONS)}"
    )


def parse_function_string(s, context=None):
    """Split ``"name"`` / ``"name(expr)"`` into ``(name, value of expr)``.

    A string that does not start with an identifier (a letter or ``_``),
    such as ``"0.5"`` or ``"-1.0"``, is evaluated as a whole and returned
    as ``(None, value)``. One that does but is neither form, such as
    ``"orthogonal(2.0"`` or ``"np.sqrt(2)"``, raises ``ValueError``
    rather than being read as a bare name.
    """
    if context is None:
        context = {"np": np, "math": math, "jnp": jnp}

    s = s.strip()
    match = re.fullmatch(r"([A-Za-z_]\w*)(?:\((.*)\))?", s)
    if match is None and re.match(r"[A-Za-z_]", s):
        raise ValueError(f"Malformed initializer: {s!r}")
    if match:
        name, expr = match.groups()
        if expr:
            try:
                value = eval(expr, {"__builtins__": {}}, context)
            except Exception as e:
                raise ValueError(f"Error evaluating expression '{expr}': {e}") from e
            return name, value
        else:
            return name, None
    else:
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
    initialization: Optional[Union[str, InitializationFunction]],
    default: Optional[InitializationFunction] = None,
) -> InitializationFunction:
    """Parse an initializer string such as ``"orthogonal(1.0)"``, or pass a
    callable initializer through; ``None`` gives ``default``.

    ``"name"`` calls the factory ``name`` (see :func:`get_initializer`)
    with no argument and ``"name(expr)"`` with the value of ``expr``. A
    string that does not start with a name, such as ``"0.5"`` or
    ``"-1.0"``, gives a constant initializer.
    """
    if initialization is None:
        if default is None:
            raise ValueError("No initializer given, and no default for it.")
        return default
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
) -> Union[nn.Module, ActivationFunction]:
    """Parse one architecture token: a width such as ``"256"`` gives a Dense
    layer (:func:`dense`, kernel ``orthogonal(1.0)`` by default),
    ``"layernorm"`` a new LayerNorm with its own scale and bias, any other
    name or a callable an activation (:func:`parse_activation`).

    ``("256", "layernorm", "relu")`` per hidden layer is RLPD's critic MLP
    (Ball et al., ICML 2023; ``ikostrikov/rlpd``, ``rlpd/networks/mlp.py``).
    """
    if str(layer).isnumeric():
        return dense(int(cast("str", layer)), kernel_init, bias_init, orthogonal(1.0))
    if layer == "layernorm":
        return nn.LayerNorm()
    try:
        return parse_activation(activation=layer)
    except ValueError as e:
        raise ValueError(
            f"Unrecognized layer {layer!r}: a width such as '256', 'layernorm'"
            f" or an activation, {sorted(_ACTIVATIONS)}"
        ) from e


def dense(
    features: int,
    kernel_init: Optional[Union[str, InitializationFunction]],
    bias_init: Optional[Union[str, InitializationFunction]],
    default_kernel_init: InitializationFunction,
    name: Optional[str] = None,
) -> nn.Module:
    """A Dense layer with ``kernel_init`` (``default_kernel_init`` for None)
    and ``bias_init`` (zeros for None), as :func:`parse_initialization`
    reads them; ``kernel_init="torch_linear"`` builds a
    :class:`TorchLinear`, which draws its bias too."""
    if kernel_init == TORCH_LINEAR:
        if bias_init is not None:
            raise ValueError(
                f"{TORCH_LINEAR!r} draws the biases too; got bias_init={bias_init!r}"
            )
        return TorchLinear(features, name=name)
    return nn.Dense(
        features,
        kernel_init=parse_initialization(kernel_init, default=default_kernel_init),
        bias_init=parse_initialization(bias_init, default=constant(0)),
        name=name,
    )


def parse_architecture(
    architecture: Sequence[Union[str, ActivationFunction]],
    kernel_init: Optional[Union[str, InitializationFunction]] = None,
    bias_init: Optional[Union[str, InitializationFunction]] = None,
) -> Sequence[Union[nn.Module, ActivationFunction]]:
    """Parse a list of string/module architecture into a list of jax modules"""
    return [parse_layer(layer, kernel_init, bias_init) for layer in architecture]


def uniform_init(bound: float) -> InitializationFunction:
    def _init(key: jax.Array, shape: tuple, dtype: jnp.dtype) -> jax.Array:
        return jax.random.uniform(
            key,
            shape=shape,
            minval=-bound,
            maxval=bound,
            dtype=dtype,
        )

    return _init


#: The kernel-initializer name that builds :class:`TorchLinear` layers.
TORCH_LINEAR = "torch_linear"


class TorchLinear(nn.Module):
    """A dense layer that starts as torch's ``nn.Linear`` does: its kernel
    and its bias each uniform in ``±1 / sqrt(fan_in)``.

    torch 1.5.1 (``torch/nn/modules/linear.py:79-84``, at its ``v1.5.1``
    tag) draws the weight with ``kaiming_uniform_(a=sqrt(5))``, whose bound
    ``sqrt(3) * sqrt(2 / (1 + a**2)) / sqrt(fan_in)`` is ``1 / sqrt(fan_in)``
    (``torch/nn/init.py:61-69, :322-327``), and the bias with the same
    bound. A layer, not an initializer: a flax bias initializer never sees
    the fan-in. Its parameters are a Dense layer's, at the same path.
    """

    features: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        bound = 1.0 / math.sqrt(x.shape[-1])
        init = uniform_init(bound)
        dense = nn.Dense(self.features, kernel_init=init, bias_init=init)
        nn.share_scope(self, dense)
        return dense(x)
