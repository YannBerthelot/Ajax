"""Dense layers and normalised MLP shared by DreamerV3 and TD-MPC2.

Both world-model agents build every network from the same hidden layer,
``Dense -> Norm -> Activation`` with the bias kept, and differ only in its
configuration:

* **DreamerV3** (arXiv:2301.04104v2 Table 4, "RMSNorm + SiLU";
  ``danijar/dreamerv3@2411f7d:dreamerv3/nets.py:367-403`` MLP,
  ``:594-635`` Linear, ``:724-758`` Norm, ``:816-884`` Initializer):
  RMSNorm with epsilon 1e-4 and a learnable scale only, SiLU, kernels from
  the fan-in truncated normal, output layers scaled by ``outscale``
  (0 for the reward and value heads, 0.01 for the policy).
* **TD-MPC2** (arXiv:2310.16828v2 Sec. 3.1 and App. A, "LayerNorm + Mish",
  1 % dropout after the first linear layer of each Q-function;
  ``nicklashansen/tdmpc2@5f6fade:tdmpc2/common/layers.py:85-122``
  NormedLinear and mlp, ``common/init.py:4-17`` weight_init): LayerNorm
  with epsilon 1e-5 and a learnable scale and shift, Mish, dropout between
  the first Linear and its norm, kernels from N(0, 0.02^2), final weights of
  the reward and Q heads zeroed.

flax's defaults match neither paper (epsilon 1e-6, and LayerNorm's fast
variance ``E[x^2] - E[x]^2``, off by ~0.2 on mean-1000 unit-variance
inputs where torch's two-pass variance is within 1e-4), so
:class:`NormedMLP` sets them explicitly. Statistics are computed in
float32: DreamerV3 at 2411f7d computed the RMS in its bf16 compute dtype,
and Ajax runs these agents in float32 throughout (deviation D4 in
``docs/world_models/deviations.md``). Output layers and heads are composed
by the agents with :func:`linear`.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from ajax.networks.utils import parse_activation, parse_initialization
from ajax.types import ActivationFunction, InitializationFunction


def _scaled(init: InitializationFunction, outscale: float) -> InitializationFunction:
    """``init`` multiplied by ``outscale`` (DreamerV3 ``nets.py:870``)."""
    if outscale == 1.0:
        return init

    def scaled_init(key: jax.Array, shape: Any, dtype: Any = jnp.float32) -> jax.Array:
        return outscale * init(key, shape, dtype)

    return scaled_init


def linear(
    features: int,
    kernel_init: Union[str, InitializationFunction],
    bias_init: Union[str, InitializationFunction] = "zeros",
    outscale: float = 1.0,
    name: Optional[str] = None,
) -> nn.Dense:
    """``nn.Dense`` with initializers given by name and an output scale.

    The papers' Linear layer: the bias is always present and
    zero-initialised, and never scaled (DreamerV3 ``nets.py:594-635``,
    ``bias: True``; TD-MPC2 ``nn.Linear`` with ``weight_init`` setting the
    bias to 0). ``outscale`` multiplies the kernel initializer;
    ``outscale = 0`` gives an exactly zero kernel, as in DreamerV3's reward
    and value heads (``configs.yaml:122, :129``) and TD-MPC2's reward and Q
    heads (``common/world_model.py:30``, ``zero_`` on their last weights).
    Parameters are float32 ``kernel [in, features]`` and ``bias
    [features]``; leading input dimensions are batch dimensions. Call it
    where a module would be constructed (inside ``setup`` or an
    ``nn.compact`` method).

    Args:
        features: output width.
        kernel_init: initializer or initializer name (see
            ``parse_initialization``), e.g. ``"trunc_normal_fan_in"``
            (DreamerV3) or ``"normal(0.02)"`` (TD-MPC2).
        bias_init: initializer or name for the bias; both papers use zeros.
        outscale: factor applied to the kernel initializer's samples.
        name: the module name; flax's automatic ``Dense_<i>`` when None.
    """
    return nn.Dense(
        features,
        kernel_init=_scaled(parse_initialization(kernel_init), outscale),
        bias_init=parse_initialization(bias_init),
        name=name,
    )


class NormedMLP(nn.Module):
    """Stack of ``layers`` hidden layers ``Dense -> [Dropout] -> Norm -> Act``.

    Every layer has width ``units``. Dropout (inverted, rate ``dropout``)
    is applied only after the Dense of the first layer and only when
    ``deterministic`` is False, using the ``'dropout'`` RNG stream
    (TD-MPC2 ``layers.py:120``, ``dropout*(i==0)``). There is no output
    layer: compose one with :func:`linear`. Use :meth:`dreamerv3` and
    :meth:`tdmpc2` for the papers' configurations.

    Args:
        layers: number of hidden layers.
        units: width of every hidden layer.
        act: activation or activation name (see ``parse_activation``).
        norm: ``"rms"`` (scale only) or ``"layer"`` (scale and shift, exact
            two-pass variance).
        norm_eps: epsilon added to the mean square or variance.
        kernel_init: initializer or initializer name of the Dense kernels.
        dropout: dropout rate of the first layer; 0 disables it.
    """

    layers: int
    units: int
    act: Union[str, ActivationFunction]
    norm: Literal["layer", "rms"]
    norm_eps: float
    kernel_init: Union[str, InitializationFunction]
    dropout: float = 0.0

    @classmethod
    def dreamerv3(cls, layers: int, units: int, **module_kwargs: Any) -> NormedMLP:
        """DreamerV3's MLP trunk: RMSNorm(eps 1e-4) and SiLU, fan-in init.

        ``nets.py:390-391`` applies ``layers`` x ``Linear(units, act='silu',
        norm='rms')`` with ``winit: normal`` and fan ``in``
        (``configs.yaml:113-129``, ``nets.py:600-601``); RMSNorm's default
        epsilon is 1e-4 (``nets.py:728``).
        """
        return cls(
            layers=layers,
            units=units,
            act="silu",
            norm="rms",
            norm_eps=1e-4,
            kernel_init="trunc_normal_fan_in",
            **module_kwargs,
        )

    @classmethod
    def tdmpc2(
        cls, layers: int, units: int, dropout: float = 0.0, **module_kwargs: Any
    ) -> NormedMLP:
        """TD-MPC2's hidden NormedLinear layers: LayerNorm(eps 1e-5) and Mish.

        ``layers.py:85-100`` (Linear -> Dropout -> LayerNorm -> Mish, torch
        LayerNorm epsilon 1e-5) and ``init.py:7-9``: torch's
        ``trunc_normal_(std=0.02)`` truncates at the absolute bounds +-2,
        i.e. 100 standard deviations, so it is N(0, 0.02^2), hence
        ``"normal(0.02)"``; jax's ``truncated_normal(0.02)`` would cut at
        +-0.04 and shrink the standard deviation to 0.0176. TD-MPC2 sets
        ``dropout=0.01`` for the Q-functions only.
        """
        return cls(
            layers=layers,
            units=units,
            act="mish",
            norm="layer",
            norm_eps=1e-5,
            kernel_init="normal(0.02)",
            dropout=dropout,
            **module_kwargs,
        )

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: Optional[bool] = None) -> jax.Array:
        """Apply the trunk to ``x [..., in]``; returns ``[..., units]``.

        ``deterministic`` must be given when ``dropout > 0``: True disables
        dropout (evaluation, TD targets, planning), False enables it and
        needs a ``'dropout'`` RNG.
        """
        act = parse_activation(self.act)
        for i in range(self.layers):
            x = linear(self.units, self.kernel_init)(x)
            if i == 0 and self.dropout > 0.0:
                x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
            if self.norm == "rms":
                x = nn.RMSNorm(epsilon=self.norm_eps, use_scale=True)(x)
            elif self.norm == "layer":
                x = nn.LayerNorm(epsilon=self.norm_eps, use_fast_variance=False)(x)
            else:
                raise ValueError(
                    f"NormedMLP norm must be 'rms' or 'layer', got {self.norm!r}"
                )
            x = act(x)
        return x
