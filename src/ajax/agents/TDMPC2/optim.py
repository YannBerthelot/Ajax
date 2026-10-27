"""TD-MPC2's optimisers: the paper-era torch Adam groups and the torch
gradient clip, with the world model's clip norm counting the policy's stale
gradients (tdmpc2_spec 2.9, 2.10; docs/world_models/deviations.md section
2). The single-task agent and the multi-task trainer both build them through
ajax.agents.TDMPC2.core.create_update_state.
"""

from __future__ import annotations

from typing import Any, Union

import jax
import jax.numpy as jnp
import optax

from ajax.networks.utils import get_adam_tx
from ajax.types import FloatOrCallable

Params = Any

# torch.nn.utils.clip_grad_norm_ adds this to the norm before dividing.
_CLIP_EPS = 1e-6


def clip_grad_norm(
    grads: Params, max_norm: float, extra_sq_norm: Union[float, jax.Array] = 0.0
) -> tuple[Params, jax.Array]:
    """``torch.nn.utils.clip_grad_norm_``: clipped gradients and the pre-clip norm.

    ``norm = sqrt(||grads||^2 + extra_sq_norm)``;
    ``grads * min(1, max_norm / (norm + 1e-6))``. ``extra_sq_norm`` adds the
    squared norm of gradients that are not returned: the paper-era
    world-model clip runs over ``model.parameters()``, which include the
    policy prior's stale post-clip gradients of the previous update
    (``tdmpc2.py:184, 236, 271``; deviations.md §2).
    """
    norm = jnp.sqrt(optax.tree_utils.tree_norm(grads, squared=True) + extra_sq_norm)
    coef = jnp.minimum(1.0, max_norm / (norm + _CLIP_EPS))
    return jax.tree_util.tree_map(lambda g: g * coef, grads), norm


def scale_learning_rate(
    learning_rate: FloatOrCallable, scale: float
) -> FloatOrCallable:
    """The learning rate times scale, a schedule's at every step."""
    if callable(learning_rate):
        schedule = learning_rate
        return lambda count: schedule(count) * scale
    return learning_rate * scale


def make_world_model_tx(
    learning_rate: FloatOrCallable = 3e-4,
    enc_lr_scale: float = 0.3,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """World-model Adam with the encoder's learning rate scaled (``tdmpc2.py:21-27``).

    Two parameter groups (``optax.multi_transform``): ``encoder`` at
    ``learning_rate * enc_lr_scale``, everything else (dynamics, reward, Q
    and, multi-task, the task embedding: ``tdmpc2.py:26``) at
    ``learning_rate``; torch Adam defaults
    (betas 0.9 / 0.999, eps 1e-8; tdmpc2_spec 2.9). Clipping is not part of
    the transformation: :func:`~ajax.agents.TDMPC2.core.update` clips once
    over all world-model gradients, with the paper-era norm, before the
    groups. ``weight_decay``
    (Ajax's; the reference has none) is :func:`get_adam_tx`'s on both
    groups, so the encoder's is multiplied by its scaled learning rate.
    """

    def labels(params: Params) -> dict[str, str]:
        return {k: "encoder" if k == "encoder" else "rest" for k in params}

    return optax.multi_transform(
        {
            "encoder": get_adam_tx(
                scale_learning_rate(learning_rate, enc_lr_scale),
                eps=1e-8,
                weight_decay=weight_decay,
            ),
            "rest": get_adam_tx(learning_rate, eps=1e-8, weight_decay=weight_decay),
        },
        labels,
    )


def make_policy_tx(
    learning_rate: FloatOrCallable = 3e-4, eps: float = 1e-5, weight_decay: float = 0.0
) -> optax.GradientTransformation:
    """Policy Adam(eps 1e-5) (``tdmpc2.py:28``); clipped in
    :func:`~ajax.agents.TDMPC2.core.update`;
    ``weight_decay`` as :func:`make_world_model_tx`'s."""
    return get_adam_tx(learning_rate, eps=eps, weight_decay=weight_decay)
