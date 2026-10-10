"""Critic-ensemble research features as composable :class:`Extension`s.

* :class:`KernelRepulsion` — SVGD-style function-space repulsion between
  the critics of an ensemble (the ``critic_loss`` phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ajax.extensions.base import Extension, ExtensionContext
from ajax.networks.networks import predict_value


def q_kernel_repulsion(q_preds: jax.Array) -> jax.Array:
    """The mean RBF kernel between the critics' Q-functions on a batch.

    ``q_preds`` is ``(num_critics, batch, ...)``; each critic's outputs
    form one feature vector. The bandwidth follows the median heuristic
    (Liu & Wang 2016, SVGD), held constant so the kernel adapts to the
    current spread without a confounding gradient; minimising the mean
    kernel pushes the critics apart in function space.
    """
    n = q_preds.shape[0]
    feats = q_preds.reshape(n, -1)
    diffs = feats[:, None, :] - feats[None, :, :]
    sq_dists = jnp.sum(diffs**2, axis=-1)
    # The median over the full matrix (n zero diagonal entries among n^2)
    # is dominated by the off-diagonal pairs for n >= 4; the 1e-8 floors
    # avoid dividing by zero when every critic predicts the same values.
    h = jax.lax.stop_gradient(
        jnp.median(sq_dists) / (jnp.log(jnp.asarray(n, dtype=feats.dtype)) + 1e-8)
        + 1e-8
    )
    return jnp.exp(-sq_dists / h).mean()


@dataclass(frozen=True)
class KernelRepulsion(Extension):
    """``coef`` times :func:`q_kernel_repulsion` of the critics on the
    critic-loss batch, added to the critic loss (REDQ's SVGD variant).

    It reads the batch every replay agent's critic step passes
    (``observations``, ``actions``, ``critic_params``, ``critic_state``),
    so it costs one more ensemble forward. Feedforward critics only:
    agents with memory reject extensions, so repulsion with recurrent
    memory is unsupported.
    """

    coef: float = 0.1
    name: str = "kernel_repulsion"

    def critic_loss(
        self, agent_state: Any, ext_state: Any, batch: Any, ctx: ExtensionContext
    ) -> jax.Array:
        del agent_state, ext_state, ctx
        x = jnp.concatenate((batch["observations"], batch["actions"]), axis=-1)
        q_preds = predict_value(batch["critic_state"], batch["critic_params"], x)
        return self.coef * q_kernel_repulsion(q_preds)


__all__ = ["KernelRepulsion", "q_kernel_repulsion"]
