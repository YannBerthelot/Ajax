"""DreamerV3's optimizer: LaProp with adaptive gradient clipping and warmup.

The paper-era ``jaxutils.Optimizer`` (``danijar/dreamerv3@2411f7d``,
unchanged at ``29eb964``: ``dreamerv3/jaxutils.py:398-550`` and the
transformations at ``:660-713``) at the reference's defaults
(``configs.yaml:99``: ``scaler: rms``, ``momentum: True``, ``lr: 4e-5``,
``eps: 1e-20``, ``beta1: 0.9``, ``beta2: 0.999``, ``agc: 0.3``, ``pmin:
1e-3``, ``warmup: 1000``, ``globclip: 0``, ``wd: 0``, ``schedule:
constant``); dreamerv3_spec 4.3-4.8 and Algorithm I. Per update, with ``g``
the gradient:

1. :func:`scale_by_agc` -- adaptive gradient clipping per **tensor** on the
   raw gradient: ``g <- g / max(1, |g| / (agc max(pmin, |p|)))`` with the
   Euclidean norms of the whole tensor and of its current parameters
   (``jaxutils.py:660-674``). The one custom step: ``optax.
   adaptive_grad_clip`` clips per unit (row), as NFNets do;
2. ``optax.scale_by_rms(beta2, eps, eps_in_sqrt=False,
   bias_correction=True)`` -- ``nu <- beta2 nu + (1 - beta2) g^2`` from
   zeros, ``u = g / (sqrt(nu / (1 - beta2^t)) + eps)`` with the epsilon
   *after* the square root (the reference's ``scale_by_rms``,
   ``:677-692``);
3. ``optax.ema(beta1, debias=True)`` -- momentum on the normalised update,
   ``mu <- (1 - beta1) u + beta1 mu``, ``m = mu / (1 - beta1^t)``: the
   reference's ``scale_by_momentum`` (``:695-713``, ``nesterov=False``),
   built from the same ``optax`` helpers ``update_moment`` and
   ``bias_correction``. RMS normalisation then momentum is LaProp, not Adam;
4. ``optax.scale_by_learning_rate`` with :func:`warmup_schedule` --
   ``-lr_k min(k / warmup, 1) m`` with ``k`` the number of earlier updates
   (``:452-455``, ``:510-524``), so the first update has learning rate 0: it
   leaves the parameters unchanged but accumulates the moments.

Two roundings differ from the reference's, by float32 rounding only: optax
multiplies ``g`` by ``1 / (sqrt(nu_hat) + eps)`` where the reference
divides, and folds the warmup factor into the learning rate where the
reference scales the update by it afterwards
(``tests/agents/DreamerV3/test_dreamerv3_train_parity.py`` bounds the
difference).

**float32 caveat** (dreamerv3_spec 4.5). With ``eps = 1e-20`` the first
RMS step is ``sign(g)`` whatever the scale of ``g`` -- unless the increment
``(1 - beta2) g^2`` falls below the smallest normal float32 (``1.18e-38``),
i.e. ``|g| < sqrt(1.18e-38 / (1 - beta2)) ~ 3.4e-18`` at ``beta2 = 0.999``.
XLA flushes such denormals to 0 (CPU, eager and jitted, in Ajax and the
reference alike), so ``nu = 0`` and the step is ``g / eps``: up to about 340
learning rates instead of at most about one.

The reference runs one optimizer over the world model, actor and critic
together (``agent.py:83-91``); Ajax keeps one instance per train state
(world model, actor, critic). Every transformation acts per tensor or per
element, and the three instances count the same updates, so feeding each
the same joint gradient gives the single optimizer's update
(``tests/agents/DreamerV3/test_dreamerv3_actor_critic.py``). The
reference's optional branches that are off by default -- global-norm
clipping, weight decay, per-module learning rates, float16 loss scaling --
are not implemented.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import optax

from ajax.agents.DreamerV3.state import DreamerV3Config, LearningRate


def scale_by_agc(clip: float = 0.3, pmin: float = 1e-3) -> optax.GradientTransformation:
    """Adaptive gradient clipping per tensor (dreamerv3_spec 4.4).

    Each update tensor ``g`` is scaled by ``1 / max(1, |g| / (clip max(pmin,
    |p|)))`` with ``|.|`` the Euclidean norm of the whole flattened tensor and
    ``p`` its current parameters (``jaxutils.py:660-674``): not the per-row
    unit-wise clipping of NFNets (``optax.adaptive_grad_clip``). Needs
    ``params``. Stateless.
    """

    def clip_tensor(update: jax.Array, param: Optional[jax.Array]) -> jax.Array:
        if param is None:
            raise ValueError("scale_by_agc needs the parameters")
        unorm = jnp.linalg.norm(update.flatten(), 2)
        pnorm = jnp.linalg.norm(param.flatten(), 2)
        upper = clip * jnp.maximum(pmin, pnorm)
        return update * (1 / jnp.maximum(1.0, unorm / upper))

    return optax.stateless_with_tree_map(clip_tensor)


def warmup_schedule(
    learning_rate: LearningRate = 4e-5, warmup: int = 1000
) -> Callable[[jax.Array], jax.Array]:
    """``k -> lr_k min(k / warmup, 1)``, ``k`` the number of earlier updates.

    ``optax.scale_by_learning_rate`` reads the count before incrementing it,
    as the reference reads its step (``jaxutils.py:510-513``): update 0 has
    learning rate 0 and update ``warmup`` the full rate (dreamerv3_spec
    4.7). ``learning_rate`` is a float -- the reference's ``optax.scale(-lr)``
    (``:452-455``) -- or a schedule ``k -> lr_k``, which the warmup
    multiplies. ``warmup = 0`` disables the warmup (``if self.warmup > 0``,
    ``:512``).
    """

    def schedule(count: jax.Array) -> jax.Array:
        rate = learning_rate(count) if callable(learning_rate) else learning_rate
        if warmup > 0:
            rate = rate * jnp.clip(count.astype(jnp.float32) / warmup, 0, 1)
        return jnp.asarray(rate, jnp.float32)

    return schedule


def laprop(
    learning_rate: LearningRate = 4e-5,
    *,
    agc: float = 0.3,
    pmin: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-20,
    warmup: int = 1000,
) -> optax.GradientTransformation:
    """The reference's optimizer chain (Algorithm I; module docstring).

    :func:`scale_by_agc` (skipped when ``agc = 0``, as in the reference,
    ``jaxutils.py:433-434``), ``optax.scale_by_rms``, ``optax.ema`` and
    ``optax.scale_by_learning_rate`` with :func:`warmup_schedule` (always a
    schedule, so that the state has the same structure with and without
    warmup). The defaults are 2411f7d's.
    """
    chain = [scale_by_agc(agc, pmin)] if agc else []
    chain += [
        optax.scale_by_rms(beta2, eps, eps_in_sqrt=False, bias_correction=True),
        optax.ema(beta1, debias=True),
        optax.scale_by_learning_rate(warmup_schedule(learning_rate, warmup)),
    ]
    return optax.chain(*chain)


def make_optimizer(config: DreamerV3Config) -> optax.GradientTransformation:
    """:func:`laprop` with the optimizer fields of ``config``."""
    return laprop(
        config.learning_rate,
        agc=config.agc,
        pmin=config.agc_pmin,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        warmup=config.warmup,
    )
