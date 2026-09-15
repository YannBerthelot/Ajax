"""PID output head: a stateful, learnable PID layer on a network's output.

Busetto et al. 2024 (arXiv:2411.06482, §4.1) close their in-context
controller with a PID layer: the transformer emits a signal ``z_k`` and
the applied input is

    u_k = K_P z_k + K_I sum_{j<=k} z_j + K_D (z_k - z_{k-1})

with three learnable gains. The integral term promotes zero steady-state
error on step references, and the authors report faster training. This
is NOT :mod:`ajax.modules.pid_actor` (gains predicted from the
observation and applied to observation-derived error terms): here the
gains are global parameters and the PID acts on the network's own output,
so it needs a carry (integral and previous signal) like a memory cell.

Interface mirrors :class:`ajax.networks.memory.MemoryCell`: time-major
``(T, B, n)`` inputs with ``(T, B)`` reset flags, ``initialize_carry``,
and ``__call__(carry, z, resets) -> (new_carry, u)``; step-wise use is
the ``T=1`` case of the same code path.
"""

from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

PIDCarry = Tuple[jax.Array, jax.Array]  # (integral (B, n), previous z (B, n))


def init_pid_carry(batch_size: int, n_outputs: int) -> PIDCarry:
    """Zero carry: no accumulated integral, previous signal ``z_{-1} = 0``."""
    zeros = jnp.zeros((batch_size, n_outputs), jnp.float32)
    return zeros, zeros


class PIDOutputHead(nn.Module):
    """Learnable PID on a time-major signal ``z``; see the module docstring.

    Args:
        n_outputs: signal width ``n`` (one gain triple per channel; the
            paper's single-input case is ``n_outputs=1``).
        use_p / use_i / use_d: enable each term; a disabled term has no
            parameter. With only ``use_p`` and ``kp_init=1`` the head is
            the identity at initialisation.
        kp_init / ki_init / kd_init: initial gains.
    """

    n_outputs: int
    use_p: bool = True
    use_i: bool = True
    use_d: bool = True
    kp_init: float = 1.0
    ki_init: float = 0.0
    kd_init: float = 0.0

    def initialize_carry(self, batch_size: int) -> PIDCarry:
        return init_pid_carry(batch_size, self.n_outputs)

    @nn.compact
    def __call__(
        self, carry: PIDCarry, z: jax.Array, resets: jax.Array
    ) -> Tuple[PIDCarry, jax.Array]:
        """``z: (T, B, n)``, ``resets: (T, B)`` -> ``(new_carry, u (T, B, n))``.

        A reset at step ``t`` clears the integral and the previous signal
        before ``z_t`` is consumed, so ``u_t`` is computed as at the start
        of an episode.
        """
        n = self.n_outputs
        gains = {}
        for name, enabled, init in (
            ("kp", self.use_p, self.kp_init),
            ("ki", self.use_i, self.ki_init),
            ("kd", self.use_d, self.kd_init),
        ):
            if enabled:
                gains[name] = self.param(
                    name, nn.initializers.constant(init), (n,), jnp.float32
                )
        if not gains:
            raise ValueError("PIDOutputHead needs at least one of P, I, D enabled")

        T, B = z.shape[:2]
        resets = resets.reshape((T, B)).astype(bool)

        def body(carry, inputs):
            integral, previous = carry
            z_t, reset_t = inputs
            mask = reset_t[:, None]
            integral = jnp.where(mask, 0.0, integral)
            previous = jnp.where(mask, 0.0, previous)
            integral = integral + z_t
            u = jnp.zeros_like(z_t)
            if "kp" in gains:
                u = u + gains["kp"] * z_t
            if "ki" in gains:
                u = u + gains["ki"] * integral
            if "kd" in gains:
                u = u + gains["kd"] * (z_t - previous)
            return (integral, z_t), u

        new_carry, u = jax.lax.scan(body, carry, (z, resets))
        return new_carry, u


__all__ = ["PIDCarry", "PIDOutputHead", "init_pid_carry"]
