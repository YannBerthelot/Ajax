"""Deterministic controller network for analytic policy gradient agents.

``Controller`` maps an observation to an action deterministically:

    encoder (optional MLP) -> memory (optional, any ``MemoryConfig`` kind)
    -> LayerNorm -> Dense(action_dim) -> PID head (optional) -> tanh (optional)

With ``memory=MemoryConfig(kind="transformer", ...)`` and ``pid`` set this
is the *contextual controller* of Busetto et al. 2024 (arXiv:2411.06482,
Fig. 1): a decoder-only transformer over the history of observations
closed by a learnable PID layer. With ``memory=None`` and ``pid=None`` it
is a plain MLP controller.

The network follows the recurrent-actor contract used by ``get_pi`` /
``evaluate``: when it carries state (memory and/or PID) it is called with
time-major ``(T, B, obs)`` inputs plus ``hidden_state`` / ``done`` and
returns ``(distribution, new_hidden_state)``; otherwise ``(B, obs)`` in,
distribution out. The distribution is ``distrax.Deterministic`` so
``mean()``/``mode()``/``sample()`` all return the control action and
``entropy()`` is zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from ajax.modules.pid_head import PIDOutputHead, init_pid_carry
from ajax.networks.memory import MemoryCell, MemoryConfig, init_carry
from ajax.networks.networks import Encoder
from ajax.types import ActivationFunction


@dataclass(frozen=True)
class PIDHeadConfig:
    """Hyper-parameters of the controller's PID output layer."""

    use_p: bool = True
    use_i: bool = True
    use_d: bool = True
    kp_init: float = 1.0
    ki_init: float = 0.0
    kd_init: float = 0.0


ControllerCarry = Tuple[Any, Any]  # (memory carry | None, pid carry | None)


class Controller(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int
    memory: Optional[MemoryConfig] = None
    pid: Optional[PIDHeadConfig] = None
    squash: bool = True

    @property
    def stateful(self) -> bool:
        return self.memory is not None or self.pid is not None

    def setup(self):
        self.encoder = (
            Encoder(input_architecture=self.input_architecture)
            if len(self.input_architecture) > 0
            else None
        )
        self.memory_cell = MemoryCell(self.memory) if self.memory is not None else None
        self.norm = nn.LayerNorm()
        self.out = nn.Dense(self.action_dim, name="out")
        self.pid_head = (
            PIDOutputHead(
                n_outputs=self.action_dim,
                use_p=self.pid.use_p,
                use_i=self.pid.use_i,
                use_d=self.pid.use_d,
                kp_init=self.pid.kp_init,
                ki_init=self.pid.ki_init,
                kd_init=self.pid.kd_init,
            )
            if self.pid is not None
            else None
        )

    def initialize_carry(self, rng: jax.Array, batch_size: int) -> ControllerCarry:
        memory_carry = (
            init_carry(self.memory, rng, batch_size)
            if self.memory is not None
            else None
        )
        pid_carry = (
            init_pid_carry(batch_size, self.action_dim)
            if self.pid is not None
            else None
        )
        return (memory_carry, pid_carry)

    def __call__(self, obs, hidden_state=None, done=None):
        if not self.stateful:
            return distrax.Deterministic(self._head(self._embed(obs)))
        if hidden_state is None or done is None:
            raise ValueError("A stateful Controller requires hidden_state and done.")
        memory_carry, pid_carry = hidden_state
        resets = done.reshape(obs.shape[:2]).astype(bool)
        h = self._embed(obs)
        if self.memory_cell is not None:
            memory_carry, h = self.memory_cell(memory_carry, h, resets)
        z = self._head(h)
        if self.pid_head is not None:
            pid_carry, z = self.pid_head(pid_carry, z, resets)
        u = jnp.tanh(z) if self.squash else z
        return distrax.Deterministic(u), (memory_carry, pid_carry)

    def _embed(self, obs):
        return self.encoder(obs) if self.encoder is not None else obs

    def _head(self, h):
        z = self.out(self.norm(h))
        if self.stateful:
            return z  # squash applied after the PID head
        return jnp.tanh(z) if self.squash else z


__all__ = ["Controller", "ControllerCarry", "PIDHeadConfig"]
