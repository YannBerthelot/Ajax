"""Deterministic controller network for analytic policy gradient agents.

``Controller`` maps an observation to an action deterministically:

    encoder (optional MLP) -> memory (optional, any ``MemoryConfig`` kind)
    -> LayerNorm -> Dense(action_dim) -> PID head (optional) -> tanh (optional)

The LayerNorm is the memory's own when it ends with one (transformer,
mamba): Busetto et al.'s code (buswayne/in-context-controller,
``transformer_onestep.py``) runs its blocks, their final LayerNorm, then
the output layer, with no second norm.

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

from collections.abc import Mapping
from dataclasses import asdict, dataclass
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
    """Hyper-parameters of the controller's PID output layer (the fields of
    :class:`~ajax.modules.pid_head.PIDOutputHead` but its width)."""

    use_p: bool = True
    use_i: bool = True
    use_d: bool = True
    kp_init: float = 1.0
    ki_init: float = 0.0
    kd_init: float = 0.0
    anti_windup: Optional[float] = None


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
            # Kept from Ajax's earlier encoder: a LayerNorm on its output
            # (the shared default is now a plain MLP).
            Encoder(input_architecture=self.input_architecture, layer_norm=True)
            if len(self.input_architecture) > 0
            else None
        )
        self.memory_cell = MemoryCell(self.memory) if self.memory is not None else None
        normalised = self.memory is not None and self.memory.ends_normalised
        self.norm = None if normalised else nn.LayerNorm()
        self.out = nn.Dense(self.action_dim, name="out")
        self.pid_head = (
            PIDOutputHead(n_outputs=self.action_dim, **asdict(self.pid))
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
        h = self.encoder(obs) if self.encoder is not None else obs
        if not self.stateful:
            return distrax.Deterministic(self._squash(self._readout(h)))
        if hidden_state is None or done is None:
            raise ValueError("A stateful Controller requires hidden_state and done.")
        memory_carry, pid_carry = hidden_state
        resets = done.reshape(obs.shape[:2]).astype(bool)
        if self.memory_cell is not None:
            memory_carry, h = self.memory_cell(memory_carry, h, resets)
        z = self._readout(h)
        if self.pid_head is not None:  # squashed after the PID head
            pid_carry, z = self.pid_head(pid_carry, z, resets)
        return distrax.Deterministic(self._squash(z)), (memory_carry, pid_carry)

    def _readout(self, h):
        return self.out(h if self.norm is None else self.norm(h))

    def _squash(self, z):
        return jnp.tanh(z) if self.squash else z


def controller_params_with_final_norm(
    params: Mapping[str, Any], memory: MemoryConfig
) -> dict[str, Any]:
    """A Controller's parameters saved before its transformer or mamba
    memory ended with a final LayerNorm, mapped onto the network built now.

    The Controller then applied its own LayerNorm (``norm``) to the memory's
    output, before the output layer; it now leaves that to the core's final
    LayerNorm (``ln_f``). Both are the same LayerNorm at the same place, so
    moving ``norm`` to ``memory_cell/core/ln_f`` keeps the outputs identical.
    ``params`` is the variables dict as saved (``{"params": ...}``),
    ``memory`` the new network's config. Only the parameters map: an
    optimizer state saved with them keeps the old tree, so a resumed
    training state must restart its optimizer.

    Expand-then-contract (CLAUDE.md, refactoring rule 10): goes with
    ``MemoryConfig.final_norm``.
    """
    if not memory.ends_normalised:
        raise ValueError(
            f"A {memory.kind} memory with final_norm={memory.final_norm} has no"
            " final LayerNorm to map the Controller's onto: the Controller keeps"
            " its own and its parameters restore unchanged."
        )
    inner = dict(params["params"])
    cell = dict(inner.get("memory_cell", {}))
    core = dict(cell.get("core", {}))
    if "norm" not in inner or not core or "ln_f" in core:
        raise ValueError(
            "These are not a Controller's parameters saved before its memory"
            " ended with ln_f: a 'norm' beside 'memory_cell', no 'ln_f' in its"
            " core."
        )
    core["ln_f"] = inner.pop("norm")
    return {**params, "params": {**inner, "memory_cell": {**cell, "core": core}}}


__all__ = [
    "Controller",
    "ControllerCarry",
    "PIDHeadConfig",
    "controller_params_with_final_norm",
]
