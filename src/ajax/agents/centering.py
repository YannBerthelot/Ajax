"""Centering: an offset a TD target subtracts, learned as an estimate of the
reward rate, so that an undiscounted (average-reward) learner's values stay
bounded and a discounted one's lose the large constant r(pi) / (1 - gamma)
(Naik, Wan, Tomar & Sutton, "Reward Centering", RLC 2024). Each rule is one
choice of relative value iteration's reference function f(Q) (Abounadi,
Bertsekas & Borkar, 2001; Devraj & Meyn's Relative Q-learning, 2021), which
at its fixed point reads the reward rate:

* RVIBatchMean: the batch mean of the target's next values, followed slowly
  (Hisaki & Ono, ICML 2024, Eq. 18), RVI-SAC's;
* TDErrorCentering: moved by the mean TD error (Naik et al. 2024, Eq. 10),
  "value-based reward centering";
* ReferencePoint: the target critics' value at one state-action pair, ASAC's
  Q(0, 0) (Adamczyk et al., 2025) and RVI-SAC's reference variant.

A rule turns the stored offset into the one the target subtracts
(``offset_for_target``), then, once the target is built, into the one stored
for the next update (``next_offset``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class CenteringInputs:
    """What a rule may read: the next values V'(s') the target bootstrapped
    (before gamma and the mask), its TD errors (the centred target minus the
    critics' mean prediction), and target_value_at(observation, action), the
    target critics' mean value at one row (None: zeros)."""

    next_values: jax.Array
    td_errors: Optional[jax.Array] = None
    target_value_at: Optional[Callable] = struct.field(pytree_node=False, default=None)


@dataclass(frozen=True)
class RVIBatchMean:
    """offset <- (1 - rate) offset + rate gain mean_B V'(s') (Hisaki & Ono,
    Eq. 18; yhisaki/average-reward-drl b5b0d31, rvi_sac.py:170, :188).
    0.005 is their Table 1's kappa and their runs' (rvi_sac.yaml)."""

    rate: float = 0.005
    gain: float = 1.0

    def offset_for_target(
        self, offset: jax.Array, inputs: CenteringInputs
    ) -> jax.Array:
        del inputs
        return offset

    def next_offset(self, offset: jax.Array, inputs: CenteringInputs) -> jax.Array:
        batch_mean = self.gain * jnp.mean(inputs.next_values)
        return (1 - self.rate) * offset + self.rate * batch_mean


@dataclass(frozen=True)
class TDErrorCentering:
    """offset <- offset + step_size mean_B delta, delta the TD error of the
    centred target (Naik et al. 2024, Eq. 10, where step_size is eta times the
    critic's step size)."""

    step_size: float

    def offset_for_target(
        self, offset: jax.Array, inputs: CenteringInputs
    ) -> jax.Array:
        del inputs
        return offset

    def next_offset(self, offset: jax.Array, inputs: CenteringInputs) -> jax.Array:
        assert inputs.td_errors is not None, "TDErrorCentering reads the TD errors"
        return offset + self.step_size * jnp.mean(inputs.td_errors)


@dataclass(frozen=True)
class ReferencePoint:
    """offset = gain Q'(s_ref, a_ref), the target critics' mean value at one
    state-action pair, read when the target is built (ASAC's Q(0, 0); zeros
    where observation or action is None)."""

    observation: Optional[tuple[float, ...]] = None
    action: Optional[tuple[float, ...]] = None
    gain: float = 1.0

    def offset_for_target(
        self, offset: jax.Array, inputs: CenteringInputs
    ) -> jax.Array:
        del offset
        assert inputs.target_value_at is not None, "ReferencePoint reads Q'(s, a)"
        reference = inputs.target_value_at(self.observation, self.action)
        return self.gain * jnp.mean(reference)

    def next_offset(self, offset: jax.Array, inputs: CenteringInputs) -> jax.Array:
        del inputs
        return offset


CenteringRule = RVIBatchMean | TDErrorCentering | ReferencePoint
