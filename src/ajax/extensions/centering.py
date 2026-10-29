"""Centering any discounted agent's values as a composable :class:`Extension`.

* :class:`ValueCentering` — its TD target minus an offset learned as the
  reward rate (ajax.agents.centering's rules).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp

from ajax.agents.centering import (
    CenteringInputs,
    CenteringRule,
    ReferencePoint,
    RVIBatchMean,
    TDErrorCentering,
)
from ajax.extensions.base import Extension, ExtensionContext

#: The on_target batch keys each rule reads, beyond the target itself.
RULE_KEYS: Mapping[type, frozenset[str]] = {
    RVIBatchMean: frozenset({"next_values"}),
    TDErrorCentering: frozenset({"next_values", "predictions"}),
    ReferencePoint: frozenset(
        {"next_values", "observations", "actions", "target_value_at"}
    ),
}


@dataclass(frozen=True)
class ValueCentering(Extension):
    """Reward centering (Naik, Wan, Tomar & Sutton, RLC 2024): the agent's TD
    target minus an offset that learns the reward rate, so that its values
    lose the constant r(pi) / (1 - gamma) and stay small as gamma nears 1.
    The rule is one of relative value iteration's reference functions
    (ajax.agents.centering): the batch mean of the next values, followed at
    rate 0.005 (Hisaki & Ono, ICML 2024; RVI-SAC's, the default), the mean
    TD error at step_size (Naik et al.'s value-based centering, step_size =
    eta times the critic's step size), or the target critics' value at a
    reference state-action pair (ASAC's Q(0, 0)). The offset starts at 0 and
    moves once an update, after the target has read it.

    It shifts a one-step target (r + gamma V(s')); a lambda-return sums
    rewards over many steps, which a constant shift does not centre (PPO,
    APO, PQN). An average-reward agent (ASAC, APO, RVI-SAC) centres its
    values itself: it is not discounted, and this module refuses it. It
    comes last of the extensions shaping the target, so that it centres the
    target the critic fits. Naik et al. centre continuing tasks: on a task
    whose episodes terminate, every step's offset also prices an episode's
    length (a shorter one pays fewer), unless the target goes on through
    terminations (ResetCost, before it in the stack). Logs
    ``centering/offset``.
    """

    rule: CenteringRule = field(default_factory=RVIBatchMean)
    name: str = "centering"
    requires: ClassVar[frozenset[str]] = frozenset({"discounted", "one_step_target"})

    def reads(self) -> Mapping[str, frozenset[str]]:
        return {"on_target": RULE_KEYS[type(self.rule)]}

    def check_stack(self, stack: Sequence[Extension]) -> None:
        """Refuse a stack where another extension shapes the target after
        it: the critic would fit a target it did not centre."""
        place = next(i for i, e in enumerate(stack) if e is self)
        later = [
            e.name for e in stack[place + 1 :] if "on_target" in e.implemented_phases()
        ]
        if later:
            raise ValueError(
                f"{self.name} centres the target the critic fits, so it comes last"
                f" of the extensions shaping it; {later} come after it."
            )

    def init_state(self, agent_state: Any, rng: jax.Array) -> jax.Array:
        del agent_state, rng
        return jnp.zeros(())

    def on_target(
        self,
        agent_state: Any,
        ext_state: jax.Array,
        batch: Any,
        target: jax.Array,
        context: ExtensionContext,
    ) -> tuple[jax.Array, dict, jax.Array]:
        del agent_state, context
        missing = [k for k in self.reads()["on_target"] if batch.get(k) is None]
        if missing:
            raise ValueError(
                f"{self.name}: the agent passes no {missing} to on_target."
            )
        inputs = CenteringInputs(
            jax.lax.stop_gradient(batch["next_values"]),
            target_value_at=reference_scorer(batch),
        )
        offset = jax.lax.stop_gradient(self.rule.offset_for_target(ext_state, inputs))
        centred = target - offset
        if isinstance(self.rule, TDErrorCentering):
            td_errors = jax.lax.stop_gradient(centred - batch["predictions"]())
            inputs = dataclasses.replace(inputs, td_errors=td_errors)
        next_offset = jax.lax.stop_gradient(self.rule.next_offset(offset, inputs))
        return centred, {"centering/offset": offset}, next_offset


def reference_scorer(batch: Any) -> Optional[Any]:
    """The batch's target_value_at at one row given as tuples (None: zeros,
    as wide as the batch's observations and actions), or None where the
    batch has none."""
    target_value_at = batch.get("target_value_at")
    if target_value_at is None:
        return None
    observations, actions = batch["observations"], batch["actions"]

    def at(observation: Optional[tuple], action: Optional[tuple]) -> jax.Array:
        row = jnp.zeros((1, observations.shape[-1]), observations.dtype)
        if observation is not None:
            row = jnp.asarray(observation, observations.dtype).reshape(1, -1)
        taken = jnp.zeros((1, actions.shape[-1]), actions.dtype)
        if action is not None:
            taken = jnp.asarray(action, actions.dtype).reshape(1, -1)
        return target_value_at(row, taken)

    return at


__all__ = ["ValueCentering"]
