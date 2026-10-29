"""The automatic reset cost as a composable :class:`Extension`.

* :class:`ResetCost` — terminations become steps into the reset state,
  charged a cost learned to hold the termination rate at most at a target
  (ajax.agents.reset_cost).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import jax

from ajax.agents.reset_cost import (
    ResetCostSettings,
    ResetCostState,
    create_reset_cost_state,
    reset_charge,
    reset_critic_input,
    step_reset_cost,
    step_reset_critic,
    update_reset_target,
)
from ajax.controller import feature_width
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.extensions.base import Extension, ExtensionContext

#: The batch keys that describe the episode a termination ended, not the
#: reset observation a continuing target reads there.
ENDED_EPISODE_TARGET_KEYS = frozenset({"raw_next_observations", "next_a_expert"})
ENDED_EPISODE_LOSS_KEYS = frozenset({"return_to_go"})


@dataclass(frozen=True)
class ResetCost(ResetCostSettings, Extension):
    """Hisaki & Ono's automatic reset cost (ICML 2024, Section 3.3), which
    their code gives SAC and ARO-DDPG as well as RVI-SAC: the agent's target
    no longer stops at a termination but bootstraps on the reset
    observation the replay stores there, and a terminated row's reward is
    charged c, a Lagrange multiplier that a reset critic's estimate of the
    termination rate drives towards keeping it at most target_rate (the
    settings, ResetCostSettings, are the authors'). Each update, before the
    agent's critics step: the charge from the current c, then one step of
    the reset critic (scored at the agent's next actions, one-hot for
    discrete ones), its rate, one Adam step of c and the reset critic's
    Polyak update at tau, the authors' SAC-with-reset's order.

    It needs a replay, whose next observation after a termination is the
    reset one (an on-policy rollout keeps the final one), and a target scored
    at the policy's next action, where the reset critic is scored too. An
    average-reward agent prices terminations itself (ASAC's penalty,
    RVI-SAC's own reset cost): it is not discounted, and this module refuses
    it. Logs ``reset_cost/cost``, ``reset_cost/rate`` and
    ``reset_cost/critic_loss``.
    """

    tau: float = 0.005
    name: str = "reset_cost"
    env_args: Any = None
    requires: ClassVar[frozenset[str]] = frozenset(
        {"replay", "next_action_target", "discounted", "one_step_target"}
    )
    context: ClassVar[frozenset[str]] = frozenset({"env_args"})
    continues_through_terminations: ClassVar[bool] = True
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "on_target": frozenset(
            {
                "actions",
                "next_actions",
                "terminated",
                "reward_scale",
            }
        )
    }

    def check_stack(self, stack: Sequence[Extension]) -> None:
        """Refuse a stack that reads, past a termination, what belongs to the
        episode that ended: an expert's next action and observation (IBRL,
        CriticBlend, LCBGatedBootstrap score them where the target now reads
        the reset observation) or an episode's return to go (Cal-QL's
        floor)."""
        clashing = [
            e.name
            for e in stack
            if e is not self
            and (
                ENDED_EPISODE_TARGET_KEYS.intersection(e.reads().get("on_target", ()))
                or ENDED_EPISODE_LOSS_KEYS.intersection(
                    e.reads().get("critic_loss", ())
                )
            )
        ]
        if clashing:
            raise ValueError(
                f"{self.name} continues the target through terminations into the"
                f" reset observation; {clashing} read the episode that ended there."
            )

    def bind_to_agent(self, **agent_context: Any) -> "ResetCost":
        """Copy the agent's env_args, which size the reset critic."""
        env_args = agent_context.get("env_args")
        if self.env_args is not None or env_args is None:
            return self
        return dataclasses.replace(self, env_args=env_args)

    def action_count(self) -> Optional[int]:
        """The number of actions a discrete agent chooses from (None for
        continuous ones)."""
        if self.env_args.continuous:
            return None
        return get_action_dim(self.env_args.env, self.env_args.env_params)

    def init_state(self, agent_state: Any, rng: jax.Array) -> ResetCostState:
        del agent_state
        observation_shape, action_shape = get_state_action_shapes(self.env_args.env)
        width = observation_shape[-1] + feature_width(self.env_args)
        width += self.action_count() or action_shape[-1]
        return create_reset_cost_state(
            rng, width, self.architecture, self.learning_rate
        )

    def on_target(
        self,
        agent_state: Any,
        ext_state: ResetCostState,
        batch: Any,
        target: jax.Array,
        context: ExtensionContext,
    ) -> tuple[jax.Array, dict, ResetCostState]:
        del agent_state, context
        missing = [k for k in self.batch_keys["on_target"] if batch.get(k) is None]
        if missing:
            raise ValueError(
                f"{self.name}: the agent passes no {missing} to on_target."
            )
        terminated = batch["terminated"]
        charged = target - batch["reward_scale"] * reset_charge(ext_state, terminated)
        count = self.action_count()
        inputs = reset_critic_input(batch["observations"], batch["actions"], count)
        next_inputs = reset_critic_input(
            batch["next_observations"], batch["next_actions"], count
        )
        state, metrics = step_reset_critic(
            ext_state, inputs, next_inputs, terminated, self.rate_tau, self.gain
        )
        state = step_reset_cost(state, self.target_rate)
        state = update_reset_target(state, self.tau)
        return (
            charged,
            {
                "reset_cost/cost": ext_state.cost.params["cost"],
                "reset_cost/rate": state.rate,
                "reset_cost/critic_loss": metrics["reset_critic_loss"],
            },
            state,
        )


__all__ = ["ResetCost"]
