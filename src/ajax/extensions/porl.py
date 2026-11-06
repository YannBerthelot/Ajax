"""PORL (Xiao et al. 2025, arXiv 2505.16856): online fine-tuning of a
pre-trained policy, with no dataset and no pre-trained critic.

Algorithm 1 runs two stages on one replay, empty at the start. For T steps the
policy stays frozen and acts epsilon-greedily while a fresh critic takes M
updates per step; from step T + 1 it is plain SAC. :class:`EpsilonGreedyPresample`
is the first stage's behaviour, and :func:`porl` gives the rest as the agent's
own settings, CloningPretrain standing for the pre-trained policy.

M, the critics and their architecture stay the agent's (num_critic_updates,
num_critics, critic_architecture), so the arm runs on any backbone. The paper's
Table 1 is num_critic_updates=16, num_critics=10, actor_learning_rate=1e-4,
alpha_learning_rate=1e-4, batch_size=512 (1024 on Kitchen and RLBench), LayerNorm
critics of four layers of 256 (three of 512 on Adroit and Kitchen) and
presample_steps=20_000 (5_000 on Adroit).

A stateful expert's clone needs the expert's memory in its observations, through
features: ExpertStateInput.for_expert(expert), exact, with the agent's
expert_policy and so one expert query per step; or PIDTermsInput, which needs no
expert but holds the error's plain integral, so it parts from a PI whose
anti-windup holds its integrator (four_tank's, while its pumps saturate).

Departures from the paper:

* The policy is a clone of an expert. Its rollouts are plant steps the paper
  does not count (ajax.ledger charges them) and they stay out of the replay.
* The paper's networks read no observation statistics; the clone's do, so holding
  its weights would not hold the policy (see :class:`EpsilonGreedyPresample`).
  The stage's rows never count in them.
* SAC's critic steps of one update share one batch, and its actor step reads the
  same batch. The paper draws M batches and steps the actor on their union. REDQ
  draws one per critic step, and steps its actor on the last.
* The paper does not say how its target reads ten critics. SAC takes the minimum
  of all of them; REDQ takes it over a random subset_size of them.
* The paper gives no starting temperature: the stage holds the agent's alpha_init,
  and fine-tuning starts from it.
* The exploiting action is the clone's mean, a reading rather than the paper's
  words ("actions from the pre-trained policy"): it calls the stage greedy, and
  its figure 9 compares it with deterministic sampling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from ajax.controller import transforms_of
from ajax.environments.interaction import get_pi
from ajax.environments.obs_norm import mirror_obs_norm
from ajax.extensions.base import Extension, ExtensionContext
from ajax.extensions.pretrain import CloningPretrain
from ajax.random_keys import derive_key


@dataclass(frozen=True)
class EpsilonGreedyPresample(Extension):
    """For the first presample_steps env steps each env takes a uniform action
    with probability epsilon and the actor's mean otherwise; then the agent's
    own action. A uniform warm-up keeps its actions. The replay stores the
    action taken.

    The observation statistics hold where pretraining left them through those
    steps, to the one row each step counts before it acts. Frozen weights alone
    are no frozen policy: the random actions reach states the clone's rollouts
    did not, and those rows would move the statistics the weights read."""

    presample_steps: int = 20_000
    epsilon: float = 0.1
    name: str = "epsilon_greedy_presample"
    action_slot: ClassVar[str] = "pre_warmup"
    requires: ClassVar[frozenset[str]] = frozenset(
        {"actor", "continuous_actions", "replay"}
    )

    def check_stack(self, stack: Sequence[Extension]) -> None:
        """The actor's mean is the executed action only without an action
        transform."""
        if transforms_of(stack):
            raise ValueError(
                "EpsilonGreedyPresample executes the actor's mean as it is: it"
                " takes no action transform (ResidualPolicy, ExpertGains)."
            )

    def action(
        self,
        agent_state: Any,
        ext_state: Any,
        obs: Any,
        rng: jax.Array,
        context: ExtensionContext,
    ) -> jax.Array:
        del ext_state
        actor = agent_state.actor_state
        greedy = get_pi(actor, actor.params, obs["observations"])[0].mean()
        choice_key = derive_key(rng, "presample_choice")
        action_key = derive_key(rng, "presample_action")
        explores = jax.random.uniform(choice_key, (*greedy.shape[:-1], 1))
        uniform = jax.random.uniform(action_key, greedy.shape, minval=-1.0, maxval=1.0)
        chosen = jnp.where(explores < self.epsilon, uniform, greedy)
        presampling = context.step < self.presample_steps
        return jnp.where(presampling, chosen, obs["post_warmup_action"])

    def pretrain(
        self, agent_state: Any, ext_state: Any, context: ExtensionContext
    ) -> tuple[Any, Any]:
        """Keep the statistics as the pretraining before it left them."""
        del ext_state, context
        return agent_state, agent_state.collector_state.obs_norm_info

    def post_update(
        self, agent_state: Any, ext_state: Any, context: ExtensionContext
    ) -> tuple[Any, Any]:
        """Put the kept statistics back, up to the stage's last update."""
        if ext_state is None:  # an agent reading no statistics
            return agent_state, ext_state
        collector = agent_state.collector_state
        holding = context.step <= self.presample_steps
        held = jax.tree.map(
            lambda kept, live: jnp.where(holding, kept, live),
            ext_state,
            collector.obs_norm_info,
        )
        collector = collector.replace(obs_norm_info=held)
        agent_state = agent_state.replace(collector_state=collector)
        return mirror_obs_norm(agent_state, held), ext_state


def porl(
    cloning: CloningPretrain,
    presample_steps: int = 20_000,
    epsilon: float = 0.1,
    features: tuple[Extension, ...] = (),
) -> dict[str, Any]:
    """The agent's keyword arguments for PORL: clone, then a critic warm-up on
    presample_steps env steps (a multiple of n_envs), then fine-tuning. Its
    critic updates start with the first step. Only they run in the warm-up:
    Algorithm 1 has no actor or temperature step before the one following step
    T + 1, which the start gates count in steps collected."""
    first_actor_step = presample_steps + 1
    return {
        "extensions": (
            *features,
            cloning,
            EpsilonGreedyPresample(presample_steps, epsilon),
        ),
        "learning_starts": 0,
        "policy_update_start": first_actor_step,
        "alpha_update_start": first_actor_step,
    }


__all__ = ["EpsilonGreedyPresample", "porl"]
