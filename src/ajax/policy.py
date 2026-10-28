"""How a trained agent acts: the part of an evaluation step before the
environment steps, which evaluate runs on its rollout (ajax.evaluate)."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import jax

from ajax.controller import ActingState, Controller, call_expert
from ajax.environments.action_stage import choose_in_evaluation
from ajax.environments.interaction import get_pi
from ajax.networks.distributions import latent


def get_deterministic_action_and_entropy_fn(actor_state, recurrent, continuous):
    """Return a function mapping (obs, done, hidden) → (action, entropy, hidden).

    In recurrent mode the hidden state is an explicit input/output so the
    eval loop can thread it through its carry (the live carry on
    ``actor_state`` belongs to the training envs and has the wrong batch
    size here). Non-recurrent mode passes ``hidden`` through untouched.
    """

    def fn(obs: jax.Array, done: Optional[jax.Array], hidden):
        if actor_state is None:
            raise ValueError("Actor not initialized.")
        state = actor_state.replace(hidden_state=hidden) if recurrent else actor_state
        pi, new_state = get_pi(state, actor_state.params, obs, done, recurrent)
        action = pi.mean() if continuous else pi.mode()
        entropy = latent(pi).entropy()
        if recurrent:
            # drop the single-step time axis added by get_pi
            action = action.squeeze(0)
            entropy = entropy.squeeze(0)
            return action, entropy, new_state.hidden_state
        return action, entropy, hidden

    return fn


class Acted(NamedTuple):
    """One acting step, per env, in agent units."""

    actions: jax.Array  # the executed action
    actor_actions: jax.Array  # the actor's (the gains under gain mode)
    entropy: jax.Array
    actor_hidden: Any  # the actor's memory after the step
    expert_state: Any  # the tracked expert's state after the step
    actor_input: jax.Array  # what the actor read, which the critics score
    # 1 where a chooser handed the expert the step; None without choosers.
    handed: Optional[jax.Array]


def make_act(
    actor_state: Any,
    recurrent: bool,
    continuous: bool,
    expert_policy: Any = None,
    agent_state: Any = None,
    controller: Optional[Controller] = None,
    command: Optional[Any] = None,
) -> Callable[..., Acted]:
    """A pure function acting once, as the evaluation acts:
    act(obs, done, expert_obs, expert_state, actor_hidden, command_value,
    rng) -> Acted.

    It acts as the agent's controller says, on an ActingState: the actor's
    input (Controller.observe), its deterministic action, then the executed
    one (Controller.act). The expert acts in agent units on expert_obs, the
    environment's own observation. A command-conditioned actor reads
    command.actor_input(obs, command_value). The controller's choosers (an
    extension's eval_action phase) may then replace the executed action by
    the expert's (choose_in_evaluation), on keys derived from rng, so they
    move no other draw.
    """
    controller = Controller() if controller is None else controller
    # The statistics the agent's controller normalises with, as they stand.
    obs_norm = (
        agent_state.collector_state.obs_norm_info if controller.normalize_obs else None
    )
    actor = get_deterministic_action_and_entropy_fn(actor_state, recurrent, continuous)

    def act(
        obs: jax.Array,
        done: jax.Array,
        expert_obs: Optional[jax.Array],
        expert_state: Any,
        actor_hidden: Any,
        command_value: Any,
        rng: jax.Array,
    ) -> Acted:
        acting = ActingState(
            obs=obs,
            done=done,
            timestep=None,
            expert_state=expert_state,
            carry=actor_hidden,
            obs_norm=obs_norm,
        )
        observations = actor_input = controller.observe(acting)
        if command is not None:
            actor_input = command.actor_input(actor_input, command_value)

        actor_actions, entropy, new_actor_hidden = actor(
            actor_input, done if recurrent else None, actor_hidden
        )
        actions, new_expert_state = controller.act(
            actor_actions, acting, expert_policy, expert_obs
        )
        handed = None
        if controller.choosers:
            expert_action = None
            if expert_policy is not None:
                expert_action, _ = call_expert(expert_policy, expert_state, expert_obs)
            choice = {
                "executed_action": actions,
                "expert_action": expert_action,
                "observations": observations,  # the critics' input
                "raw_obs": expert_obs,
            }
            choice, handed = choose_in_evaluation(
                agent_state, acting, controller.choosers, choice, rng
            )
            actions = choice["executed_action"]
        return Acted(
            actions,
            actor_actions,
            entropy,
            new_actor_hidden,
            new_expert_state,
            actor_input,
            handed,
        )

    return act
