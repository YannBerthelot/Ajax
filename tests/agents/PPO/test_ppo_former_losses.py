"""The losses' 0.4.0 forms, train_PPO.value_loss_function and
policy_loss_function, kept until 0.5.0: each call warns and returns what
compute_value_loss and compute_policy_loss return on the same inputs.
grokking_units (rl/ppo_probe.py) calls them in the two feedforward forms of
the first test."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.PPO import train_PPO
from ajax.agents.PPO.PPO import PPO, compute_policy_loss, compute_value_loss
from ajax.networks.memory import MemoryConfig

NET = ("16", "relu")
OLD = "ajax.agents.PPO.train_PPO"
NEW = "ajax.agents.PPO.PPO"


def _networks_and_batch(shape: tuple, memory: Any = None) -> tuple:
    """A PPO's actor and critic, and random observations, actions, stored
    log-probabilities, advantages and value targets of the given leading
    shape (its last axis the envs)."""
    agent = PPO(
        "Pendulum-v1",
        n_envs=shape[-1],
        actor_architecture=NET,
        critic_architecture=NET,
        memory=memory,
    )
    state = agent.init(jax.random.PRNGKey(0), agent.make_context(1)[1])
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    obs = jax.random.normal(keys[0], (*shape, 3))
    rest = [jax.random.normal(key, (*shape, 1)) for key in keys[1:]]
    return state.actor_state, state.critic_state, obs, *rest


def _assert_same(old: Any, new: Any) -> None:
    jax.tree.map(np.testing.assert_array_equal, old, new)


def test_grokking_units_calls_return_the_new_losses() -> None:
    actor, critic, obs, actions, log_probs, gae, targets = _networks_and_batch((8,))
    key = jax.random.PRNGKey(2)
    with pytest.warns(FutureWarning, match=f"{OLD}.value_loss_function is {NEW}"):
        old_value = train_PPO.value_loss_function(
            critic.params, critic, obs, targets, False, 0.5
        )
    with pytest.warns(FutureWarning, match=f"{OLD}.policy_loss_function is {NEW}"):
        old_policy = train_PPO.policy_loss_function(
            actor.params,
            actor,
            obs,
            actions,
            log_probs,
            gae,
            False,
            0.2,
            0.01,
            True,
            entropy_rng=key,
        )
    minibatch = {
        "obs": obs,
        "actions": actions,
        "log_probs": log_probs,
        "raw_actions": None,
    }
    new_value = compute_value_loss(critic.params, critic, minibatch, targets, 0.5)
    new_policy = compute_policy_loss(
        actor.params, actor, minibatch, gae, 0.2, 0.01, True, entropy_key=key
    )
    _assert_same(old_value, new_value)
    _assert_same(old_policy, new_policy)


def test_a_recurrent_call_replays_from_the_memory_it_is_given() -> None:
    """The remaining keywords too: the extensions' term, the pre-squash
    actions, the episode starts and each network's starting memory."""
    memory = MemoryConfig(kind="gru", hidden_size=4)
    actor, critic, obs, actions, log_probs, gae, targets = _networks_and_batch(
        (4, 2), memory
    )
    resets = jnp.zeros((4, 2), bool).at[2, 1].set(True)

    def extra(params: Any, network: Any, observations: jax.Array, last: Any) -> tuple:
        return 0.1 * jnp.mean(observations), {"extra": jnp.mean(observations)}

    with pytest.warns(FutureWarning):
        old_value = train_PPO.value_loss_function(
            critic.params,
            critic,
            obs,
            targets,
            recurrent=True,
            vf_coef=0.5,
            extra_loss_fn=extra,
            resets=resets,
            initial_hidden=critic.hidden_state,
        )
        old_policy = train_PPO.policy_loss_function(
            actor.params,
            actor,
            obs,
            actions,
            log_probs,
            gae,
            recurrent=True,
            clip_coef=0.2,
            ent_coef=0.0,
            advantage_normalization=False,
            extra_loss_fn=extra,
            raw_actions=actions,
            resets=resets,
            initial_hidden=actor.hidden_state,
        )
    minibatch = {
        "obs": obs,
        "actions": actions,
        "log_probs": log_probs,
        "raw_actions": actions,
        "resets": resets,
        "critic_hidden": critic.hidden_state,
        "actor_hidden": actor.hidden_state,
    }
    new_value = compute_value_loss(
        critic.params, critic, minibatch, targets, 0.5, extra
    )
    new_policy = compute_policy_loss(
        actor.params, actor, minibatch, gae, 0.2, 0.0, False, extra
    )
    _assert_same(old_value, new_value)
    _assert_same(old_policy, new_policy)
