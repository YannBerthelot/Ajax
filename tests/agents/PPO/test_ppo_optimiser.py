"""PPO's optimiser is ppo2's (openai/baselines @ea25b9e): one Adam with eps
1e-5 (``ppo2/model.py:100``) down the policy loss plus ``vf_coef`` 0.5 times
half the squared value error (``ppo2.py:22``, ``model.py:75``, ``:91``), its
gradient clipped to global norm 0.5 over every parameter at once
(``model.py:105-107``)."""

from __future__ import annotations

from typing import cast

import jax
import numpy as np
import optax

from ajax.agents import steps
from ajax.agents.PPO.PPO import PPO, PPOConfig


def test_PPO_defaults_to_ppo2s_optimiser() -> None:
    agent = PPO(env_id="Pendulum-v1")
    config = cast(PPOConfig, agent.agent_config)
    assert (config.max_grad_norm, config.vf_coef) == (0.5, 0.5)
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.learning_rate, args.eps, args.max_grad_norm) == (3e-4, 1e-5, None)


def test_joint_step_is_one_clipped_adam_over_both_networks() -> None:
    """Three joint steps of PPO's own actor and critic equal three steps of
    one optax Adam (eps 1e-5) over both parameter trees together, clipped to
    their joint norm 0.5. The gradients' scales (norms ~4, ~0.04, ~1.2) make the
    clip bind on some steps and not others, and the actor's and critic's
    norms differ, so clipping each network on its own would move them
    differently."""
    agent = PPO(env_id="Pendulum-v1", n_envs=2, n_steps=8, batch_size=8)
    state = agent.init(jax.random.PRNGKey(0), agent.context(8)[1])
    actor, critic = state.actor_state, state.critic_state
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4, eps=1e-5))
    params = start = (actor.params, critic.params)
    opt_state, norms = tx.init(params), []
    for i, scale in enumerate((1e-2, 1e-4, 3e-3)):
        leaves, tree = jax.tree.flatten(params)
        keys = jax.random.split(jax.random.PRNGKey(i), len(leaves))
        noise = [
            scale * jax.random.normal(k, p.shape)
            for k, p in zip(keys, leaves, strict=False)
        ]
        actor_grads, critic_grads = jax.tree.unflatten(tree, noise)
        grads = (actor_grads, jax.tree.map(lambda g: 3.0 * g, critic_grads))
        norms.append(float(optax.global_norm(grads)))
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        actor, critic = steps.joint_step(actor, critic, *grads, max_grad_norm=0.5)
    assert min(norms) < 0.5 < max(norms), norms
    got = (actor.params, critic.params)
    moved = jax.tree.map(lambda p, s: p - s, got, start)
    expected = jax.tree.map(lambda p, s: p - s, params, start)
    for m, e in zip(jax.tree.leaves(moved), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(m, e, rtol=1e-4, atol=1e-8)
