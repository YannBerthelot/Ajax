"""PPO's optimiser is ppo2's (openai/baselines @ea25b9e): one TensorFlow 1 Adam
(tf.train.AdamOptimizer) with eps 1e-5 (``ppo2/model.py:100``) down the policy
loss plus ``vf_coef`` 0.5 times
half the squared value error (``ppo2.py:22``, ``model.py:75``, ``:91``), its
gradient clipped to global norm 0.5 over every parameter at once
(``model.py:105-107``). Recurrent PPO and unroll_length take optax's Adam, as
their references (PureJaxRL, brax) do."""

from __future__ import annotations

import jax
import numpy as np
import optax

from ajax.agents import update_steps
from ajax.agents.PPO.PPO import PPO
from ajax.networks.memory import MemoryConfig
from ajax.networks.utils import scale_by_tf1_adam


def test_PPO_defaults_to_ppo2s_optimiser() -> None:
    agent = PPO(env_id="Pendulum-v1")
    assert (agent.max_grad_norm, agent.vf_coef) == (0.5, 0.5)
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.learning_rate, args.eps, args.max_grad_norm) == (3e-4, 1e-5, None)
        assert args.optimizer == "tf1_adam"


def test_recurrent_and_brax_minibatches_take_optaxs_adam() -> None:
    recurrent = PPO(env_id="CartPole-v1", memory=MemoryConfig("gru", 8))
    brax = PPO(env_id="Pendulum-v1", unroll_length=8, num_minibatches=4)
    for agent in (recurrent, brax):
        assert agent.actor_optimizer_args.optimizer == "adam"
        assert agent.critic_optimizer_args.optimizer == "adam"


def test_the_clipped_together_step_is_one_clipped_adam_over_both_networks() -> None:
    """Three steps of PPO's own actor and critic, their gradients clipped
    together, equal three steps of one TensorFlow 1 Adam (eps 1e-5) over both
    parameter trees together, clipped to their joint norm 0.5. The gradients'
    scales (norms ~4, ~0.04, ~1.2) make the clip bind on some steps and not
    others, and the actor's and critic's norms differ, so clipping each
    network on its own would move them differently. Each network's norm
    returned is that of its gradients before the clip."""
    agent = PPO(env_id="Pendulum-v1", n_envs=2, n_steps=8, batch_size=8)
    state = agent.init(jax.random.PRNGKey(0), agent.make_context(8)[1])
    actor, critic = state.actor_state, state.critic_state
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        scale_by_tf1_adam(0.9, 0.999, 1e-5),
        optax.scale(-3e-4),
    )
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
        actor, critic, actor_norm, critic_norm = (
            update_steps.apply_gradients_clipped_together(
                actor, critic, *grads, max_grad_norm=0.5
            )
        )
        np.testing.assert_allclose(actor_norm, optax.global_norm(grads[0]), rtol=1e-6)
        np.testing.assert_allclose(critic_norm, optax.global_norm(grads[1]), rtol=1e-6)
    assert min(norms) < 0.5 < max(norms), norms
    got = (actor.params, critic.params)
    moved = jax.tree.map(lambda p, s: p - s, got, start)
    expected = jax.tree.map(lambda p, s: p - s, params, start)
    for m, e in zip(jax.tree.leaves(moved), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(m, e, rtol=1e-4, atol=1e-8)
