"""APO's optimiser is its official code's (``xtma/apo``, ``apo/algos/apg/``):
one Adam over the whole model (``base.py:28``) down the clipped surrogate
plus ``vf_coef`` times the value loss minus ``ent_coef`` times the entropy
(``appo.py:145-158``), the gradient clipped to global norm ``max_grad_norm``
over every parameter at once (``appo.py:106-112``; 10 and 3e-4 in
``apo/experiments/configs/mujoco/apg/mujoco_appo.py:5, :8``)."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ajax.agents.APO import train_APO
from ajax.agents.APO.APO import APO
from ajax.agents.APO.state import APOConfig
from ajax.environments.interaction import get_pi
from ajax.extensions.base import ExtensionStack
from ajax.networks.networks import predict_value

B, VALUE_BIAS = 16, 0.3


def test_APO_defaults_to_its_reference_codes_optimiser() -> None:
    agent = APO(env_id="Pendulum-v1")
    config = cast(APOConfig, agent.agent_config)
    assert (config.max_grad_norm, config.vf_coef, config.ent_coef) == (10.0, 1.0, 0)
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.learning_rate, args.eps, args.max_grad_norm) == (3e-4, 1e-8, None)


def test_clip_grad_norm_scales_as_torchs_clip_grad_norm() -> None:
    """torch 1.5.1's ``clip_grad_norm_`` multiplies every gradient by
    ``max_norm / (norm + 1e-6)`` when that is below 1: at norm 2e-6 and
    max_norm 1e-6, by a third."""
    grads = ({"a": jnp.array([1.2e-6, 0.0])}, {"b": jnp.array([1.6e-6])})
    clipped = train_APO.clip_grad_norm(grads, 1e-6)
    np.testing.assert_allclose(clipped[0]["a"], [0.4e-6, 0.0], rtol=1e-5)
    np.testing.assert_allclose(clipped[1]["b"], [1.6e-6 / 3], rtol=1e-5)
    kept = train_APO.clip_grad_norm(grads, 1e-5)
    for k, g in zip(jax.tree.leaves(kept), jax.tree.leaves(grads), strict=True):
        np.testing.assert_array_equal(k, g)
    assert train_APO.clip_grad_norm(grads, None) is grads


def _minibatches(state: Any) -> list[dict]:
    """Three minibatches whose value targets differ in scale (x100, x1,
    x10), so the clip binds on some steps and not on others."""
    out = []
    for i, scale in enumerate((100.0, 1.0, 10.0)):
        k = jax.random.split(jax.random.PRNGKey(i), 4)
        obs = jax.random.normal(k[0], (B, 3))
        actions = jax.random.uniform(k[1], (B, 1), minval=-1.0, maxval=1.0)
        pi, _ = get_pi(state.actor_state, state.actor_state.params, obs)
        out.append(
            {
                "obs": obs,
                "raw_obs": obs,
                "actions": actions,
                "raw_actions": actions,
                "log_probs": pi.log_prob(actions).sum(-1, keepdims=True),
                "gae": jax.random.normal(k[2], (B, 1)),
                "targets": scale * jax.random.normal(k[3], (B, 1)),
                "entropy_key": k[0],
                "extension_key": k[1],
            }
        )
    return out


@pytest.mark.parametrize(
    "options", [{}, {"vf_coef": 0.5, "max_grad_norm": 2.0, "ent_coef": 0.1}]
)
def test_APO_steps_its_whole_model_with_one_clipped_adam(options: dict) -> None:
    """Three of APO's steps equal three steps of one ``optax`` Adam over
    the actor's and the critic's parameters together, clipped to their
    joint norm, down the summed loss computed by hand. Clipping each
    network on its own, or dropping ``vf_coef``, moves the parameters."""
    agent = APO(env_id="Pendulum-v1", n_envs=2, n_steps=8, batch_size=8, **options)
    config = cast(APOConfig, agent.agent_config)
    args = (agent.actor_optimizer_args, agent.critic_optimizer_args)
    state = train_APO.init_APO(
        jax.random.PRNGKey(0), agent.env_args, *args, agent.network_args
    )
    actor, critic = state.actor_state, state.critic_state
    clip = cast(float, config.max_grad_norm)

    def loss(params: tuple, mb: dict) -> jax.Array:
        policy, _ = train_APO.policy_loss_function(
            params[0],
            actor,
            mb["obs"],
            mb["actions"],
            mb["log_probs"],
            mb["gae"],
            clip_coef=config.clip_range,
            ent_coef=config.ent_coef,
            entropy_key=mb["entropy_key"],
        )
        v = predict_value(critic, params[1], mb["obs"]).squeeze(0)
        target = mb["targets"] - config.nu * VALUE_BIAS
        return policy + config.vf_coef * 0.5 * jnp.mean((v - target) ** 2)

    tx = optax.chain(
        optax.clip_by_global_norm(clip),
        optax.adam(3e-4, eps=1e-8),
    )
    params = start = (actor.params, critic.params)
    opt_state, norms = tx.init(params), []
    for mb in _minibatches(state):
        grads = jax.grad(loss)(params, mb)
        norms.append(float(optax.global_norm(grads)))
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        state, _ = train_APO.update_networks(
            state, mb, config, jnp.float32(VALUE_BIAS), ExtensionStack(()), 1
        )
    assert min(norms) < clip < max(norms), norms  # the clip binds on some steps
    got = (state.actor_state.params, state.critic_state.params)
    moved = jax.tree.map(lambda p, s: p - s, got, start)
    expected = jax.tree.map(lambda p, s: p - s, params, start)
    for m, e in zip(jax.tree.leaves(moved), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(m, e, rtol=1e-4, atol=1e-7)
