"""APO's optimiser is its official code's (``xtma/apo``, ``apo/algos/apg/``):
one Adam over the whole model (``base.py:28``) down the clipped surrogate
plus ``vf_coef`` times the value loss minus ``ent_coef`` times the entropy
(``appo.py:145-158``), the gradient clipped to global norm ``max_grad_norm``
over every parameter at once (``appo.py:106-112``; 3e-4, 10 and the entropy
coefficient 0.01 in ``apo/experiments/configs/mujoco/apg/mujoco_appo.py:5,
:8, :9``)."""

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
from ajax.agents.PPO import train_PPO
from ajax.agents.PPO.core import recompute_log_prob
from ajax.environments.interaction import get_pi
from ajax.extensions.base import ExtensionStack
from ajax.networks.networks import predict_value
from tests.environments import paid_action

B, VALUE_BIAS = 16, 0.3


def test_APO_defaults_to_its_reference_codes_optimiser() -> None:
    agent = APO(env_id="Pendulum-v1")
    config = cast(APOConfig, agent.agent_config)
    assert (config.max_grad_norm, config.vf_coef, config.ent_coef) == (10.0, 1.0, 0.01)
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.learning_rate, args.eps, args.max_grad_norm) == (3e-4, 1e-8, None)


@pytest.mark.parametrize("env_id", ["Pendulum-v1", "paid-action-3d", "CartPole-v1"])
def test_APO_entropy_bonus_is_its_reference_codes(env_id: str) -> None:
    """At APO's defaults and zero advantages (a zero surrogate), its policy
    loss is the official code's entropy term (``appo.py:155-156``): minus
    0.01 times the batch mean of the entropy, the Gaussian's ``log_std +
    log sqrt(2 pi e)`` summed over the action dimensions
    (``rlpyt/distributions/gaussian.py:81-97``), so the term lowers each
    log-std's loss gradient by 0.01. On the three-dimensional box a mean
    over the dimensions would give a third of both."""
    if env_id == "paid-action-3d":
        env, env_params = paid_action.make(dim=3)
        agent = APO(env_id=env, env_params=env_params, n_envs=2)
    else:
        agent = APO(env_id=env_id, n_envs=2)
    config = cast(APOConfig, agent.agent_config)
    args = (agent.actor_optimizer_args, agent.critic_optimizer_args)
    key = jax.random.PRNGKey(0)
    actor = train_APO.init_APO(
        key, agent.env_args, *args, agent.network_args
    ).actor_state
    params: Any = actor.params
    continuous = "log_std_param" in params["params"]
    if continuous:  # a log-std away from its initial 0
        log_std = jnp.full_like(params["params"]["log_std_param"], 0.3)
        params = {"params": {**params["params"], "log_std_param": log_std}}
    obs = jax.random.normal(key, (B, 3 if continuous else 4))
    pi, _ = get_pi(actor, params, obs)
    actions = pi.sample(seed=key).reshape(B, -1)
    old = recompute_log_prob(pi, actions, actions)

    def loss(p: Any) -> jax.Array:
        return train_APO.policy_loss_function(
            p,
            actor,
            obs,
            actions,
            old,
            gae=jnp.zeros((B, 1)),
            clip_coef=train_APO.resolve_clip_coef(config, 0),
            ent_coef=config.ent_coef,
            raw_actions=actions,
        )[0]

    if continuous:
        action_dim = params["params"]["log_std_param"].shape[-1]
        assert action_dim == (3 if env_id == "paid-action-3d" else 1)
        entropy = action_dim * (0.3 + 0.5 * np.log(2 * np.pi * np.e))
    else:
        probs = np.asarray(pi.probs)
        entropy = float(np.mean(-np.sum(probs * np.log(probs), -1)))
    np.testing.assert_allclose(loss(params), -0.01 * entropy, rtol=1e-5)
    if continuous:
        grad = jax.grad(loss)(params)["params"]["log_std_param"]
        np.testing.assert_allclose(grad, -0.01, rtol=1e-5)


def test_APO_clips_with_torchs_clip_grad_norm() -> None:
    """torch 1.5.1's ``clip_grad_norm_`` multiplies every gradient by
    ``max_norm / (norm + 1e-6)`` when that is below 1: at norm 2e-6 and
    max_norm 1e-6, by a third (``tf.clip_by_global_norm``, PPO's, by a
    half)."""
    clip = train_PPO.clip_global_norm
    grads = ({"a": jnp.array([1.2e-6, 0.0])}, {"b": jnp.array([1.6e-6])})
    clipped = clip(grads, 1e-6, train_APO.TORCH_CLIP_EPS)
    np.testing.assert_allclose(clipped[0]["a"], [0.4e-6, 0.0], rtol=1e-5)
    np.testing.assert_allclose(clipped[1]["b"], [1.6e-6 / 3], rtol=1e-5)
    np.testing.assert_allclose(clip(grads, 1e-6)[1]["b"], [0.8e-6], rtol=1e-5)
    kept = clip(grads, 1e-5, train_APO.TORCH_CLIP_EPS)
    for k, g in zip(jax.tree.leaves(kept), jax.tree.leaves(grads), strict=True):
        np.testing.assert_array_equal(k, g)
    assert clip(grads, None) is grads


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
                "valid": jnp.ones((B, 1)),  # no time limit
                "gae": jax.random.normal(k[2], (B, 1)),
                "targets": scale * jax.random.normal(k[3], (B, 1)),
                "entropy_key": k[0],
                "extension_key": k[1],
                "a_expert": None,
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
            clip_coef=train_APO.resolve_clip_coef(config, 0),
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
