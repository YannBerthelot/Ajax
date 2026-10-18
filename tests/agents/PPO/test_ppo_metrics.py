"""PPO's and APO's diagnostics: each logged metric equals its definition,
recomputed here from the same networks and batch."""

from __future__ import annotations

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ajax.agents.APO import train_APO
from ajax.agents.APO.APO import APO
from ajax.agents.PPO import train_PPO
from ajax.agents.PPO.core import (
    approximate_kl,
    explained_variance,
    mean_policy_std,
    recompute_log_prob,
)
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PPO.utils import _compute_gae
from ajax.environments.interaction import get_pi
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import ExtensionStack
from ajax.networks.distributions import Squash
from ajax.networks.networks import predict_value

KEY = jax.random.PRNGKey(0)
NET = ("16", "relu")
BATCH = 8


def _batch(agent) -> tuple[jax.Array, ...]:
    """Observations, actions, stored log-probabilities, advantages and
    value targets, all drawn at random."""
    obs_shape, _ = get_state_action_shapes(agent.env_args.env)
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    obs = jax.random.normal(keys[0], (BATCH, *obs_shape))
    rest = (jax.random.normal(k, (BATCH, 1)) for k in keys[1:])
    return (obs, *rest)


def test_the_helpers_follow_their_formulas() -> None:
    log_ratio = jnp.array([0.1, -0.2, 0.0, 0.5])
    kl = np.mean(np.expm1(np.asarray(log_ratio)) - np.asarray(log_ratio))
    np.testing.assert_allclose(approximate_kl(jnp.exp(log_ratio), log_ratio), kl)
    assert float(approximate_kl(jnp.ones(3), jnp.zeros(3))) == 0.0

    targets = jnp.array([1.0, 2.0, 3.0, 4.0])
    # 1 - Var([0, 0, 0, -1]) / Var([1, 2, 3, 4]) = 1 - 0.1875 / 1.25
    preds = jnp.array([1.0, 2.0, 3.0, 5.0])
    np.testing.assert_allclose(explained_variance(preds, targets), 0.85, rtol=1e-6)
    assert float(explained_variance(targets, targets)) == 1.0
    assert np.isnan(explained_variance(preds, jnp.ones(4)))

    normal = distrax.Normal(jnp.zeros(2), jnp.array([1.0, 3.0]))
    assert float(mean_policy_std(normal)) == 2.0
    assert float(mean_policy_std(Squash(normal))) == 2.0  # the latent's
    assert np.isnan(mean_policy_std(distrax.Categorical(logits=jnp.zeros(3))))


def test_ppo_losses_report_kl_std_and_explained_variance() -> None:
    agent = PPO(
        "Pendulum-v1", n_envs=1, actor_architecture=NET, critic_architecture=NET
    )
    state = train_PPO.init_PPO(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    actor, critic = state.actor_state, state.critic_state
    obs, actions, log_probs, gae, targets = _batch(agent)

    _, p_aux = train_PPO.policy_loss_function(
        actor.params, actor, obs, actions, log_probs, gae, False, 0.2, 0.0, True
    )
    pi, _ = get_pi(actor, actor.params, obs)
    log_ratio = np.asarray(recompute_log_prob(pi, actions, None) - log_probs)
    np.testing.assert_allclose(
        p_aux.approx_kl, np.mean(np.expm1(log_ratio) - log_ratio), rtol=1e-5
    )
    np.testing.assert_allclose(p_aux.policy_std, np.mean(pi.stddev()), rtol=1e-6)

    _, v_aux = train_PPO.value_loss_function(critic.params, critic, obs, targets, False)
    v = np.asarray(predict_value(critic, critic.params, obs).squeeze(0))
    t = np.asarray(targets)
    np.testing.assert_allclose(
        v_aux.explained_variance, 1 - np.var(t - v) / np.var(t), rtol=1e-5
    )


def test_a_ppo_update_logs_each_networks_norm_before_the_clip() -> None:
    """One epoch of one minibatch: the logged norms are those of the two
    losses' gradients over the whole rollout, before the joint clip to 0.5."""
    agent = PPO(
        "Pendulum-v1",
        n_envs=2,
        n_steps=4,
        num_minibatches=1,
        n_epochs=1,
        actor_architecture=NET,
        critic_architecture=NET,
        expose_recent_rollout=True,
    )
    state, _ = agent.train(seed=0, n_timesteps=8)
    assert state.last_rollout is not None
    if state.last_rollout.obs.ndim == 4:  # a leading seed axis
        state = jax.tree.map(lambda x: x[0], state)
    rollout, config = state.last_rollout, agent.agent_config
    assert rollout is not None
    _, aux = train_PPO.update_agent(
        state,
        rollout,
        state,
        config,
        agent.env_args,
        agent.env_args.mode,
        False,
        ExtensionStack(()),
        8,
        2,
    )

    actor, critic = state.actor_state, state.critic_state
    values = predict_value(critic, critic.params, rollout.obs).squeeze(0)
    next_values = predict_value(critic, critic.params, rollout.next_obs).squeeze(0)
    gae, targets = _compute_gae(
        values=values,
        next_values=next_values,
        rewards=rollout.reward,
        terminateds=rollout.terminated,
        truncateds=rollout.truncated,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    actions, log_probs = train_PPO.rollout_actions(rollout)

    def value_loss(p):
        return train_PPO.value_loss_function(
            p, critic, rollout.obs, targets, False, vf_coef=config.vf_coef
        )[0]

    def policy_loss(p):
        return train_PPO.policy_loss_function(
            p,
            actor,
            rollout.obs,
            actions,
            log_probs,
            gae,
            False,
            config.clip_range,
            config.ent_coef,
            config.normalize_advantage,
            raw_actions=rollout.raw_action,
        )[0]

    v_norm = optax.global_norm(jax.grad(value_loss)(critic.params))
    p_norm = optax.global_norm(jax.grad(policy_loss)(actor.params))
    np.testing.assert_allclose(aux.value.grad_norm, v_norm, rtol=1e-4)
    np.testing.assert_allclose(aux.policy.grad_norm, p_norm, rtol=1e-4)
    assert max(float(v_norm), float(p_norm)) > 0.5  # the clip binds


def _apo_state(agent):
    return train_APO.init_APO(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )


def test_apo_steps_report_the_value_bias_rate_and_gradient_norms() -> None:
    agent = APO(
        "Pendulum-v1", n_envs=1, actor_architecture=NET, critic_architecture=NET
    )
    state = _apo_state(agent).replace(average_reward=jnp.asarray(0.3))
    obs, actions, log_probs, gae, targets = _batch(agent)
    stack, b, nu = ExtensionStack(()), jnp.asarray(0.7), 0.1

    _, v_aux = train_APO.update_value_functions(
        state, obs, targets, nu, b, stack, 1, KEY
    )
    critic = state.critic_state
    v_grads = jax.grad(
        lambda p: train_APO.value_loss_function(p, critic, obs, targets, nu, b)[0]
    )(critic.params)
    np.testing.assert_allclose(v_aux.grad_norm, optax.global_norm(v_grads), rtol=1e-6)
    assert float(v_aux.value_bias) == np.float32(0.7)
    assert float(v_aux.average_reward) == np.float32(0.3)
    v = np.asarray(predict_value(critic, critic.params, obs).squeeze(0))
    t = np.asarray(targets)
    np.testing.assert_allclose(
        v_aux.explained_variance, 1 - np.var(t - v) / np.var(t), rtol=1e-5
    )

    # APO's policy is squashed: the actions are pre-tanh samples.
    _, p_aux = train_APO.update_policy(
        state,
        obs,
        actions,
        gae,
        log_probs,
        0.2,
        0.0,
        stack,
        1,
        KEY,
        raw_actions=actions,
        entropy_key=KEY,
    )
    actor = state.actor_state
    p_grads = jax.grad(
        lambda p: train_APO.policy_loss_function(
            p, actor, obs, actions, log_probs, gae, 0.2, 0.0, actions, KEY
        )[0]
    )(actor.params)
    np.testing.assert_allclose(p_aux.grad_norm, optax.global_norm(p_grads), rtol=1e-6)
    pi, _ = get_pi(actor, actor.params, obs)
    log_ratio = np.asarray(recompute_log_prob(pi, actions, actions) - log_probs)
    np.testing.assert_allclose(
        p_aux.approx_kl, np.mean(np.expm1(log_ratio) - log_ratio), rtol=1e-5
    )
    np.testing.assert_allclose(
        p_aux.policy_std, np.mean(pi.distribution.stddev()), rtol=1e-6
    )
