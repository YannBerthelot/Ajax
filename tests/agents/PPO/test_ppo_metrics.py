"""PPO's and APO's diagnostics: each logged metric equals its definition,
recomputed here from the same networks and batch; a frozen actor's ratio is 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ajax.agents import minibatch
from ajax.agents.APO import train_APO
from ajax.agents.APO.APO import APO
from ajax.agents.diagnostics import approximate_kl, explained_variance, mean_policy_std
from ajax.agents.PPO import train_PPO
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PPO.utils import _compute_gae
from ajax.environments.interaction import get_pi
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import Extension, ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.distributions import Squash, recompute_log_prob
from ajax.networks.networks import predict_value
from ajax.schedule import at
from tests.environments import paid_action

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
    np.testing.assert_allclose(approximate_kl(log_ratio), kl, rtol=1e-6)
    assert float(approximate_kl(jnp.zeros(3))) == 0.0

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

    _, (p_aux, _) = train_PPO.policy_loss_function(
        actor.params, actor, obs, actions, log_probs, gae, False, 0.2, 0.0, True
    )
    pi, _ = get_pi(actor, actor.params, obs)
    log_ratio = np.asarray(recompute_log_prob(pi, actions, None) - log_probs)
    np.testing.assert_allclose(
        p_aux.approx_kl, np.mean(np.expm1(log_ratio) - log_ratio), rtol=1e-5
    )
    np.testing.assert_allclose(p_aux.policy_std, np.mean(pi.stddev()), rtol=1e-6)

    _, (v_aux, _) = train_PPO.value_loss_function(
        critic.params, critic, obs, targets, False
    )
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
        False,
        ExtensionStack(()),
        8,
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
    actions, log_probs = minibatch.rollout_actions(rollout)

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
    """APO steps both networks at once (train_APO.update_networks); each
    network's logged gradient norm is that of its own loss term, taken
    before the joint clip, as PPO logs them."""
    agent = APO(
        "Pendulum-v1", n_envs=1, actor_architecture=NET, critic_architecture=NET
    )
    config = agent.agent_config
    state = _apo_state(agent).replace(average_reward=jnp.asarray(0.3))
    obs, actions, log_probs, gae, targets = _batch(agent)
    stack, b = ExtensionStack(()), jnp.asarray(0.7)
    minibatch = {
        "obs": obs,
        "actions": actions,
        "log_probs": log_probs,
        "gae": gae,
        "targets": targets,
        "raw_actions": None,  # APO's reference head is unsquashed
        "raw_obs": None,
        "a_expert": None,
        "entropy_key": KEY,
    }
    _, aux = train_APO.update_networks(state, minibatch, config, b, stack, 1)
    v_aux, p_aux = cast(train_APO.ValueAuxiliaries, aux.value), aux.policy

    critic = state.critic_state
    v_grads = jax.grad(
        lambda p: train_APO.value_loss_function(
            p, critic, obs, targets, config.nu, b, config.vf_coef
        )[0]
    )(critic.params)
    np.testing.assert_allclose(v_aux.grad_norm, optax.global_norm(v_grads), rtol=1e-6)
    assert float(v_aux.value_bias) == np.float32(0.7)
    assert float(v_aux.average_reward) == np.float32(0.3)
    v = np.asarray(predict_value(critic, critic.params, obs).squeeze(0))
    t = np.asarray(targets)
    np.testing.assert_allclose(
        v_aux.explained_variance, 1 - np.var(t - v) / np.var(t), rtol=1e-5
    )

    actor = state.actor_state
    clip = at(config.clip_range, state.collector_state.timestep)
    p_grads = jax.grad(
        lambda p: train_APO.policy_loss_function(
            p, actor, obs, actions, log_probs, gae, clip, config.ent_coef, None, KEY
        )[0]
    )(actor.params)
    np.testing.assert_allclose(p_aux.grad_norm, optax.global_norm(p_grads), rtol=1e-6)
    pi, _ = get_pi(actor, actor.params, obs)
    log_ratio = np.asarray(recompute_log_prob(pi, actions, None) - log_probs)
    np.testing.assert_allclose(
        p_aux.approx_kl, np.mean(np.expm1(log_ratio) - log_ratio), rtol=1e-5
    )
    np.testing.assert_allclose(p_aux.policy_std, np.mean(pi.stddev()), rtol=1e-6)


@dataclass(frozen=True)
class _ZeroActorLoss(Extension):
    """An actor-loss term of 0: PPO's update then takes its extension path."""

    name: str = "zero_actor_loss"

    def actor_loss(self, agent_state: Any, ext_state: Any, batch: Any, ctx: Any) -> Any:
        return 0.0


WIDE = {"squash": True, "log_std_init": 1.0}  # std 2.7: tanh saturates


@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (PPO, WIDE),
        (PPO, WIDE | {"extensions": (_ZeroActorLoss(),)}),
        (APO, {"normalize_observations": True}),
    ],
    ids=["PPO-squash", "PPO-extension", "APO-normalized"],
)
def test_a_frozen_actor_recomputes_its_rollouts_log_probs(
    cls: Any, kwargs: dict
) -> None:
    """With the actor's learning rate at 0, each update's policy is the one
    that acted, so it recomputes every stored log-probability exactly: from
    the stored pre-tanh sample (through arctanh a saturated tanh is off),
    summed over the 2 action dimensions, on the observations the actor saw
    (normalised). The ratio is 1 and nothing clips at clip_range 1e-4
    (tests/probing/test_bookkeeping's Q8 cells)."""
    env, params = paid_action.make((-1.0, 1.0), length=8, dim=2)
    agent = cls(
        env,
        env_params=params,
        n_envs=1,
        n_steps=32,
        batch_size=32,
        n_epochs=2,
        actor_learning_rate=0.0,
        clip_range=1e-4,
        actor_architecture=NET,
        critic_architecture=NET,
        **kwargs,
    )
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    _, out = agent.train(seed=[0], n_timesteps=128, logging_config=logging)
    clip = np.asarray(out["policy/clip_fraction"])
    assert clip.size >= 4 and (clip == 0).all(), clip


def test_the_entropy_bonus_lowers_the_policy_loss() -> None:
    """loss = surrogate - ent_coef * entropy (Schulman et al., 2017, eq. 9):
    ent_coef 0.1 takes a tenth of the entropy off the loss. The coefficient
    reaches the update from the constructor."""
    agent = PPO(
        "Pendulum-v1",
        n_envs=1,
        ent_coef=0.1,
        actor_architecture=NET,
        critic_architecture=NET,
    )
    assert agent.agent_config.ent_coef == 0.1
    state = train_PPO.init_PPO(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    actor = state.actor_state
    obs, actions, log_probs, gae, _ = _batch(agent)

    def loss(ent_coef: float) -> tuple:
        args = (obs, actions, log_probs, gae, False, 0.2, ent_coef, True)
        total, (aux, _) = train_PPO.policy_loss_function(actor.params, actor, *args)
        return float(total), float(aux.entropy)

    (with_bonus, entropy), (without, _) = loss(0.1), loss(0.0)
    np.testing.assert_allclose(with_bonus - without, -0.1 * entropy, rtol=1e-5)
