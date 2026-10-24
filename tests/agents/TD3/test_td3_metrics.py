"""TD3's and SAC's diagnostics on a fixed batch: the twin critics' gap,
the share of target noise the clip changes and each network's gradient
norm equal their definitions."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ajax.agents.critics import critic_loss_fn
from ajax.agents.families import TrainContext
from ajax.agents.recurrent import q_values
from ajax.agents.SAC import core as sac_core
from ajax.agents.SAC import train_SAC
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3, compute_td3_td_target, policy_loss_function
from ajax.environments.interaction import init_collector_state
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import ExtensionStack
from ajax.state import Transition

KEY = jax.random.PRNGKey(0)
NET = ("16", "relu")
BATCH = 8
EMPTY = ExtensionStack(())


def _batch(agent) -> Transition:
    obs_shape, act_shape = get_state_action_shapes(agent.env_args.env)
    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    obs = jax.random.normal(keys[0], (BATCH, *obs_shape))
    return Transition(
        obs=obs,
        action=jnp.tanh(jax.random.normal(keys[1], (BATCH, *act_shape))),
        reward=jax.random.normal(keys[2], (BATCH, 1)),
        terminated=jnp.zeros((BATCH, 1)),
        truncated=jnp.zeros((BATCH, 1)),
        next_obs=jax.random.normal(keys[3], (BATCH, *obs_shape)),
    )


def context(agent: TD3, stack: ExtensionStack = EMPTY) -> TrainContext:
    """What TD3's init and update read, outside a training run."""
    env_args = agent.env_args
    return TrainContext(
        env_args=env_args,
        network_args=agent.network_args,
        actor_optimizer_args=agent.actor_optimizer_args,
        critic_optimizer_args=agent.critic_optimizer_args,
        extensions=stack,
        total_timesteps=1,
        collector=partial(
            init_collector_state,
            env_args=env_args,
            mode=env_args.mode,
            buffer=agent.buffer,
        ),
        buffer=agent.buffer,
    )


def test_td3_steps_report_the_gap_the_noise_clip_and_gradient_norms() -> None:
    agent = TD3("Pendulum-v1", actor_architecture=NET, critic_architecture=NET)
    ctx = context(agent)
    state = agent.init(KEY, ctx)
    batch, config = _batch(agent), agent.agent_config
    _, v_aux, _ = agent.critic_update(state, batch, None, ctx)

    critic = state.critic_state
    q = q_values(critic, critic.params, batch.obs, batch.action)
    np.testing.assert_allclose(v_aux.q_gap, np.mean(np.abs(q[0] - q[1])), rtol=1e-5)
    # The target noise is drawn on the first of the three keys the step splits.
    key = jax.random.split(state.rng, 3)[0]
    target, clip_frac, _ = compute_td3_td_target(
        state.actor_state,
        critic,
        key,
        batch.next_obs,
        batch.terminated,
        batch.reward,
        config.gamma,
        config.target_policy_noise,
        config.target_noise_clip,
        config.reward_scale,
    )
    assert float(v_aux.target_noise_clip_frac) == float(clip_frac)
    # TD3 sums its critics' squared errors (sfujim/TD3); critic_loss_fn averages them.
    v_grads = jax.grad(
        lambda p: config.num_critics
        * critic_loss_fn(p, critic, batch.obs, batch.action, target)[0]
    )(critic.params)
    np.testing.assert_allclose(v_aux.grad_norm, optax.global_norm(v_grads), rtol=1e-6)

    _, (p_aux, _) = agent.actor_update(state, batch, None, ctx)
    actor = state.actor_state
    p_grads = jax.grad(lambda p: policy_loss_function(p, actor, critic, batch.obs)[0])(
        actor.params
    )
    np.testing.assert_allclose(p_aux.grad_norm, optax.global_norm(p_grads), rtol=1e-6)


def test_the_clip_share_counts_the_noise_samples_the_clip_changes() -> None:
    agent = TD3("Pendulum-v1", actor_architecture=NET, critic_architecture=NET)
    state = agent.init(KEY, context(agent))
    batch = _batch(agent)

    def clip_frac(sigma: float, c: float) -> float:
        return float(
            compute_td3_td_target(
                state.actor_state,
                state.critic_state,
                KEY,
                batch.next_obs,
                batch.terminated,
                batch.reward,
                0.99,
                sigma,
                c,
                1.0,
            )[1]
        )

    noise = jax.random.normal(KEY, batch.action.shape)
    assert clip_frac(1.0, 0.5) == float((jnp.abs(noise) > 0.5).mean())
    assert clip_frac(0.0, 0.5) == 0.0


def test_sac_steps_report_each_networks_gradient_norm() -> None:
    agent = SAC("Pendulum-v1", actor_architecture=NET, critic_architecture=NET)
    state = train_SAC.init_SAC(
        KEY,
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
        agent.buffer,
    )
    batch = _batch(agent)
    _, v_aux, _ = train_SAC.update_value_functions(state, batch, False, 0.99, EMPTY)
    critic, actor = state.critic_state, state.actor_state
    alpha = jnp.exp(state.alpha.params["log_alpha"])
    target = sac_core.compute_td_target(
        actor,
        critic,
        batch.next_obs,
        batch.terminated,
        batch.reward,
        0.99,
        alpha,
        jax.random.split(state.rng, 3)[0],  # the step's target key
        False,
    )
    v_grads = jax.grad(
        lambda p: critic_loss_fn(p, critic, batch.obs, batch.action, target)[0]
    )(critic.params)
    np.testing.assert_allclose(v_aux.grad_norm, optax.global_norm(v_grads), rtol=1e-6)

    _, p_aux, *_ = train_SAC.update_policy(state, batch.obs, None, EMPTY)
    policy_key = state.draw_keys(2)[1]  # the step's policy key
    p_grads = jax.grad(
        lambda p: sac_core.soft_policy_loss(
            p, actor, critic, batch.obs, alpha, policy_key
        )[0]
    )(actor.params)
    np.testing.assert_allclose(p_aux.grad_norm, optax.global_norm(p_grads), rtol=1e-6)
