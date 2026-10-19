"""Smoke + key-set tests for ``ajax.extensions.instrumentation``.

End-to-end checks that each of the three EVarEst measurement /
instrumentation extensions:

* :class:`ConditioningMetrics`
* :class:`BiasVoreDecomposition`
* :class:`BiasVorePenalty`      — additive critic-loss penalty

wires into the Extension surface on PPO (on-policy, through
``agent_state.last_rollout``). BiasVorePenalty's coefficient and penalty
also have unit checks, and every logged metric of the two others meets
its closed form on a known rollout, critic and residual. Every agent's
critic lookup is checked in tests/agents/test_agent_contract.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import gymnax
import jax
import jax.numpy as jnp
import pytest

from ajax.agents.PPO.PPO import PPO
from ajax.extensions.instrumentation import (
    COND_METRIC_KEYS,
    EVAREST_DECOMP_KEYS,
    BiasVoreDecomposition,
    BiasVorePenalty,
    ConditioningMetrics,
    _sample_transitions,
    _takes_action,
    evarest_coeff,
)
from ajax.logging.wandb_logging import LoggingConfig
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import EnvironmentConfig, NetworkConfig, OptimizerConfig, Transition

# Evaluate (and return the evaluations) without a logging backend.
EVALUATE = LoggingConfig(config={}, log_frequency=64, use_wandb=False)

# --------------------------------------------------------------------------
# BiasVorePenalty: standalone unit checks (no agent in the loop)
# --------------------------------------------------------------------------


def _q_network_batch() -> dict:
    """A DQN/PQN critic-loss batch whose taken Q-values are (1, 5, 1)
    against zero targets: the residual's variance is 32/9."""
    q_values = jnp.array([[1.0, 2.0], [3.0, 5.0], [1.0, 0.0]])
    q_state = SimpleNamespace(
        params=None, apply_fn=lambda _, obs: SimpleNamespace(q_values=q_values)
    )
    return {
        "q_state": q_state,
        "observations": jnp.zeros((3, 1)),
        "actions": jnp.array([0, 1, 0]),
        "targets": jnp.zeros((3, 1)),
    }


def test_bias_vore_penalty_alpha_half_is_zero():
    """alpha=0.5 ⇒ coeff = 0 ⇒ no penalty, on a batch whose residual
    variance is not zero."""
    ext = BiasVorePenalty(alpha=0.5)
    assert ext._coeff() == 0.0
    assert float(ext.critic_loss(None, None, _q_network_batch(), None)) == 0.0


def test_bias_vore_penalty_scales_the_residual_variance():
    """The penalty is ``loss_scale * coeff * Var(q_taken - target)``."""
    ext = BiasVorePenalty(alpha=0.3, loss_scale=0.5)
    penalty, metrics = ext.critic_loss(None, None, _q_network_batch(), None)
    assert float(penalty) == pytest.approx(0.5 * evarest_coeff(0.3) * 32 / 9)
    assert metrics["bias_vore/penalty"] is penalty
    assert float(metrics["bias_vore/residual_var"]) == pytest.approx(32 / 9)


def test_bias_vore_penalty_coeff_matches_paper_formula():
    """coeff = (1 - 2 alpha) / alpha for alpha != 0.5."""
    for alpha in (0.1, 0.3, 0.7, 0.9):
        ext = BiasVorePenalty(alpha=alpha)
        assert ext._coeff() == pytest.approx(evarest_coeff(alpha))


def test_bias_vore_penalty_coeff_override_wins():
    """An explicit ``coeff_override`` takes precedence over alpha."""
    ext = BiasVorePenalty(alpha=0.1, coeff_override=0.0)
    assert ext._coeff() == 0.0
    ext = BiasVorePenalty(alpha=0.5, coeff_override=3.7)
    assert ext._coeff() == pytest.approx(3.7)


# --------------------------------------------------------------------------
# The measurement batch
# --------------------------------------------------------------------------


def test_a_rollout_batch_is_one_row_per_step():
    """A ``(T, n_envs, ...)`` rollout gives ``(T * n_envs, ...)`` rows, cut
    at ``on_policy_take``: srank and the dormant fraction read a 2-D
    feature matrix, not one matrix per time step. Either end of an episode
    stops a rollout's bootstrap."""
    obs = jnp.arange(4 * 2 * 3.0).reshape(4, 2, 3)
    flags = jnp.zeros((4, 2), dtype=bool)
    rollout = Transition(
        obs=obs,
        action=jnp.ones((4, 2, 1)),
        reward=jnp.ones((4, 2)),
        terminated=flags.at[2, 1].set(True),
        truncated=flags.at[1, 0].set(True),
        next_obs=obs + 1.0,
    )
    state = SimpleNamespace(collector_state=None, last_rollout=rollout)
    rows, actions, rewards, next_rows, done = _sample_transitions(
        state, jax.random.PRNGKey(0), None, 6
    )
    assert jnp.array_equal(rows, obs.reshape(8, 3)[:6])
    assert jnp.array_equal(next_rows, rows + 1.0)
    assert actions.shape == rewards.shape == (6, 1)
    assert done[:, 0].tolist() == [False, False, True, False, False, True]


# --------------------------------------------------------------------------
# On PPO, through the exposed rollout
# --------------------------------------------------------------------------


def _train_ppo_with_extensions(extensions, n_steps=32):
    agent = PPO(
        env_id="CartPole-v1",
        n_envs=2,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        n_steps=n_steps,
        batch_size=n_steps,
        n_epochs=1,
        expose_recent_rollout=True,
        extensions=extensions,
    )
    return agent.train(seed=42, n_timesteps=128, logging_config=EVALUATE)


def test_conditioning_and_decomposition_compose_on_ppo():
    """Both fold into PPO's eval_metrics phase together: the Cond/* and
    EVarEst/* keys are among the evaluations ``train`` returns."""
    out = _train_ppo_with_extensions(
        [
            ConditioningMetrics(on_policy_batch=16),
            BiasVoreDecomposition(on_policy_batch=16),
        ]
    )
    # ``train`` returns ``(state, metrics)``.
    assert isinstance(out, tuple) and len(out) == 2
    _, metrics = out
    for key in COND_METRIC_KEYS + EVAREST_DECOMP_KEYS:
        assert key in metrics, f"missing {key!r}; got {list(metrics)}"


def test_bias_vore_penalty_alpha_nonhalf_runs():
    """alpha=0.3 ⇒ non-zero penalty; PPO still trains end-to-end."""
    _, metrics = _train_ppo_with_extensions(
        [BiasVorePenalty(alpha=0.3, loss_scale=0.5)]
    )
    assert metrics is not None


# --------------------------------------------------------------------------
# Every metric against its closed form
# --------------------------------------------------------------------------
#: Live units' activation: under the dormancy threshold unless normalised
#: by the layer's mean.
LIVE = 0.1
#: The critic's features are the observations of a (T=4, n_envs=2) rollout,
#: rows LIVE (e_0, e_1, e_2, e_0) then zeros: rank 3 (each step's rows,
#: read apart, give four equal singular values), one dead unit of 4; its
#: parameters have norm 5; the residual (0.5, -3.5, ...) has mean -1.5 and
#: variance 4.
CLOSED_FORM = {
    "Cond/critic_srank": 3.0,
    "Cond/critic_dormant_frac": 0.25,
    "Cond/critic_feat_norm": 4 * LIVE / 8,
    "Cond/critic_weight_norm": 5.0,
    "EVarEst/bias_sq": 2.25,
    "EVarEst/var_resid": 4.0,
    "EVarEst/mse": 6.25,
    "EVarEst/abs_bias": 1.5,
}


def test_every_metric_has_a_closed_form():
    assert set(CLOSED_FORM) == set(COND_METRIC_KEYS + EVAREST_DECOMP_KEYS)


@pytest.mark.parametrize("key", COND_METRIC_KEYS + EVAREST_DECOMP_KEYS)
def test_a_metric_takes_its_closed_form(key):
    rows = jnp.zeros((8, 4)).at[jnp.arange(4), jnp.array([0, 1, 2, 0])].set(LIVE)
    obs, flags = rows.reshape(4, 2, 4), jnp.zeros((4, 2), dtype=bool)
    rollout = Transition(
        obs=obs,
        action=jnp.zeros((4, 2, 1)),
        reward=jnp.zeros((4, 2)),
        terminated=flags,
        truncated=flags,
        next_obs=obs,
    )
    critic = SimpleNamespace(
        params={"w": jnp.array([3.0, 4.0])}, apply_fn=lambda _, x, method: x[None]
    )
    state = SimpleNamespace(
        collector_state=None, last_rollout=rollout, critic_state=critic
    )
    residual = jnp.tile(jnp.array([0.5, -3.5]), 4)
    extensions = (
        ConditioningMetrics(critic_loc="critic", concat_action=False),
        BiasVoreDecomposition(value_target_fn=lambda *_: (jnp.zeros(8), residual)),
    )
    key0 = jax.random.PRNGKey(0)
    metrics = {
        k: v
        for e in extensions
        for k, v in e.eval_metrics(state, None, key0, None).items()
    }
    assert float(metrics[key]) == pytest.approx(CLOSED_FORM[key], rel=1e-6)


def test_a_cnn_critic_reads_the_observation_alone():
    """A CNN state-value critic refuses an input one column wide too (the
    action appended), where an MLP builds other shapes: the instrumentation
    reads it as a state-value critic all the same."""
    env, params = gymnax.make("CartPole-v1")
    env_config = EnvironmentConfig(env, params, n_envs=1, continuous=False)
    network = NetworkConfig(("8", "relu"), ("8", "relu"), cnn_image_shape=(2, 2, 1))
    adam = OptimizerConfig(learning_rate=1e-3)
    key = jax.random.PRNGKey(0)
    _, critic = get_initialized_actor_critic(
        key, env_config, adam, adam, network, continuous=False, num_critics=1
    )
    assert not _takes_action(critic, jnp.zeros((2, 4)), jnp.zeros((2, 1)), key)
