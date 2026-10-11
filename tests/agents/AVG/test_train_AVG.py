"""AVG-specific training tests.

Tests the unique AVG mechanics: the running-average value updates that
AVG uses instead of a target network (``update_AVG_values``), the order
of its actor and critic steps and the episode ends its target cuts. Shared behaviors (loss shapes, updates,
training loop, make_train) are covered by the probing suite and the smoke
test in ``test_AVG.py``.
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.AVG import train_AVG
from ajax.agents.AVG.AVG import AVG
from ajax.agents.AVG.state import AVGConfig
from ajax.agents.AVG.train_AVG import init_AVG, update_AVG_values
from ajax.environments.interaction import Transition
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.base import ExtensionStack
from ajax.state import AlphaConfig, NetworkConfig, OptimizerConfig


@pytest.fixture
def avg_state(env_config):
    return init_AVG(
        key=jax.random.PRNGKey(0),
        env_args=env_config,
        actor_optimizer_args=OptimizerConfig(learning_rate=3e-4),
        critic_optimizer_args=OptimizerConfig(learning_rate=3e-4),
        network_args=NetworkConfig(
            actor_architecture=["64", "relu", "64"],
            critic_architecture=["64", "relu", "64"],
            squash=True,
            penultimate_normalization=True,
        ),
        alpha_args=AlphaConfig(learning_rate=3e-4, alpha_init=1.0),
    )


@pytest.mark.parametrize(
    "terminated, return_folds, gamma_mean",
    [(0.0, False, 0.99), (1.0, True, 0.0)],
    ids=["running", "terminal"],
)
def test_update_AVG_values(env_config, avg_state, terminated, return_folds, gamma_mean):
    """Every step folds its entropy-regularised reward and its discount (0 at
    a termination) into their statistics; only an episode end folds the
    return."""
    observation_shape, action_shape = get_state_action_shapes(env_config.env)
    rollout = Transition(
        obs=jnp.ones((env_config.n_envs, *observation_shape)),
        action=jnp.ones((env_config.n_envs, *action_shape)),
        next_obs=jnp.ones((env_config.n_envs, *observation_shape)),
        reward=jnp.array([[1.0]]),
        terminated=jnp.array([[terminated]]),
        truncated=jnp.array([[0.0]]),
        log_prob=jnp.array([[-1.0]]),
    )
    agent_config = AVGConfig(gamma=0.99, target_entropy=-1.0)

    updated_state = update_AVG_values(avg_state, rollout, agent_config)
    alpha = jnp.exp(updated_state.alpha.params["log_alpha"])

    assert updated_state.reward.count[0] > avg_state.reward.count[0]
    assert updated_state.gamma.count[0] > avg_state.gamma.count[0]
    folded = updated_state.G_return.count[0] > avg_state.G_return.count[0]
    assert bool(folded) == return_folds
    assert jnp.allclose(updated_state.reward.mean, jnp.array([[1.0 - alpha * -1]]))
    assert jnp.allclose(updated_state.gamma.mean, jnp.array([[gamma_mean]]))


def _one_step() -> tuple[AVG, Any, Transition]:
    """A small AVG, its initial state and one running transition."""
    agent = AVG("Pendulum-v1", actor_architecture=("8", "relu"))
    state = init_AVG(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
    )
    obs_shape, action_shape = get_state_action_shapes(agent.env_args.env)
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    column = jnp.zeros((1, 1))
    raw = jax.random.normal(keys[1], (1, *action_shape))
    transition = Transition(
        obs=jax.random.normal(keys[0], (1, *obs_shape)),
        action=jnp.tanh(raw),
        reward=jnp.ones((1, 1)),
        terminated=column,
        truncated=column,
        next_obs=jax.random.normal(keys[2], (1, *obs_shape)),
        raw_action=raw,
    )
    return agent, state, transition


def test_actor_and_critic_step_from_the_same_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The official AVG code (``gauthamvasan/avg``, ``AVG.update``) builds
    both losses before either optimizer step: the actor's loss reads the
    critic before its step, the critic's TD target the actor before its."""
    agent, state, transition = _one_step()
    agent_config = cast(AVGConfig, agent.agent_config)
    read: dict[str, Any] = {}
    policy_loss, td_target = (
        train_AVG.policy_loss_function,
        train_AVG.compute_avg_td_target,
    )

    def spy_policy_loss(actor_params: Any, actor_state: Any, critic: Any, *a: Any):
        read["critic"] = critic.params
        return policy_loss(actor_params, actor_state, critic, *a)

    def spy_td_target(actor_state: Any, *a: Any):
        read["actor"] = actor_state.params
        return td_target(actor_state, *a)

    monkeypatch.setattr(train_AVG, "policy_loss_function", spy_policy_loss)
    monkeypatch.setattr(train_AVG, "compute_avg_td_target", spy_td_target)
    new_state, _ = train_AVG.update_agent(
        state, transition, agent_config, ExtensionStack(()), 1_000
    )

    for name, before, after in (
        ("critic", state.critic_state.params, new_state.critic_state.params),
        ("actor", state.actor_state.params, new_state.actor_state.params),
    ):
        assert jax.tree.all(jax.tree.map(np.array_equal, read[name], before)), name
        assert not jax.tree.all(jax.tree.map(np.array_equal, before, after)), name


def test_a_truncated_step_bootstraps_and_a_terminated_one_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The official AVG code passes only ``terminated`` as the update's done
    (``gauthamvasan/avg``, ``avg.py``): at a time limit the critic's target
    still bootstraps on the final observation; only a termination cuts it."""
    agent, state, transition = _one_step()
    agent_config = cast(AVGConfig, agent.agent_config)
    targets: list[jax.Array] = []
    td_target = train_AVG.compute_avg_td_target

    def spy_td_target(*a: Any):
        target, next_log_probs = td_target(*a)
        targets.append(target)
        return target, next_log_probs

    monkeypatch.setattr(train_AVG, "compute_avg_td_target", spy_td_target)
    one, zero = jnp.ones((1, 1)), jnp.zeros((1, 1))
    for terminated, truncated in ((zero, zero), (zero, one), (one, zero)):
        step = transition.replace(terminated=terminated, truncated=truncated)
        train_AVG.update_value_functions(
            state, step, agent_config, ExtensionStack(()), 1_000
        )

    running, truncated, terminated = targets
    reward = transition.reward * agent_config.reward_scale
    np.testing.assert_array_equal(truncated, running)
    assert not np.allclose(running, reward)
    np.testing.assert_allclose(terminated, reward)
