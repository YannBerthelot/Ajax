"""APO's value loss in its 0.4.0 form, train_APO.value_loss_function, kept
until 0.5.0: a call by position or by its old keywords (critic_states, b)
warns and returns what compute_value_loss returns."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.APO import train_APO
from ajax.agents.APO.APO import APO, compute_value_loss


def test_the_former_value_loss_returns_the_new_one() -> None:
    agent = APO("Pendulum-v1", n_envs=1, critic_architecture=("8", "tanh"))
    critic = agent.init(jax.random.PRNGKey(0), agent.make_context(1)[1]).critic_state
    obs = jax.random.normal(jax.random.PRNGKey(1), (3, 3))
    targets, valid = jnp.array([[1.0], [2.0], [3.0]]), jnp.array([[1.0], [1.0], [0.0]])
    new = compute_value_loss(critic.params, critic, obs, targets, 0.1, 0.5, 1.0, valid)
    match = "train_APO.value_loss_function is ajax.agents.APO.APO.compute_value_loss"
    with pytest.warns(FutureWarning, match=match):
        by_position = train_APO.value_loss_function(
            critic.params, critic, obs, targets, 0.1, 0.5, 1.0, valid
        )
    with pytest.warns(FutureWarning, match=match):
        by_keyword = train_APO.value_loss_function(
            critic.params,
            critic_states=critic,
            observations=obs,
            value_targets=targets,
            nu=0.1,
            b=0.5,
            vf_coef=1.0,
            valid=valid,
        )
    jax.tree.map(np.testing.assert_array_equal, by_position, new)
    jax.tree.map(np.testing.assert_array_equal, by_keyword, new)
