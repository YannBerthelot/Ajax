"""CrossQ's own parts: the authors' settings, the critics' one training pass
over the rows and their next states, and the policy delay."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ajax import CrossQ
from ajax.agents.CrossQ.CrossQ import batch_renorm_mlp
from ajax.agents.CrossQ.train_CrossQ import init_CrossQ, update_critics
from ajax.extensions.base import ExtensionStack
from ajax.state import Transition


def test_the_defaults_are_the_authors() -> None:
    """adityab/CrossQ@a318a26, train.py -algo crossq."""
    agent = CrossQ("Pendulum-v1")
    assert agent.network_args.actor_architecture == batch_renorm_mlp(256)
    assert agent.network_args.critic_architecture == batch_renorm_mlp(2048)
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.learning_rate, args.beta_1, args.eps) == (1e-3, 0.5, 1e-8)
    assert agent.alpha_args.learning_rate == 1e-3
    config = agent.agent_config
    assert (config.learning_starts, config.policy_delay) == (5000, 3)
    assert config.target_entropy == -1.0 and agent.train_kwargs["num_critics"] == 2


def input_means(batch_stats: Any) -> jax.Array:
    """Each critic's running mean of its input, (obs, action)."""
    return batch_stats["ensemble"]["encoder"]["network"]["layers_0"]["mean"]


def test_the_critics_normalise_the_rows_and_their_next_states_together() -> None:
    """One critic step moves the input's running mean by 1 - 0.99 times the
    mean over the rows and their next states, not over the rows alone."""
    agent = CrossQ("Pendulum-v1", actor_width=8, critic_width=8)
    state = init_CrossQ(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
        agent.alpha_args,
        agent.buffer,
        num_critics=2,
    )
    assert state.critic_state.target_params is None
    obs = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    column = jnp.ones((8, 1))
    batch = Transition(obs, 0.5 * column, column, 0 * column, 0 * column, obs + 4.0)
    state, _, _ = update_critics(
        state, batch, agent.agent_config, ExtensionStack(()), 1
    )
    both = jnp.concatenate([obs, obs + 4.0]).mean(0)
    np.testing.assert_allclose(
        input_means(state.critic_state.batch_stats)[:, :3],
        jnp.tile(0.01 * both, (2, 1)),
        rtol=1e-5,
    )


def steps(batch_stats: Any) -> set:
    """The pass counts of a network's Batch Renorm layers."""
    return {int(x.max()) for x in jax.tree.leaves(batch_stats) if x.dtype.kind == "i"}


def test_the_actor_trains_on_every_policy_delay_th_update() -> None:
    """Updates from the 8th collected step to the 64th: 57 critic passes, 19
    actor ones (the 3rd, 6th, ..., 57th update), each layer counting its own."""
    agent = CrossQ(
        "Pendulum-v1",
        actor_width=8,
        critic_width=8,
        learning_starts=8,
        batch_size=8,
        buffer_size=128,
    )
    state, _ = agent.train(seed=0, n_timesteps=64)
    assert int(jnp.asarray(state.n_updates)[0]) == 57
    assert steps(state.critic_state.batch_stats) == {57}
    assert steps(state.actor_state.batch_stats) == {19}
