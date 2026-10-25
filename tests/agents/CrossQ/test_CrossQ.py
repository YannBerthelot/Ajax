"""CrossQ's own parts: the authors' settings, the critics' one training pass
over the rows and their next states, and the policy delay."""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ajax import CrossQ
from ajax.agents.CrossQ.CrossQ import make_batch_renorm_mlp
from ajax.agents.families import TrainContext
from ajax.agents.loop import TrainLoop
from ajax.state import Transition


def test_the_defaults_are_the_authors() -> None:
    """adityab/CrossQ@a318a26, train.py -algo crossq."""
    agent = CrossQ("Pendulum-v1")
    assert agent.network_args.actor_architecture == make_batch_renorm_mlp(256)
    assert agent.network_args.critic_architecture == make_batch_renorm_mlp(2048)
    for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
        assert (args.learning_rate, args.beta_1, args.eps) == (1e-3, 0.5, 1e-8)
    assert agent.alpha_args.learning_rate == 1e-3
    assert (agent.learning_starts, agent.policy_delay) == (5000, 3)
    assert agent.target_entropy == -1.0 and agent.num_critics == 2


def test_the_temperature_takes_the_learning_rate_unless_given_its_own() -> None:
    assert CrossQ("Pendulum-v1", learning_rate=3e-4).alpha_args.learning_rate == 3e-4
    agent = CrossQ("Pendulum-v1", alpha_learning_rate=1e-4)
    assert agent.alpha_args.learning_rate == 1e-4


def input_means(batch_stats: Any) -> jax.Array:
    """Each critic's running mean of its input, (obs, action)."""
    return batch_stats["ensemble"]["encoder"]["network"]["layers_0"]["mean"]


def test_the_critics_normalise_the_rows_and_their_next_states_together() -> None:
    """One critic step moves the input's running mean by 1 - 0.99 times the
    mean over the rows and their next states, not over the rows alone."""
    agent = CrossQ("Pendulum-v1", actor_width=8, critic_width=8)
    loop = TrainLoop.create(agent.env_args, 1, 1)
    context = TrainContext(  # what init and the update read: no extension
        env_args=agent.env_args,
        network_args=agent.network_args,
        actor_optimizer_args=agent.actor_optimizer_args,
        critic_optimizer_args=agent.critic_optimizer_args,
        extensions=agent.extension_stack,
        total_timesteps=1,
        make_collector=partial(
            agent.make_collector, loop=loop, network_args=agent.network_args
        ),
        buffer=agent.buffer,
    )
    state = agent.init(jax.random.PRNGKey(0), context)
    assert state.critic_state.target_params is None
    obs = jax.random.normal(jax.random.PRNGKey(1), (8, 3))
    column = jnp.ones((8, 1))
    batch = Transition(obs, 0.5 * column, column, 0 * column, 0 * column, obs + 4.0)
    state, _, _ = agent.update_critic(state, batch, context)
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
