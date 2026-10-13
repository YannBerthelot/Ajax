import jax
import jax.numpy as jnp
import numpy as np

from ajax.agents.recurrent import actor_dist, q_values
from ajax.agents.TD3.networks import get_initialized_td3_actor_critic
from ajax.agents.TD3.TD3 import TD3
from ajax.agents.TD3.train_TD3 import policy_loss_function
from ajax.types import NormalizationInfo


def _nets():
    agent = TD3(env_id="Pendulum-v1")
    return get_initialized_td3_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )


def test_td3_networks_are_plain_mlps():
    """Fujimoto's actor and critic are plain MLPs (sfujim/TD3, TD3.py): only
    Dense kernels and biases, no LayerNorm on the critic's encoder output."""
    for net in _nets():
        paths = [
            [getattr(k, "key", k) for k in path]
            for path, _ in jax.tree_util.tree_flatten_with_path(net.params)[0]
        ]
        assert {path[-1] for path in paths} == {"kernel", "bias"}, paths
        assert not any("norm" in str(k).lower() for p in paths for k in p), paths


def test_the_actor_loss_runs_q1_alone_as_the_ensemble_reads_it():
    """The feedforward actor loss runs the first critic alone, through
    predict_value's observation statistics: its loss and actor gradient are
    the full ensemble's Q1 ones up to float32 rounding."""
    actor, critic = _nets()
    stats = NormalizationInfo(
        var=jnp.full(3, 4.0), count=jnp.ones(1), mean=jnp.ones(3), mean_2=None
    )
    critic = critic.replace(obs_norm_info=stats)
    obs = jax.random.normal(jax.random.PRNGKey(1), (16, 3))

    def ensemble_q1(params):
        actions = actor_dist(actor, params, obs).mean()
        return -q_values(critic, critic.params, obs, actions)[0].mean()

    def q1_alone(params):
        return policy_loss_function(params, actor, critic, obs)[0]

    expected = jax.value_and_grad(ensemble_q1)(actor.params)
    got = jax.value_and_grad(q1_alone)(actor.params)
    for e, g in zip(jax.tree.leaves(expected), jax.tree.leaves(got), strict=True):
        np.testing.assert_allclose(g, e, rtol=1e-5, atol=1e-6)

    # Only Q1 runs: no matmul of the actor's gradient reads the whole
    # ensemble's kernels, which the ensemble's Q1 does.
    ensemble_kernels = {k.shape for k in jax.tree.leaves(critic.params) if k.ndim == 3}

    def matmul_operands(loss):
        jaxpr = jax.make_jaxpr(jax.grad(loss))(actor.params).jaxpr
        dots = [e for e in jaxpr.eqns if e.primitive.name == "dot_general"]
        return {v.aval.shape for e in dots for v in e.invars}

    assert ensemble_kernels <= matmul_operands(ensemble_q1)
    assert not ensemble_kernels & matmul_operands(q1_alone)
