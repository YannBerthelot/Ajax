import jax

from ajax.agents.TD3.networks import get_initialized_td3_actor_critic
from ajax.agents.TD3.TD3 import TD3


def test_td3_networks_are_plain_mlps():
    """Fujimoto's actor and critic are plain MLPs (sfujim/TD3, TD3.py): only
    Dense kernels and biases, no LayerNorm on the critic's encoder output."""
    agent = TD3(env_id="Pendulum-v1")
    nets = get_initialized_td3_actor_critic(
        jax.random.PRNGKey(0),
        agent.env_args,
        agent.actor_optimizer_args,
        agent.critic_optimizer_args,
        agent.network_args,
    )
    for net in nets:
        paths = [
            [getattr(k, "key", k) for k in path]
            for path, _ in jax.tree_util.tree_flatten_with_path(net.params)[0]
        ]
        assert {path[-1] for path in paths} == {"kernel", "bias"}, paths
        assert not any("norm" in str(k).lower() for p in paths for k in p), paths
