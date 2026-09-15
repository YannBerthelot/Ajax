"""Tests for the APG Controller network."""

import distrax
import jax
import jax.numpy as jnp
import pytest

from ajax.agents.APG.networks import Controller, PIDHeadConfig
from ajax.networks.memory import MemoryConfig

T, B, OBS, ACT = 6, 3, 4, 2


def build(**kwargs):
    net = Controller(input_architecture=("8", "relu"), action_dim=ACT, **kwargs)
    obs = jax.random.normal(jax.random.PRNGKey(0), (T, B, OBS))
    done = jnp.zeros((T, B), bool)
    if net.stateful:
        carry = net.initialize_carry(jax.random.PRNGKey(1), B)
        params = net.init(jax.random.PRNGKey(2), obs, hidden_state=carry, done=done)
    else:
        carry = None
        params = net.init(jax.random.PRNGKey(2), obs[0])
    return net, params, carry, obs, done


def test_memoryless_controller_is_deterministic_and_squashed():
    net, params, _, obs, _ = build()
    pi = net.apply(params, obs[0])
    assert isinstance(pi, distrax.Deterministic)
    assert pi.mean().shape == (B, ACT)
    assert jnp.all(jnp.abs(pi.mean()) <= 1.0)
    assert jnp.allclose(pi.entropy(), 0.0)
    assert jnp.array_equal(pi.sample(seed=jax.random.PRNGKey(0)), pi.mean())


def test_unsquashed_controller_can_exceed_unit_box():
    net = Controller(input_architecture=(), action_dim=1, squash=False)
    obs = 50.0 * jnp.ones((B, OBS))
    params = net.init(jax.random.PRNGKey(0), obs)
    params = jax.tree.map(lambda p: jnp.ones_like(p), params)
    assert jnp.any(jnp.abs(net.apply(params, obs).mean()) > 1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"memory": MemoryConfig(kind="gru", hidden_size=8)},
        {"pid": PIDHeadConfig()},
        {
            "memory": MemoryConfig(
                kind="transformer", hidden_size=8, num_heads=2, window=4
            ),
            "pid": PIDHeadConfig(),
        },
    ],
    ids=["memory-only", "pid-only", "transformer+pid"],
)
def test_stateful_controller_step_matches_sequence(kwargs):
    net, params, carry, obs, done = build(**kwargs)
    pi_seq, _ = net.apply(params, obs, hidden_state=carry, done=done)
    u_seq = pi_seq.mean()
    assert u_seq.shape == (T, B, ACT)
    c = carry
    outs = []
    for t in range(T):
        pi_t, c = net.apply(
            params, obs[t : t + 1], hidden_state=c, done=done[t : t + 1]
        )
        outs.append(pi_t.mean()[0])
    assert jnp.allclose(jnp.stack(outs), u_seq, atol=1e-5)


def test_stateful_controller_requires_carry():
    net, params, _, obs, _ = build(pid=PIDHeadConfig())
    with pytest.raises(ValueError, match="hidden_state"):
        net.apply(params, obs)


def test_carry_structure_is_stable_and_zero():
    net, params, carry, obs, done = build(
        memory=MemoryConfig(kind="lstm", hidden_size=8), pid=PIDHeadConfig()
    )
    _, new_carry = net.apply(params, obs, hidden_state=carry, done=done)
    assert jax.tree.structure(new_carry) == jax.tree.structure(carry)
    assert all(jnp.all(leaf == 0) for leaf in jax.tree.leaves(carry))


def test_pid_gains_are_parameters_of_the_controller():
    net, params, *_ = build(pid=PIDHeadConfig(ki_init=0.3))
    gains = params["params"]["pid_head"]
    assert set(gains) == {"kp", "ki", "kd"} and jnp.all(gains["ki"] == 0.3)
