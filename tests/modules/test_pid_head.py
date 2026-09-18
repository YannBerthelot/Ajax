"""Tests for the learnable PID output head (ajax.modules.pid_head)."""

import jax
import jax.numpy as jnp
import pytest

from ajax.modules.pid_head import PIDOutputHead, init_pid_carry

T, B, N = 9, 3, 2


def make(**kwargs):
    head = PIDOutputHead(n_outputs=N, **kwargs)
    carry = head.initialize_carry(B)
    z = jax.random.normal(jax.random.PRNGKey(0), (T, B, N))
    resets = jnp.zeros((T, B), bool)
    params = head.init(jax.random.PRNGKey(1), carry, z, resets)
    return head, params, carry, z, resets


def test_init_carry_is_zero_and_shaped():
    integral, previous = init_pid_carry(B, N)
    assert integral.shape == (B, N) and previous.shape == (B, N)
    assert jnp.all(integral == 0) and jnp.all(previous == 0)


def test_needs_at_least_one_term():
    head = PIDOutputHead(n_outputs=N, use_p=False, use_i=False, use_d=False)
    with pytest.raises(ValueError, match="at least one"):
        head.init(
            jax.random.PRNGKey(0),
            head.initialize_carry(B),
            jnp.zeros((1, B, N)),
            jnp.zeros((1, B), bool),
        )


def test_parameters_follow_enabled_terms():
    _, params, *_ = make(use_i=False)
    assert set(params["params"]) == {"kp", "kd"}
    _, params, *_ = make()
    assert set(params["params"]) == {"kp", "ki", "kd"}
    assert jnp.all(params["params"]["kp"] == 1.0)


def test_p_only_with_unit_gain_is_identity():
    head, params, carry, z, resets = make(use_i=False, use_d=False)
    _, u = head.apply(params, carry, z, resets)
    assert jnp.allclose(u, z)


def test_integral_and_derivative_terms_on_known_signals():
    head, params, carry, _, resets = make(kp_init=0.0, ki_init=1.0, kd_init=0.0)
    ones = jnp.ones((T, B, N))
    (integral, previous), u = head.apply(params, carry, ones, resets)
    # I on a constant: running count 1, 2, ..., T
    assert jnp.allclose(u[:, 0, 0], jnp.arange(1, T + 1))
    assert jnp.allclose(integral, T) and jnp.allclose(previous, 1.0)
    head, params, carry, _, resets = make(kp_init=0.0, ki_init=0.0, kd_init=1.0)
    ramp = jnp.broadcast_to(jnp.arange(T, dtype=jnp.float32)[:, None, None], (T, B, N))
    _, u = head.apply(params, carry, ramp, resets)
    # D on a ramp from z_{-1}=0: 0 then 1, 1, ...
    assert jnp.allclose(u[0], 0.0) and jnp.allclose(u[1:], 1.0)


def test_step_vs_sequence_equivalence():
    head, params, carry, z, _ = make(kp_init=0.5, ki_init=0.2, kd_init=0.3)
    resets = jnp.zeros((T, B), bool).at[4, 1].set(True)
    _, u_seq = head.apply(params, carry, z, resets)
    outs = []
    c = carry
    for t in range(T):
        c, u_t = head.apply(params, c, z[t : t + 1], resets[t : t + 1])
        outs.append(u_t[0])
    assert jnp.allclose(jnp.stack(outs), u_seq, atol=1e-6)


def test_reset_clears_memory_like_a_fresh_start():
    head, params, carry, z, _ = make(kp_init=0.5, ki_init=0.2, kd_init=0.3)
    resets = jnp.zeros((T, B), bool).at[4].set(True)
    _, u = head.apply(params, carry, z, resets)
    _, u_fresh = head.apply(params, carry, z[4:], jnp.zeros((T - 4, B), bool))
    assert jnp.allclose(u[4:], u_fresh, atol=1e-6)
    assert not jnp.allclose(
        u[4:], head.apply(params, carry, z, jnp.zeros((T, B), bool))[1][4:]
    )


def test_gains_receive_gradients():
    head, params, carry, z, resets = make(ki_init=0.1, kd_init=0.1)

    def loss(p):
        _, u = head.apply(p, carry, z, resets)
        return (u**2).sum()

    grads = jax.grad(loss)(params)["params"]
    for name in ("kp", "ki", "kd"):
        assert grads[name].shape == (N,) and jnp.all(jnp.isfinite(grads[name]))
        assert jnp.any(grads[name] != 0)


def test_anti_windup_holds_the_integral_while_saturated():
    import jax
    import jax.numpy as jnp

    from ajax.modules.pid_head import PIDOutputHead

    T, B = 30, 1
    z = (
        jnp.ones((T, B, 1)) * 2.0
    )  # a large constant signal: the output rails immediately
    resets = jnp.zeros((T, B), bool).at[0].set(True)
    for aw, expect_bounded in ((None, False), (3.0, True)):
        head = PIDOutputHead(
            n_outputs=1, kp_init=1.0, ki_init=0.5, kd_init=0.0, anti_windup=aw
        )
        params = head.init(jax.random.PRNGKey(0), head.initialize_carry(B), z, resets)
        (integral, _), u = head.apply(params, head.initialize_carry(B), z, resets)
        # without anti-windup the integral grows without bound; with it, it stops once |u| > 3
        assert (float(integral[0, 0]) < 10.0) == expect_bounded, (
            aw,
            float(integral[0, 0]),
        )
        assert bool(
            jnp.all(jnp.diff(u[:, 0, 0]) >= -1e-6)
        )  # monotone in both cases here
