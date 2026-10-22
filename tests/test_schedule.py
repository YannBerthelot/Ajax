"""The warmup-cosine learning-rate schedule (``ajax.schedule``)."""

import jax.numpy as jnp
import pytest

from ajax.schedule import at, linear, warmup_cosine_schedule


def test_linear_ramps_then_stays():
    eps = linear(1.0, 0.1, 50)
    assert eps(jnp.asarray(0)) == 1.0
    assert jnp.allclose(eps(jnp.asarray(25)), 0.55)
    assert jnp.allclose(eps(jnp.asarray(50)), 0.1)
    assert jnp.allclose(eps(jnp.asarray(90)), 0.1)
    assert linear(0.0, 1.0, 0)(jnp.asarray(0)) == 0.0  # at least one step


def test_at_reads_a_constant_or_a_function_of_the_timestep():
    assert at(0.3, jnp.asarray(7)) == 0.3
    assert at(lambda t: 2 * t, jnp.asarray(7)) == 14


def test_warmup_cosine_profile():
    sched = warmup_cosine_schedule(
        1.0, warmup_steps=10, total_steps=100, end_value_fraction=0.1
    )
    assert sched(0) == 0.0
    assert jnp.allclose(sched(10), 1.0)
    assert jnp.allclose(sched(100), 0.1)
    assert sched(5) < sched(10) and sched(50) < sched(10) and sched(50) > sched(100)


def test_warmup_cosine_validation():
    with pytest.raises(ValueError, match="total_steps"):
        warmup_cosine_schedule(1.0, 0, 0)
    with pytest.raises(ValueError, match="warmup_steps"):
        warmup_cosine_schedule(1.0, 20, 10)
    with pytest.raises(ValueError, match="warmup_steps"):
        warmup_cosine_schedule(1.0, 10, 10)  # no decay step left
