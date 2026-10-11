"""The warmup-cosine learning-rate schedule (``ajax.schedule``)."""

import jax.numpy as jnp
import pytest

from ajax.schedule import warmup_cosine_schedule


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
