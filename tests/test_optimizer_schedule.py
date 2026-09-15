"""Tests for AdamW weight decay in get_adam_tx and the warmup-cosine schedule."""

import jax
import jax.numpy as jnp
import pytest
from flax.serialization import to_state_dict

from ajax.networks.utils import get_adam_tx
from ajax.schedule import warmup_cosine_schedule
from ajax.state import OptimizerConfig


def _one_step(tx, params, grads):
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    return jax.tree.map(lambda p, u: p + u, params, updates)


def test_zero_weight_decay_is_plain_adam():
    params = {"w": jnp.ones(3)}
    grads = {"w": jnp.zeros(3)}
    new = _one_step(get_adam_tx(1e-2), params, grads)
    assert jnp.allclose(new["w"], 1.0)  # zero grad, no decay -> unchanged


def test_positive_weight_decay_shrinks_params_with_zero_grad():
    params = {"w": jnp.ones(3)}
    grads = {"w": jnp.zeros(3)}
    new = _one_step(get_adam_tx(1e-2, weight_decay=0.1), params, grads)
    assert jnp.allclose(new["w"], 1.0 - 1e-2 * 0.1)


def test_weight_decay_composes_with_clipping_and_rejects_negative():
    tx = get_adam_tx(1e-2, max_grad_norm=0.5, clipped=True, weight_decay=0.1)
    params = {"w": jnp.ones(3)}
    new = _one_step(tx, params, {"w": jnp.full(3, 100.0)})
    assert jnp.all(new["w"] < 1.0) and jnp.all(jnp.isfinite(new["w"]))
    with pytest.raises(ValueError, match="non-negative"):
        get_adam_tx(1e-2, weight_decay=-1.0)


def test_optimizer_config_round_trips_into_get_adam_tx():
    cfg = OptimizerConfig(learning_rate=1e-3, weight_decay=0.05)
    tx = get_adam_tx(**to_state_dict(cfg))
    new = _one_step(tx, {"w": jnp.ones(2)}, {"w": jnp.zeros(2)})
    assert jnp.allclose(new["w"], 1.0 - 1e-3 * 0.05)


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
