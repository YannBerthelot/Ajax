"""BatchRenorm and the running statistics a network keeps apart from its
trained parameters (``LoadedTrainState.batch_stats``)."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ajax import SAC
from ajax.agents.recurrent import actor_dist_train
from ajax.networks.networks import (
    Actor,
    MultiCritic,
    init_network_state,
    predict_value,
    predict_value_train,
)
from ajax.networks.utils import BatchRenorm, parse_layer

X = jax.random.normal(jax.random.PRNGKey(0), (8, 3)) * 2.0 + 1.0
EPS = 1e-3


def stats(**values) -> dict:
    return {"mean": jnp.zeros(3), "var": jnp.ones(3), "steps": jnp.asarray(0)} | values


def run(batch_stats: dict, train: bool):
    variables = {"params": {"scale": jnp.ones(3), "bias": jnp.zeros(3)}}
    variables["batch_stats"] = batch_stats
    if not train:
        return BatchRenorm().apply(variables, X), batch_stats
    out, updates = BatchRenorm().apply(variables, X, mutable=["batch_stats"])
    return out, updates["batch_stats"]


def test_it_normalises_by_the_running_statistics_unless_training() -> None:
    held = stats(mean=jnp.full(3, 0.5), var=jnp.full(3, 4.0))
    out, after = run(held, train=False)
    np.testing.assert_allclose(out, (X - 0.5) / np.sqrt(4.0 + EPS), rtol=1e-6)
    assert after is held


def test_it_trains_as_batch_norm_through_the_warm_up() -> None:
    """The authors' first 100k passes: the batch's own statistics, the
    running ones moved by 1 - momentum, the pass counted."""
    out, after = run(stats(), train=True)
    mean, var = X.mean(0), X.var(0)
    np.testing.assert_allclose(
        out, (X - mean) / np.sqrt(var + EPS), rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(after["mean"], 0.01 * mean, rtol=1e-5)
    np.testing.assert_allclose(after["var"], 0.99 + 0.01 * var, rtol=1e-5)
    assert int(after["steps"]) == 1


def test_after_the_warm_up_it_corrects_the_batch_towards_the_running_statistics() -> (
    None
):
    """r and d (Ioffe 2017, eq. 3), clipped to 3 and 5: here r is clipped,
    the batch's spread being far above the running one."""
    ra_mean, ra_var = jnp.full(3, 0.2), jnp.full(3, 0.1)
    out, _ = run(stats(mean=ra_mean, var=ra_var, steps=jnp.asarray(100_000)), True)
    mean, var = X.mean(0), X.var(0)
    ra_std = np.sqrt(ra_var + EPS)
    r = np.clip(np.sqrt(var + EPS) / ra_std, 1 / 3, 3)
    d = np.clip((mean - ra_mean) / ra_std, -5, 5)
    renorm_mean, renorm_var = mean - d * np.sqrt(var) / r, var / r**2
    expected = (X - renorm_mean) / np.sqrt(renorm_var + EPS)
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)


def first_mean(batch_stats: Any) -> jax.Array:
    """The running mean of an ensemble critic's first layer."""
    return batch_stats["ensemble"]["encoder"]["network"]["layers_0"]["mean"]


def test_a_critic_keeps_its_statistics_apart_from_its_trained_parameters() -> None:
    """Each critic of the ensemble has its own statistics; reads use them,
    training passes return them updated; the optimiser holds only params."""
    critic = MultiCritic(input_architecture=("batchrenorm", "8", "relu"), num=2)
    state = init_network_state(X[:1], critic, jax.random.PRNGKey(1), optax.adam(1e-3))
    assert "batch_stats" not in state.params and state.batch_stats is not None
    held = first_mean(state.batch_stats)
    assert held.shape == (2, 3)
    q, updated = predict_value_train(state, state.params, X)
    assert updated is not None
    moved = first_mean(updated)
    np.testing.assert_allclose(moved, jnp.tile(0.01 * X.mean(0), (2, 1)), rtol=1e-5)
    assert q.shape == predict_value(state, state.params, X).shape == (2, 8, 1)


def test_an_actor_without_running_statistics_trains_as_it_reads() -> None:
    actor = Actor(input_architecture=("8", "relu"), action_dim=2, continuous=True)
    state = init_network_state(X[:1], actor, jax.random.PRNGKey(2), optax.adam(1e-3))
    pi, batch_stats = actor_dist_train(state, state.params, X)
    assert state.batch_stats is None and batch_stats is None
    np.testing.assert_array_equal(pi.mean(), state.apply(state.params, X).mean())


def test_an_agent_reading_its_networks_in_eval_mode_refuses_the_layer() -> None:
    assert isinstance(parse_layer("batchrenorm"), BatchRenorm)
    with pytest.raises(ValueError, match="'batchrenorm' layer would never learn"):
        SAC("Pendulum-v1", critic_architecture=("batchrenorm", "32", "relu"))
