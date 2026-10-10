"""maybe_eval_and_log: the scan-index-gated eval + log step of the agents
with their own evaluation (TrainLoop.evaluate_every)."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ajax.log import maybe_eval_and_log


@struct.dataclass
class State:
    eval_rng: jax.Array
    n_logs: jax.Array
    value: jax.Array


def metrics_fn(state, aux):
    return {"timestep": aux, "Train/value": state.value}


def evaluate_fn(state, key):
    return {"Eval/episodic mean reward": state.value * 2.0 + 0.0 * key[0]}


def extra_eval_metrics(state, key):
    return {"Eval/extra": jnp.float32(7.0)}


PER_ITERATION = 10  # env steps


def run(iterations, log=True, every=3, seeds=2):
    calls = []

    def log_fn(metrics, index):
        calls.append(
            (int(index), int(metrics["timestep"]), float(metrics["Eval/extra"]))
        )

    state = State(
        eval_rng=jax.random.split(jax.random.PRNGKey(0), seeds),
        n_logs=jnp.zeros(seeds, jnp.int32),
        value=jnp.arange(seeds, dtype=jnp.float32),
    )

    def step(state, index, iteration):
        return maybe_eval_and_log(
            state,
            (iteration + 1) * PER_ITERATION,
            index,
            iteration,
            metrics_fn=metrics_fn,
            evaluate_fn=evaluate_fn,
            extra_eval_metrics=extra_eval_metrics,
            log=log,
            log_fn=log_fn,
            every=every,
        )

    outputs = []
    for i in iterations:
        # batched state (resume / curriculum), unbatched iteration
        state, metrics = jax.vmap(step, in_axes=(0, 0, None))(
            state, jnp.arange(seeds), jnp.asarray(i)
        )
        outputs.append(metrics)
    jax.effects_barrier()
    return state, outputs, calls


def test_logs_every_few_iterations_from_the_scan_index():
    state, outputs, calls = run(range(7))  # every 3 iterations
    assert sorted(calls) == [(0, 30, 7.0), (0, 60, 7.0), (1, 30, 7.0), (1, 60, 7.0)]
    np.testing.assert_array_equal(state.n_logs, [2, 2])
    logged = outputs[2]
    assert set(logged) == {
        "timestep",
        "Train/value",
        "Eval/episodic mean reward",
        "Eval/extra",
    }
    np.testing.assert_allclose(logged["Eval/episodic mean reward"], [0.0, 2.0])
    # non-logging iterations return the same structure, NaN / -1 filled
    skipped = outputs[0]
    np.testing.assert_array_equal(skipped["timestep"], [-1, -1])
    assert np.isnan(skipped["Eval/episodic mean reward"]).all()


def test_absolute_iteration_indices_keep_the_cadence_across_a_resume():
    _, _, calls = run(range(4, 9), seeds=1)  # resumed at iteration 4
    assert [c[1] for c in calls] == [60, 90]  # iterations 5 and 8


def test_disabled_logging_evaluates_nothing():
    state, outputs, calls = run(range(4), log=False)
    assert calls == []
    np.testing.assert_array_equal(state.n_logs, [0, 0])
    assert all(np.isnan(o["Eval/episodic mean reward"]).all() for o in outputs)


def test_eval_and_extra_metrics_get_the_two_halves_of_the_eval_key():
    """``eval_key, extra_key = split(eval_rng)``: APG's order, pinned."""
    state = State(
        eval_rng=jax.random.PRNGKey(5), n_logs=jnp.int32(0), value=jnp.float32(0.0)
    )
    _, metrics = maybe_eval_and_log(
        state,
        jnp.int32(1),
        0,
        jnp.int32(0),
        metrics_fn=metrics_fn,
        evaluate_fn=lambda state, key: {"Eval/key": jax.random.uniform(key)},
        extra_eval_metrics=lambda state, key: {"Eval/extra": jax.random.uniform(key)},
        log=True,
        log_fn=lambda metrics, index: None,
        every=1,
    )
    jax.effects_barrier()
    eval_key, extra_key = jax.random.split(state.eval_rng)
    np.testing.assert_array_equal(metrics["Eval/key"], jax.random.uniform(eval_key))
    np.testing.assert_array_equal(metrics["Eval/extra"], jax.random.uniform(extra_key))


def test_the_evaluation_stays_a_real_cond_under_the_seed_vmap():
    """Batched agent state, unbatched iteration: the gate lowers to a ``cond``
    (the evaluation runs only on logging iterations), never to a select
    that would evaluate on every iteration."""
    seeds = 3
    state = State(
        eval_rng=jax.random.split(jax.random.PRNGKey(0), seeds),
        n_logs=jnp.zeros(seeds, jnp.int32),
        value=jnp.arange(seeds, dtype=jnp.float32),
    )

    def step(state, index, iteration):
        return maybe_eval_and_log(
            state,
            iteration * 10,
            index,
            iteration,
            metrics_fn=metrics_fn,
            evaluate_fn=evaluate_fn,
            extra_eval_metrics=extra_eval_metrics,
            log=True,
            log_fn=lambda metrics, index: None,
            every=3,
        )

    jaxpr = jax.make_jaxpr(jax.vmap(step, in_axes=(0, 0, None)))(
        state, jnp.arange(seeds), jnp.int32(2)
    ).jaxpr
    names = [e.primitive.name for e in jaxpr.eqns]
    assert names.count("cond") == 1
