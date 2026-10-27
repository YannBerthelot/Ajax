"""maybe_eval_and_log: the scan-index-gated eval + log step of the agents
with their own evaluation (TrainLoop.evaluate_every)."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from ajax.log import MetricsAccumulator, maybe_eval_and_log
from ajax.random_keys import derive_key


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


def make_state(seeds):
    """``seeds`` agent states (batched), the values ``0 .. seeds - 1``."""
    return State(
        eval_rng=jax.random.split(jax.random.PRNGKey(0), seeds),
        n_logs=jnp.zeros(seeds, jnp.int32),
        value=jnp.arange(seeds, dtype=jnp.float32),
    )


def make_step(log_fn=lambda metrics, index: None, log=True, every=3):
    """The agents' step: ``(state, seed index, iteration) -> (state,
    metrics)``, ``PER_ITERATION`` env steps per iteration."""

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

    return step


def run(iterations, log=True, every=3, seeds=2):
    calls = []

    def log_fn(metrics, index):
        calls.append(
            (int(index), int(metrics["timestep"]), float(metrics["Eval/extra"]))
        )

    state, step = make_state(seeds), make_step(log_fn, log, every)

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
    assert outputs == [None] * 4


def test_disabled_logging_does_not_even_trace_the_evaluation():
    def traced(*_):
        raise AssertionError("traced without logging")

    state = State(
        eval_rng=jax.random.PRNGKey(0), n_logs=jnp.int32(0), value=jnp.float32(0.0)
    )
    step = jax.jit(
        lambda state, iteration: maybe_eval_and_log(
            state,
            None,
            0,
            iteration,
            metrics_fn=traced,
            evaluate_fn=traced,
            extra_eval_metrics=traced,
            log=False,
            log_fn=traced,
            every=1,
        )
    )
    state, metrics = step(state, jnp.int32(0))
    assert metrics is None and int(state.n_logs) == 0


def test_each_evaluation_draws_the_keys_of_its_timestep():
    """``eval_key, extra_key = split(fold_in(online key, timestep))``
    (APG's order), the timestep the metrics log: the evaluation and the
    extensions' metrics draw new keys at each evaluation."""
    state = State(
        eval_rng=jax.random.PRNGKey(5), n_logs=jnp.int32(0), value=jnp.float32(0.0)
    )
    draws = []
    for iteration in range(2):
        timestep = 10 * (iteration + 1)
        state, metrics = maybe_eval_and_log(
            state,
            jnp.int32(timestep),
            0,
            jnp.int32(iteration),
            metrics_fn=metrics_fn,
            evaluate_fn=lambda state, key: {"Eval/key": jax.random.uniform(key)},
            extra_eval_metrics=lambda state, key: {
                "Eval/extra": jax.random.uniform(key)
            },
            log=True,
            log_fn=lambda metrics, index: None,
            every=1,
        )
        online = derive_key(jax.random.PRNGKey(5), "online_evaluation")
        keys = jax.random.split(jax.random.fold_in(online, timestep))
        eval_draw, extra_draw = (jax.random.uniform(key) for key in keys)
        np.testing.assert_array_equal(metrics["Eval/key"], eval_draw)
        np.testing.assert_array_equal(metrics["Eval/extra"], extra_draw)
        draws.append((float(eval_draw), float(extra_draw)))
    assert draws[0][0] != draws[1][0] and draws[0][1] != draws[1][1]


def test_the_evaluation_stays_a_real_cond_under_the_seed_vmap():
    """Batched agent state, unbatched iteration: the gate lowers to a ``cond``
    (the evaluation runs only on logging iterations), never to a select
    that would evaluate on every iteration."""
    seeds = 3
    jaxpr = jax.make_jaxpr(jax.vmap(make_step(), in_axes=(0, 0, None)))(
        make_state(seeds), jnp.arange(seeds), jnp.int32(2)
    ).jaxpr
    names = [e.primitive.name for e in jaxpr.eqns]
    assert names.count("cond") == 1


def test_the_accumulator_means_the_updates_since_the_last_log():
    """NaN before an update, the mean of each metric after, zeroed where a
    log was written."""
    metrics = MetricsAccumulator.zeros(("loss", "grad"))
    assert np.isnan(np.asarray(metrics.mean()["loss"]))
    metrics = metrics.add({"loss": 1.0, "grad": 4}).add({"loss": 3.0, "grad": 2})
    means = {name: float(value) for name, value in metrics.mean().items()}
    assert means == {"loss": 2.0, "grad": 3.0}
    kept = metrics.reset_where(jnp.asarray(False))
    assert int(kept.count) == 2 and float(kept.total["loss"]) == 4.0
    restarted = metrics.reset_where(jnp.asarray(True))
    assert int(restarted.count) == 0 and float(restarted.total["loss"]) == 0.0
