"""The shared training loop (``ajax.agents.loop``): its steps on a stand-in
state (the update gate before ``learning_starts``, the extensions'
``post_update`` key, one gradient step, an agent's own update schedule and
evaluation), and on a tiny APO run the evaluations ``train`` returns with a
logging config. The agents' smoke and resume tests and the probes run the
loop end to end."""

import dataclasses
import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from ajax.agents import loop as loop_module
from ajax.agents.APO.APO import APO
from ajax.agents.loop import TrainLoop, critic_step, gradient_step
from ajax.agents.recurrent import RecurrentCarries, bootstrap_cuts
from ajax.checkpoint import restore_into, save_checkpoint
from ajax.extensions.base import Extension, ExtensionStack
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import Transition


@struct.dataclass
class Collector:
    timestep: jax.Array


@struct.dataclass
class State:
    rng: jax.Array
    collector_state: Collector
    value: jax.Array
    ext_state: tuple = ()


@struct.dataclass
class Metrics:
    loss: jax.Array


@dataclass(frozen=True)
class Count(Extension):
    name: str = "count"

    def init_state(self, agent_state, rng):
        return jnp.asarray(0)

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + 1


def _loop(*extensions: Extension) -> TrainLoop:
    return TrainLoop(
        env_args=None,  # type: ignore[arg-type]
        total_timesteps=100,
        num_episode_test=1,
        stack=ExtensionStack(extensions),
        log=False,
        log_fn=print,
        log_frequency=None,
    )


def _state(timestep: int, ext_state: tuple = ()) -> State:
    return State(
        rng=jax.random.PRNGKey(0),
        collector_state=Collector(jnp.asarray(timestep)),
        value=jnp.asarray(1.0),
        ext_state=ext_state,
    )


def _update(state: State) -> tuple[State, Metrics]:
    return state.replace(value=state.value + 1.0), Metrics(loss=state.value)


def test_no_update_before_learning_starts() -> None:
    """Before ``learning_starts`` the state is untouched and every metric
    is a (1,) NaN; from it the update runs, its metrics reshaped to (1,)."""
    loop = _loop()
    early, metrics = loop.maybe_update(_state(9), 10, _update, Metrics)
    assert float(early.value) == 1.0 and metrics.loss.shape == (1,)
    assert np.isnan(metrics.loss).all()
    ready, metrics = loop.maybe_update(_state(10), 10, _update, Metrics)
    assert float(ready.value) == 2.0
    np.testing.assert_array_equal(metrics.loss, [1.0])


def test_post_update_draws_a_key_only_with_extensions() -> None:
    """No extension: the state as it was, its key unused. One: folded once,
    on a fresh key split from the state's."""
    state = _state(0)
    assert _loop().post_update(state) is state
    counted = _loop(Count()).post_update(_state(0, (jnp.asarray(0),)))
    assert int(counted.ext_state[0]) == 1
    _, rng = jax.random.split(state.rng)
    np.testing.assert_array_equal(counted.rng, rng)


def test_the_post_update_runs_after_the_update() -> None:
    """Inside the gate, the extensions see the updated state."""

    @dataclass(frozen=True)
    class Record(Extension):
        name: str = "record"

        def post_update(self, agent_state, ext_state, ctx):
            return agent_state, agent_state.value  # the value it saw

    state = _state(5, (jnp.asarray(0.0),))
    out, _ = _loop(Record()).maybe_update(state, 0, _update, Metrics)
    assert float(out.ext_state[0]) == 2.0


def test_on_policy_runs_the_budget_in_whole_rollouts_plus_one() -> None:
    loop = dataclasses.replace(_loop(), env_args=SimpleNamespace(n_envs=4))  # type: ignore[arg-type]
    assert loop.n_rollouts(8) == 100 // 32 + 1


def test_gradient_step_applies_the_loss_gradient() -> None:
    params = {"w": jnp.asarray(3.0)}
    train_state = TrainState.create(apply_fn=None, params=params, tx=optax.sgd(0.5))

    def loss_fn(p):
        return p["w"] ** 2, {"w": p["w"]}

    stepped, aux = gradient_step(train_state, loss_fn)
    assert float(stepped.params["w"]) == 3.0 - 0.5 * 6.0
    assert float(aux["w"]) == 3.0 and int(stepped.step) == 1


# --- Where a replayed target stops bootstrapping ------------------------------
# Rows: running, truncated, terminated, both.
_ENDS = Transition(
    obs=jnp.zeros((4, 1)),
    action=jnp.zeros((4, 1)),
    reward=jnp.zeros((4, 1)),
    terminated=jnp.array([[0.0], [0.0], [1.0], [1.0]]),
    truncated=jnp.array([[0.0], [1.0], [0.0], [1.0]]),
    next_obs=jnp.zeros((4, 1)),
)


def test_a_replayed_target_cuts_at_terminations_and_sequences_at_any_end() -> None:
    """A row stores its final observation at a time limit; a sequence's
    next observations are the next rows, so it still cuts there."""
    resets = jnp.zeros((1, 4), dtype=bool)
    carries = RecurrentCarries(resets, resets, *(None,) * 6)
    np.testing.assert_array_equal(bootstrap_cuts(_ENDS, None)[:, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(bootstrap_cuts(_ENDS, carries)[:, 0], [0, 1, 1, 1])


@dataclass(frozen=True)
class AddDones(Extension):
    name: str = "add_dones"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target + batch["dones"]


def test_the_extensions_see_the_cuts_the_target_used() -> None:
    """critic_step hands the extensions the agent's own cuts (here the
    terminations alone), not terminated OR truncated."""
    params = {"w": jnp.zeros(())}
    critic = TrainState.create(apply_fn=None, params=params, tx=optax.sgd(1.0))
    state = SimpleNamespace(
        collector_state=Collector(jnp.asarray(0)), critic_state=critic, ext_state=((),)
    )

    def value_loss(params, target_q):
        return jnp.mean((params["w"] - target_q) ** 2), target_q

    _, seen = critic_step(
        state,
        _ENDS,
        jnp.zeros((4, 1)),
        value_loss,
        ExtensionStack((AddDones(),)),
        jax.random.PRNGKey(0),
        100,
        rewards=_ENDS.reward,
        dones=_ENDS.terminated,
        gamma=0.9,
        reward_scale=1.0,
    )
    np.testing.assert_array_equal(seen[:, 0], [0, 0, 1, 1])


# --- Evaluations without a logging backend -----------------------------------
def _config(**kw: Any) -> LoggingConfig:
    return LoggingConfig(config={}, log_frequency=10, use_wandb=False, **kw)


def test_the_logging_worker_starts_only_for_a_backend() -> None:
    with mock.patch.object(loop_module, "start_async_logging") as start:
        TrainLoop.create(None, 64, 1)  # type: ignore[arg-type]
        TrainLoop.create(None, 64, 1, logging_config=_config())  # type: ignore[arg-type]
        start.assert_not_called()
        TrainLoop.create(None, 64, 1, logging_config=_config(use_tensorboard=True))  # type: ignore[arg-type]
        start.assert_called_once()


@dataclass(frozen=True)
class Counted(Count):
    """``Count`` reporting its (integer) count at every evaluation."""

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {"count": ext_state}


SEEDS = [0, 1]
# 16 steps an iteration, the 4 evaluations at 16, 32, 48 and 64 of 6 rows
# (64 // 10): the gate lags the 10-step frequency.
BUDGET, ROWS = 64, 6


def _apo() -> APO:
    net = ("8", "tanh")
    return APO(
        "Pendulum-v1",
        n_envs=2,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        actor_architecture=net,
        critic_architecture=net,
        extensions=(Counted(),),
    )


@functools.cache
def _logged_run() -> tuple[Any, dict, list]:
    """A logged run without a backend: its state, the evaluations ``train``
    returns and the (seed index, metrics) the logger receives."""
    events: list = []

    def capture(metrics: dict, index: Any, **_: Any) -> None:
        events.append((int(index), {k: np.asarray(v) for k, v in metrics.items()}))

    with mock.patch.object(loop_module, "vmap_log", capture):
        state, rows = _apo().train(
            seed=SEEDS, n_timesteps=BUDGET, num_episode_test=1, logging_config=_config()
        )
        jax.effects_barrier()
    return state, rows, events


def test_train_returns_the_evaluations_the_logger_receives() -> None:
    """Every key logged, one row per evaluation in order, then the -1 / NaN
    sentinel up to the most one call can make."""
    _, rows, events = _logged_run()
    assert set(rows) == set(events[0][1]) and "count" in rows
    assert all(np.shape(v) == (len(SEEDS), ROWS) for v in rows.values())
    for seed in range(len(SEEDS)):
        logged = [m for i, m in events if i == seed]
        assert [int(m["timestep"]) for m in logged] == [16, 32, 48, 64]
        for row, metrics in enumerate(logged):
            for key, value in metrics.items():
                np.testing.assert_array_equal(rows[key][seed, row], value, key)
        for key, value in rows.items():
            empty = -1 if np.issubdtype(value.dtype, np.integer) else np.nan
            np.testing.assert_array_equal(value[seed, len(logged) :], empty, key)


def test_a_logged_run_resumes_from_its_checkpoint(tmp_path: Any) -> None:
    """A checkpoint of ``train``'s whole result restores into an unlogged
    run's skeleton, and the resumed run returns its own evaluations."""
    state, rows, _ = _logged_run()
    path = str(tmp_path / "checkpoint.pkl")
    save_checkpoint((state, rows), path)
    agent = _apo()
    skeleton = agent.train(seed=SEEDS, n_timesteps=1, num_episode_test=1)
    assert skeleton[1] is None
    # Logged, a call too short to evaluate returns no rows.
    _, no_rows = agent.train(seed=SEEDS, n_timesteps=1, logging_config=_config())
    assert no_rows["timestep"].shape == (len(SEEDS), 0)
    resumed, resumed_rows = agent.train(
        seed=SEEDS,
        n_timesteps=BUDGET,
        num_episode_test=1,
        logging_config=_config(),
        initial_state=restore_into(skeleton, path),
    )
    save_checkpoint((resumed, resumed_rows), path)
    # The second call evaluates nothing: ajax.log's gate stops at its own
    # budget, which the restored timestep is past.
    np.testing.assert_array_equal(resumed.collector_state.timestep, 2 * 80)
    assert resumed_rows["timestep"].shape == (len(SEEDS), ROWS)


# --- Agents acting, training and evaluating their own way --------------------
@struct.dataclass
class Logged(State):
    """A stand-in state that evaluates: an evaluation key and a log count."""

    eval_rng: jax.Array = struct.field(default_factory=lambda: jax.random.PRNGKey(1))
    n_logs: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))


def test_repeat_update_folds_post_update_after_each_update() -> None:
    """``n`` updates, each followed by ``post_update``; none for ``n = 0``."""
    state, loop = _state(5, (jnp.asarray(0),)), _loop(Count())
    run = jax.jit(lambda s, n: loop.repeat_update(s, n, _update))
    out, same = run(state, 3), run(state, 0)
    assert float(out.value) == 4.0 and int(out.ext_state[0]) == 3
    assert float(same.value) == 1.0 and int(same.ext_state[0]) == 0


def test_a_tick_schedule_is_one_unbatched_while_under_the_seed_vmap() -> None:
    """A count from unbatched values (a train ratio of the tick) is one
    ``while`` whose predicate is not batched, though the states are per
    seed: no select-masked loop to the largest count."""
    loop = _loop()
    run = jax.vmap(lambda s, n: loop.repeat_update(s, n, _update), in_axes=(0, None))
    states = jax.vmap(lambda v: _state(0).replace(value=v))(jnp.arange(4.0))
    jaxpr = jax.make_jaxpr(run)(states, 3).jaxpr
    loops = [e for e in jaxpr.eqns if e.primitive.name == "while"]
    assert len(loops) == 1
    assert loops[0].params["cond_jaxpr"].jaxpr.outvars[0].aval.shape == ()
    np.testing.assert_array_equal(run(states, 3).value, np.arange(4.0) + 3)


def test_an_agents_own_evaluation_runs_every_few_ticks() -> None:
    """On the ticks ``t`` with ``(t + 1) % every == 0``: the agent's metrics,
    its evaluation and the extensions' metrics, one more log, ``after_log``
    told it logged; the -1 timestep sentinel and NaN on the others."""
    loop = dataclasses.replace(_loop(Counted()), log=True, log_fn=lambda *_: None)
    evaluation = loop_module.Evaluation(
        metrics=lambda s, aux: {"timestep": s.collector_state.timestep},
        every=2,
        evaluate=lambda s, key: {"Eval/value": s.value},
        after_log=lambda s, logged: s.replace(value=jnp.where(logged, 0.0, s.value)),
    )
    state = Logged(
        rng=jax.random.PRNGKey(0),
        collector_state=Collector(jnp.asarray(7)),
        value=jnp.asarray(3.0),
        ext_state=(jnp.asarray(5),),
    )
    step = jax.jit(lambda s, t: loop.evaluate_every(s, None, 0, t, evaluation))
    skipped, metrics = step(state, 0)
    assert int(metrics["timestep"]) == -1 and np.isnan(metrics["Eval/value"])
    assert int(skipped.n_logs) == 0 and float(skipped.value) == 3.0
    logged, metrics = step(state, 1)
    assert int(metrics["timestep"]) == 7 and float(metrics["Eval/value"]) == 3.0
    assert int(metrics["count"]) == 5 and int(logged.n_logs) == 1
    assert float(logged.value) == 0.0
    # Rows for every evaluation a call of 5 ticks can make; none without.
    assert evaluation.count(5) == 3
    assert dataclasses.replace(evaluation, every=None).count(5) == 0
