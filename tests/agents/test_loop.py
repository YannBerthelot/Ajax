"""The shared training loop (``ajax.agents.loop``): its steps on a stand-in
state (the update gate before ``learning_starts``, the extensions'
``post_update`` key, one gradient step, an agent's own update schedule and
evaluation), and on a tiny APO run the evaluations ``train`` returns with a
logging config. The agents' smoke and resume tests and the probes run the
loop end to end."""

import dataclasses
import functools
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import struct
from flax.training.train_state import TrainState

from ajax.agents import loop as loop_module
from ajax.agents.APO.APO import APO
from ajax.agents.loop import TrainLoop
from ajax.agents.SAC.SAC import SAC
from ajax.agents.update_steps import (
    REPLAY_ACTOR_KEYS,
    actor_step,
    apply_on_target,
    compute_extensions_critic_loss,
    compute_gradients,
    critic_step,
    gradient_step,
    run_actor_if,
    run_if,
)
from ajax.checkpoint import restore_into, save_checkpoint
from ajax.evaluate import evaluate
from ajax.extensions.base import Extension, ExtensionStack, merge_metrics
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import Transition
from ajax.types import Replaceable
from tests.agents.extension_stubs import CounterExt


@struct.dataclass
class Collector:
    timestep: jax.Array


@struct.dataclass
class State(Replaceable):
    rng: jax.Array
    collector_state: Collector
    value: jax.Array
    ext_state: tuple = ()


@struct.dataclass
class Metrics:
    loss: jax.Array


def _loop(*extensions: Extension) -> TrainLoop:
    return TrainLoop(
        env_args=None,  # type: ignore[arg-type]
        total_timesteps=100,
        num_episode_test=1,
        extensions=ExtensionStack(extensions),
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
    is a (1,) NaN; from it the update runs, its metrics reshaped to (1,).
    The steps collected count every env's, not the collector's iterations.
    A given timestep (the step's own, for an update without a replay)
    replaces the steps collected."""
    two_envs = SimpleNamespace(n_envs=2)
    loop = dataclasses.replace(_loop(), env_args=two_envs)  # type: ignore[arg-type]
    early, metrics = loop.maybe_update(_state(9), 10, _update)
    assert float(early.value) == 1.0 and metrics.loss.shape == (1,)
    assert np.isnan(metrics.loss).all()
    ready, metrics = loop.maybe_update(_state(10), 10, _update)
    assert float(ready.value) == 2.0
    np.testing.assert_array_equal(metrics.loss, [1.0])
    collected_at = jnp.asarray(8)
    early, _ = loop.maybe_update(_state(10), 10, _update, collected_at)
    assert float(early.value) == 1.0


@pytest.mark.parametrize(
    "uniform_steps, uniform", [(None, [True, True, False]), (0, [False] * 3)]
)
def test_the_collector_acts_uniformly_until_uniform_steps(uniform_steps, uniform):
    """Before ``learning_starts`` the collector acts uniformly, unless
    ``uniform_steps`` sets another warm-up (0: the policy acts from the
    first step, as TQC's authors' does)."""
    seen = []

    def collect(agent_state, _, uniform, **kwargs):
        seen.append(bool(uniform))
        tick = agent_state.collector_state.timestep + 1
        return agent_state.replace(collector_state=Collector(tick)), None

    loop = dataclasses.replace(
        _loop(), env_args=SimpleNamespace(mode="gymnax", n_envs=1)
    )
    built = {}

    def train(self, init, iteration, *args, **kwargs):
        built["iteration"] = iteration

    with (
        mock.patch.object(loop_module, "collect_experience", collect),
        mock.patch.object(TrainLoop, "train", train),
        mock.patch.object(
            TrainLoop, "evaluate_and_log", lambda self, s, *a, **k: (s, None)
        ),
    ):
        loop.off_policy(None, lambda s, _: _update(s), 2, uniform_steps=uniform_steps)
        state = _state(0)
        for tick in range(3):
            state, _ = built["iteration"](state, 0, tick)
    assert seen == uniform


def test_post_update_leaves_the_updates_key() -> None:
    """No extension: the state as it was. One: folded once, on a key derived
    from the state's, which stays as it was."""
    state = _state(0)
    assert _loop().post_update(state) is state
    counted = _loop(CounterExt()).post_update(_state(0, (jnp.asarray(0),)))
    assert int(counted.ext_state[0]) == 1
    np.testing.assert_array_equal(counted.rng, state.rng)


def test_a_resumed_call_trains_its_budget_toward_the_run_horizon() -> None:
    """A call from timestep 80 to the run's horizon 144 trains 64 steps,
    folds its extensions over the whole run, and has a row for each log
    point of the run past 80 (90 to 140)."""
    seen = []

    @dataclass(frozen=True)
    class Total(Extension):
        name: str = "total"

        def post_update(self, agent_state, ext_state, context):
            seen.append(context.total_steps)
            return agent_state, ext_state

    resumed = dataclasses.replace(
        _loop(Total()),
        total_timesteps=144,
        start_timestep=80,
        log=True,
        log_frequency=10,
    )
    assert (resumed.budget, resumed.n_evaluations) == (64, 6)
    resumed.post_update(_state(100, (jnp.asarray(0),)))
    assert seen == [144]


def test_a_call_told_a_later_horizon_trains_to_its_own_end() -> None:
    """A first call to 64 of a run to 100 trains 64 steps (whole rollouts
    plus one: 3 of 32), folds its extensions over the whole run and has a
    row for each log point up to where it reaches, never past the horizon."""
    seen = []

    @dataclass(frozen=True)
    class Total(Extension):
        name: str = "total"

        def post_update(self, agent_state, ext_state, context):
            seen.append(context.total_steps)
            return agent_state, ext_state

    first = dataclasses.replace(
        _loop(Total()),
        env_args=SimpleNamespace(n_envs=4),  # type: ignore[arg-type]
        end_timestep=64,
        log=True,
        log_frequency=10,
    )
    assert (first.budget, first.n_evaluations, first.n_rollouts(8)) == (64, 6, 3)
    assert (first.evaluations_until(96), first.evaluations_until(160)) == (9, 10)
    first.post_update(_state(10, (jnp.asarray(0),)))
    assert seen == [100]


def test_the_post_update_runs_after_the_update() -> None:
    """Inside the gate, the extensions see the updated state."""

    @dataclass(frozen=True)
    class Record(Extension):
        name: str = "record"

        def post_update(self, agent_state, ext_state, context):
            return agent_state, agent_state.value  # the value it saw

    state = _state(5, (jnp.asarray(0.0),))
    out, _ = _loop(Record()).maybe_update(state, 0, _update)
    assert float(out.ext_state[0]) == 2.0


def test_on_policy_runs_the_budget_in_whole_rollouts_plus_one() -> None:
    loop = dataclasses.replace(_loop(), env_args=SimpleNamespace(n_envs=4))  # type: ignore[arg-type]
    with pytest.warns(UserWarning, match="not a multiple"):
        assert loop.n_rollouts(8) == 100 // 32 + 1


def test_a_budget_off_whole_iterations_warns() -> None:
    """A budget that is not a whole number of iterations warns, naming the
    iteration: the run stops at a multiple of it, so an evaluation due at
    the horizon never happens. A whole number stays quiet."""
    loop = dataclasses.replace(_loop(), env_args=SimpleNamespace(n_envs=4))  # type: ignore[arg-type]
    with pytest.warns(
        UserWarning, match=r"not a multiple of the 32 env steps.*n_envs x n_steps"
    ):
        loop.n_rollouts(8)
    with pytest.warns(
        UserWarning, match=r"of the 3 env steps of one iteration \(n_envs\)"
    ):
        loop.warn_if_partial(3, "n_envs")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert loop.n_rollouts(5) == 100 // 20 + 1
        loop.warn_if_partial(4, "n_envs")


def test_a_split_call_warns_at_its_own_end_and_a_run_count_stays_quiet() -> None:
    """A call to 60 of a run to 100, not whole rollouts of 32, warns about
    timestep 60, its own end. Counting the whole run's rollouts (PPO's
    forced-reset cadence) does not warn: that count is not the call's
    budget."""
    first = dataclasses.replace(
        _loop(),
        env_args=SimpleNamespace(n_envs=4),  # type: ignore[arg-type]
        end_timestep=60,
    )
    with pytest.warns(UserWarning, match=r"evaluation due at timestep 60 "):
        assert first.n_rollouts(8) == 60 // 32 + 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert first.n_rollouts(8, 100) == 100 // 32 + 1


def test_gradient_step_applies_the_loss_gradient() -> None:
    params = {"w": jnp.asarray(3.0)}
    train_state = TrainState.create(apply_fn=None, params=params, tx=optax.sgd(0.5))

    def loss_fn(p):
        return p["w"] ** 2, {"w": p["w"]}

    stepped, aux, grad_norm = gradient_step(train_state, loss_fn)
    assert float(stepped.params["w"]) == 3.0 - 0.5 * 6.0
    assert float(aux["w"]) == 3.0 and int(stepped.step) == 1
    assert float(grad_norm) == 6.0


def test_gradient_step_reports_the_norm_before_the_clip() -> None:
    """The norm is the raw gradients': an optimiser that clips them to 1
    steps by 1, and the reported norm stays 6."""
    params = {"w": jnp.asarray(3.0), "v": jnp.asarray(0.0)}
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.sgd(1.0))
    train_state = TrainState.create(apply_fn=None, params=params, tx=tx)

    def loss_fn(p):
        return p["w"] ** 2 + 0.0 * p["v"], None

    stepped, _, grad_norm = gradient_step(train_state, loss_fn)
    assert float(stepped.params["w"]) == 2.0 and float(grad_norm) == 6.0


def test_compute_gradients_returns_the_gradients_and_the_aux() -> None:
    """For one network's parameters, and for a tuple of two networks' (APO's
    summed loss), the gradients having the parameters' structure."""
    grads, aux = compute_gradients(
        lambda params: (params["w"] ** 2, {"w": params["w"]}), {"w": jnp.asarray(3.0)}
    )
    assert float(grads["w"]) == 6.0 and float(aux["w"]) == 3.0

    def summed_loss(params: tuple) -> tuple:
        actor_params, critic_params = params
        return actor_params["w"] ** 2 + 3.0 * critic_params["v"], critic_params["v"]

    both = ({"w": jnp.asarray(2.0)}, {"v": jnp.asarray(5.0)})
    (actor_grads, critic_grads), aux = compute_gradients(summed_loss, both)
    assert float(actor_grads["w"]) == 4.0 and float(critic_grads["v"]) == 3.0
    assert float(aux) == 5.0


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


@dataclass(frozen=True)
class AddDones(Extension):
    name: str = "add_dones"

    def on_target(self, agent_state, ext_state, batch, target, context):
        return target + batch["dones"]


def test_the_extensions_see_the_cuts_the_target_used() -> None:
    """critic_step hands the extensions the agent's own cuts (here the
    terminations alone), not terminated OR truncated."""
    params = {"w": jnp.zeros(())}
    critic = TrainState.create(apply_fn=None, params=params, tx=optax.sgd(1.0))
    state = SimpleNamespace(
        collector_state=Collector(jnp.asarray(0)),
        critic_state=critic,
        actor_state=None,
        ext_state=((),),
    )

    def value_loss(params, target_q):
        return jnp.mean((params["w"] - target_q) ** 2), target_q

    _, seen, grad_norm, _ = critic_step(
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
    # d/dw mean((w - target)^2) at w = 0 on the target [0, 0, 1, 1]: -1.
    assert float(grad_norm) == 1.0


@dataclass(frozen=True)
class PullTerm(Extension):
    """Adds ``w * mean(targets)`` to the critic loss and logs the targets."""

    name: str = "pull"

    def critic_loss(self, agent_state, ext_state, batch, context):
        mean = batch["targets"].mean()
        return batch["critic_params"]["w"] * mean, {"targets": mean}


def test_the_critic_steps_halves_compose_to_the_step() -> None:
    """An agent whose target comes out of its own loss (CrossQ) folds the
    extensions with critic_step's two halves inside it: the same step, the
    same metrics as critic_step's."""
    critic = TrainState.create(
        apply_fn=None, params={"w": jnp.asarray(0.5)}, tx=optax.sgd(1.0)
    )
    state = SimpleNamespace(
        collector_state=Collector(jnp.asarray(0)),
        critic_state=critic,
        actor_state=None,
        ext_state=((), ()),
    )
    stack = ExtensionStack((AddDones(), PullTerm()))
    key = jax.random.PRNGKey(0)
    target_args = {
        "rewards": _ENDS.reward,
        "dones": _ENDS.terminated,
        "gamma": 0.9,
        "reward_scale": 1.0,
    }

    def value_loss(params, target_q):
        return jnp.mean((params["w"] - target_q) ** 2), target_q

    stepped, _, _, metrics = critic_step(
        state, _ENDS, jnp.ones((4, 1)), value_loss, stack, key, 100, **target_args
    )

    def loss_fn(params):
        target_q, target_metrics = apply_on_target(
            stack, state, _ENDS, jnp.ones((4, 1)), key, 100, **target_args
        )
        loss, _ = value_loss(params, target_q)
        extra, loss_metrics = compute_extensions_critic_loss(
            stack, state, _ENDS, params, target_q, key, 100
        )
        return loss + extra, merge_metrics(target_metrics, loss_metrics)

    composed, own_metrics, _ = gradient_step(critic, loss_fn)
    assert float(composed.params["w"]) == float(stepped.params["w"])
    assert jax.tree.map(float, own_metrics) == jax.tree.map(float, metrics)


@dataclass(frozen=True)
class SquareTerm(Extension):
    """Adds ``w ** 2`` to the actor loss and logs the agent's loss."""

    name: str = "square"
    seen: dict = dataclasses.field(default_factory=dict, hash=False, compare=False)

    def actor_loss(self, agent_state, ext_state, batch, context):
        self.seen.update(dict.fromkeys(batch))
        return batch["actor_params"]["w"] ** 2, {"agent_loss": batch["agent_loss"]}


def test_actor_step_steps_down_the_loss_plus_the_extensions_terms() -> None:
    """The extensions see the rows' entries and the critics the agent passes
    with the differentiated parameters, the mean action and the agent's
    loss; the actor steps down the sum: d/dw (2w + w^2) at w = 3 is 8."""
    actor = TrainState.create(
        apply_fn=None, params={"w": jnp.asarray(3.0)}, tx=optax.sgd(0.5)
    )
    state = SimpleNamespace(
        collector_state=Collector(jnp.asarray(0)), actor_state=actor, ext_state=((),)
    )
    term = SquareTerm()
    zeros = jnp.zeros((1, 1))

    def policy_loss(params):
        return 2.0 * params["w"], jnp.asarray(1.0), params["w"] + 1.0

    stepped, aux, grad_norm, metrics = actor_step(
        state,
        policy_loss,
        ExtensionStack((term,)),
        jax.random.PRNGKey(0),
        100,
        Transition(zeros, zeros, zeros, zeros, zeros, zeros),
    )
    assert float(stepped.params["w"]) == 3.0 - 0.5 * 8.0 and float(grad_norm) == 8.0
    assert float(aux) == 1.0 and float(jax.tree.leaves(metrics)[0]) == 6.0
    assert set(term.seen) == set(REPLAY_ACTOR_KEYS)


def test_run_if_runs_the_update_only_where_the_condition_holds() -> None:
    def double(x):
        return x * 2

    assert float(run_if(jnp.asarray(True), double, jnp.asarray(3.0))) == 6.0
    assert float(run_if(jnp.asarray(False), double, jnp.asarray(3.0))) == 3.0


def test_run_if_fills_a_skipped_updates_metrics() -> None:
    """Skipped, the metrics keep the update's structure: NaN, and -1 in integer
    leaves."""

    def step(x):
        return x + 1.0, {"loss": x * 2, "count": jnp.asarray(1)}

    x, metrics = run_if(jnp.asarray(True), step, jnp.asarray(3.0))
    assert float(x) == 4.0 and float(metrics["loss"]) == 6.0
    x, metrics = run_if(jnp.asarray(False), step, jnp.asarray(3.0))
    assert float(x) == 3.0 and np.isnan(metrics["loss"])
    assert int(metrics["count"]) == -1


@struct.dataclass
class ActorState(Replaceable):
    value: jax.Array
    last_actor_metrics: Any = None


def test_a_skipped_actor_step_logs_the_last_step_that_ran() -> None:
    """run_actor_if: NaN before the actor's first step, then each skipped
    update logs the last real step's metrics, which the state keeps; with
    no gate (the Python True) the step runs and nothing is kept."""
    state, logged = ActorState(value=jnp.asarray(1.0)), []
    for due in (False, True, False, True):
        state, metrics = run_actor_if(jnp.asarray(due), _update, state)
        logged.append(float(metrics.loss))
    assert np.isnan(logged[0]) and logged[1:] == [1.0, 1.0, 2.0]
    assert float(state.value) == 3.0 and float(state.last_actor_metrics.loss) == 2.0
    ungated, metrics = run_actor_if(True, _update, ActorState(value=jnp.asarray(1.0)))
    assert ungated.last_actor_metrics is None and float(metrics.loss) == 1.0


# --- Evaluations without a logging backend -----------------------------------
def _config(log_frequency: int = 10, **kw: Any) -> LoggingConfig:
    return LoggingConfig(config={}, log_frequency=log_frequency, use_wandb=False, **kw)


def test_the_logging_worker_starts_only_for_a_backend() -> None:
    with mock.patch.object(loop_module, "start_async_logging") as start:
        TrainLoop.create(None, 64, 1)  # type: ignore[arg-type]
        TrainLoop.create(None, 64, 1, logging_config=_config())  # type: ignore[arg-type]
        start.assert_not_called()
        TrainLoop.create(None, 64, 1, logging_config=_config(use_tensorboard=True))  # type: ignore[arg-type]
        start.assert_called_once()


@dataclass(frozen=True)
class Counted(CounterExt):
    """``CounterExt`` reporting its (integer) count at every evaluation."""

    def eval_metrics(self, agent_state, ext_state, rng, context):
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
    # The run continues from 80 (the first call's extra rollout) to its
    # horizon 80 + 64: it evaluates there, in rows 144 // 10 - 80 // 10.
    np.testing.assert_array_equal(resumed.collector_state.timestep, 2 * 80)
    logged = [96, 112, 128, 144, -1, -1]
    np.testing.assert_array_equal(resumed_rows["timestep"], [logged] * len(SEEDS))


def test_a_rollout_past_the_calls_end_evaluates_before_the_horizon() -> None:
    """Told the run ends at 144, a call of 64 steps evaluates at its extra
    rollout's 80 (a 16-step frequency), as the run's one call would, and
    returns a row for it."""
    _, rows = _apo().train(
        seed=SEEDS,
        n_timesteps=BUDGET,
        num_episode_test=1,
        logging_config=_config(log_frequency=16),
        total_timesteps=144,
    )
    logged = [16, 32, 48, 64, 80]
    np.testing.assert_array_equal(rows["timestep"], [logged] * len(SEEDS))


class _HalfTorque:
    """A stateful expert (SAC's convention)."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return jnp.clip(-0.5 * obs[..., :1], -1.0, 1.0), state


def _expert_rows(normalize_observations: bool) -> tuple[Any, Any, np.ndarray]:
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        expert_policy=_HalfTorque(),
        normalize_observations=normalize_observations,
    )
    state, rows = agent.train(
        seed=SEEDS, n_timesteps=BUDGET, num_episode_test=2, logging_config=_config()
    )
    return agent, state, np.asarray(rows["Eval/episodic mean expert reward"])


@pytest.mark.parametrize("normalize", [False, True], ids=["raw", "normalised"])
def test_each_evaluation_reports_the_experts_return_on_its_episodes(
    normalize: bool,
) -> None:
    """Every evaluation reports the return the expert's rollout in
    ``evaluate`` earns on the agent's episodes (its key), an observation
    normaliser or not (the expert acts on the environment's own
    observations)."""
    agent, state, rows = _expert_rows(normalize_observations=normalize)
    for seed in range(len(SEEDS)):
        actor_state = jax.tree.map(lambda x, s=seed: x[s], state.actor_state)
        for n_logs in range(4):
            key = jax.random.fold_in(state.eval_rng[seed], n_logs)
            expert_return = evaluate(
                agent.env_args.env,
                actor_state,
                2,
                jax.random.split(key)[0],
                agent.env_args.env_params,
                expert_policy=agent.expert_policy,
            )[-1]
            np.testing.assert_allclose(
                rows[seed, n_logs], float(expert_return), rtol=1e-6
            )
    assert len(set(rows[0, :4])) == 4  # fresh episodes at each evaluation


# --- Agents acting, training and evaluating their own way --------------------
@struct.dataclass
class Logged(State):
    """A stand-in state that evaluates: an evaluation key and a log count."""

    eval_rng: jax.Array = struct.field(default_factory=lambda: jax.random.PRNGKey(1))
    n_logs: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))


def test_repeat_update_folds_post_update_after_each_update() -> None:
    """``n`` updates, each followed by ``post_update``; none for ``n = 0``."""
    state, loop = _state(5, (jnp.asarray(0),)), _loop(CounterExt())
    run = jax.jit(lambda s, n: loop.repeat_update(s, n, _update))
    out, same = run(state, 3), run(state, 0)
    assert float(out.value) == 4.0 and int(out.ext_state[0]) == 3
    assert float(same.value) == 1.0 and int(same.ext_state[0]) == 0


def test_a_per_seed_count_runs_each_seed_its_own_updates() -> None:
    """A count read from the per-seed state (batched under the seed ``vmap``)
    runs each seed exactly its own number of updates and ``post_update``s."""
    loop = _loop(CounterExt())
    run = jax.vmap(lambda s, n: loop.repeat_update(s, n, _update))
    states = jax.vmap(lambda v: _state(0, (jnp.asarray(0),)))(jnp.arange(3))
    out = run(states, jnp.array([0, 2, 5]))
    np.testing.assert_array_equal(out.value, [1.0, 3.0, 6.0])
    np.testing.assert_array_equal(out.ext_state[0], [0, 2, 5])


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
    assert int(metrics.get("count", -1)) == 5 and int(logged.n_logs) == 1
    assert float(logged.value) == 0.0
    # Rows for every evaluation a call of 5 ticks can make; none without.
    assert evaluation.count(5) == 3
    assert dataclasses.replace(evaluation, every=None).count(5) == 0
