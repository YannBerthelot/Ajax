"""The shared training loop's steps (``ajax.agents.loop``), on a stand-in
state: the update gate before ``learning_starts``, the extensions'
``post_update`` key, one gradient step. The agents' smoke and resume tests
and the probes run the loop end to end."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from ajax.agents.loop import TrainLoop, gradient_step
from ajax.extensions.base import Extension, ExtensionStack


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


def test_gradient_step_applies_the_loss_gradient() -> None:
    params = {"w": jnp.asarray(3.0)}
    train_state = TrainState.create(apply_fn=None, params=params, tx=optax.sgd(0.5))

    def loss_fn(p):
        return p["w"] ** 2, {"w": p["w"]}

    stepped, aux = gradient_step(train_state, loss_fn)
    assert float(stepped.params["w"]) == 3.0 - 0.5 * 6.0
    assert float(aux["w"]) == 3.0 and int(stepped.step) == 1
