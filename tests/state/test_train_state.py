import jax.numpy as jnp
import optax

from ajax.state import LoadedTrainState

PARAMS = {"weights": jnp.array([1.0, 2.0])}
TARGET_PARAMS = {"weights": jnp.array([0.8, 1.8])}
TX = optax.adam(learning_rate=0.001)


def test_loaded_train_state_initialization():
    """create() keeps the params, and sets ``recurrent`` from whether a
    hidden state is given."""
    state = LoadedTrainState.create(apply_fn=lambda x: x, params=PARAMS, tx=TX)
    assert state.hidden_state is None
    assert state.target_params is None
    assert state.params == PARAMS
    assert not state.recurrent

    hidden_state = jnp.array([0.5, 0.5])
    state_with_hidden = LoadedTrainState.create(
        apply_fn=lambda x: x,
        params=PARAMS,
        tx=TX,
        hidden_state=hidden_state,
        target_params=TARGET_PARAMS,
    )
    assert jnp.array_equal(state_with_hidden.hidden_state, hidden_state)
    assert state_with_hidden.target_params == TARGET_PARAMS
    assert state_with_hidden.recurrent


def test_loaded_train_state_soft_update():
    """Polyak averaging: tau of the params, 1 - tau of the target."""
    state = LoadedTrainState.create(
        apply_fn=lambda x: x, params=PARAMS, tx=TX, target_params=TARGET_PARAMS
    )
    updated_state = state.soft_update(tau=0.5)
    assert jnp.allclose(updated_state.target_params["weights"], jnp.array([0.9, 1.9]))
