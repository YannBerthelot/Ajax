"""The training-fraction view of an environment state (``load_state``)."""

import jax
from gymnax.environments.classic_control.pendulum import Pendulum

from ajax.state import load_state


def test_a_missing_attribute_of_the_loaded_state_raises_attribute_error():
    """The view copies the state's fields and adds the fraction; an attribute
    neither has raises AttributeError, so getattr's default applies (the
    evaluation's logging asks for the normaliser's statistics this way)."""
    env = Pendulum()
    _, state = env.reset(jax.random.PRNGKey(0), env.default_params)
    loaded = load_state(state, train_time_fraction=0.25)
    assert loaded.theta == state.theta and loaded.train_time_fraction == 0.25
    assert loaded.linear_schedule == 0.75
    assert getattr(loaded, "normalization_info", None) is None
