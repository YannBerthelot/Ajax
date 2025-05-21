import jax.numpy as jnp

from ajax.environments.interaction import init_rolling_mean, update_rolling_mean


def test_init_rolling_mean():
    window_size = 5
    cumulative_reward = jnp.zeros((3, 1))  # Example batch size of 3
    last_return = jnp.zeros((3, 1))

    rolling_mean_state = init_rolling_mean(
        window_size=window_size,
        cumulative_reward=cumulative_reward,
        last_return=last_return,
    )

    assert rolling_mean_state.buffer.shape == (window_size, 3, 1)
    assert jnp.all(
        rolling_mean_state.buffer == 0
    ), "Buffer should be initialized to zeros."
    assert jnp.all(
        rolling_mean_state.index == 0
    ), "Index should be initialized to zeros."
    assert jnp.all(
        rolling_mean_state.count == 0
    ), "Count should be initialized to zeros."
    assert jnp.all(rolling_mean_state.sum == 0), "Sum should be initialized to zeros."
    assert jnp.allclose(
        rolling_mean_state.cumulative_reward, cumulative_reward
    ), "Cumulative reward should match the input."
    assert jnp.allclose(
        rolling_mean_state.last_return, last_return
    ), "Last return should match the input."


def test_update_rolling_mean_within_window():
    window_size = 5
    cumulative_reward = jnp.zeros((3, 1))  # Example batch size of 3
    last_return = jnp.nan * jnp.zeros((3, 1))

    rolling_mean_state = init_rolling_mean(
        window_size=window_size,
        cumulative_reward=cumulative_reward,
        last_return=last_return,
    )

    new_value = jnp.array([[10.0], [20.0], [30.0]])
    updated_state, mean = update_rolling_mean(rolling_mean_state, new_value)

    assert jnp.all(
        updated_state.buffer[0] == new_value
    ), "New value should be added to the buffer."
    assert jnp.all(
        updated_state.index == 1
    ), "Index should be incremented for all batch elements."
    assert jnp.all(
        updated_state.count == 1
    ), "Count should be incremented for all batch elements."
    assert jnp.all(
        updated_state.sum == new_value
    ), "Sum should be updated for all batch elements."
    assert jnp.all(
        mean == new_value
    ), "Mean should equal the new value when count is 1."


def test_update_rolling_mean_overflow_window():
    window_size = 3
    cumulative_reward = jnp.zeros((2, 1))  # Example batch size of 2
    last_return = jnp.nan * jnp.zeros((2, 1))

    rolling_mean_state = init_rolling_mean(
        window_size=window_size,
        cumulative_reward=cumulative_reward,
        last_return=last_return,
    )

    values = [
        jnp.array([[10.0], [20.0]]),
        jnp.array([[30.0], [40.0]]),
        jnp.array([[50.0], [60.0]]),
        jnp.array([[70.0], [80.0]]),
    ]
    for value in values:
        rolling_mean_state, mean = update_rolling_mean(rolling_mean_state, value)

    assert jnp.allclose(
        rolling_mean_state.buffer,
        jnp.array([[[70.0], [80.0]], [[30.0], [40.0]], [[50.0], [60.0]]]),
    ), "Buffer should overwrite old values."
    assert jnp.all(
        rolling_mean_state.index == 1
    ), "Index should wrap around for all batch elements."
    assert jnp.all(
        rolling_mean_state.count == window_size
    ), "Count should not exceed window size."
    assert jnp.all(
        rolling_mean_state.sum == jnp.array([[150.0], [180.0]])
    ), "Sum should reflect the current buffer values."
    assert jnp.all(
        mean == jnp.array([[50.0], [60.0]])
    ), "Mean should be the average of the current buffer values."


def test_update_rolling_mean_partial_window():
    window_size = 5
    cumulative_reward = jnp.zeros((2, 1))  # Example batch size of 2
    last_return = jnp.nan * jnp.zeros((2, 1))

    rolling_mean_state = init_rolling_mean(
        window_size=window_size,
        cumulative_reward=cumulative_reward,
        last_return=last_return,
    )

    values = [jnp.array([[10.0], [20.0]]), jnp.array([[30.0], [40.0]])]
    for value in values:
        rolling_mean_state, mean = update_rolling_mean(rolling_mean_state, value)

    assert jnp.allclose(
        rolling_mean_state.buffer[:2],
        jnp.array([[[10.0], [20.0]], [[30.0], [40.0]]]),
    ), "Buffer should contain the added values."
    assert jnp.all(
        rolling_mean_state.index == 2
    ), "Index should reflect the number of added values."
    assert jnp.all(
        rolling_mean_state.count == 2
    ), "Count should reflect the number of added values."
    assert jnp.all(
        rolling_mean_state.sum == jnp.array([[40.0], [60.0]])
    ), "Sum should reflect the total of added values."
    assert jnp.all(
        mean == jnp.array([[20.0], [30.0]])
    ), "Mean should be the average of the added values."
