import inspect

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.linen.initializers import constant, orthogonal, variance_scaling

from ajax.networks.utils import (
    get_adam_tx,
    get_initializer,
    parse_activation,
    parse_function_string,
    parse_initialization,
    trunc_normal_fan_in,
)


def test_get_adam_tx_without_clipping():
    """Test get_adam_tx without gradient clipping."""
    tx = get_adam_tx(learning_rate=0.001, clipped=False)
    assert isinstance(tx, optax.GradientTransformationExtraArgs)


def test_get_adam_tx_with_clipping():
    """Test get_adam_tx with gradient clipping."""
    tx = get_adam_tx(learning_rate=0.001, max_grad_norm=0.5, clipped=True)
    assert isinstance(tx, optax.GradientTransformationExtraArgs)


def test_get_adam_tx_clipping_without_norm():
    """Test get_adam_tx raises ValueError when clipping is requested without max_grad_norm."""
    with pytest.raises(
        ValueError, match="Gradient clipping requested but no norm provided."
    ):
        get_adam_tx(learning_rate=0.001, max_grad_norm=None, clipped=True)


# ------------------------
# Tests for parse_function_string
# ------------------------


@pytest.mark.parametrize(
    "input_str,expected_name,expected_value",
    [
        ("constant(3)", "constant", 3),
        ("uniform(3.5)", "uniform", 3.5),
        ("foo(math.sqrt(4))", "foo", 2.0),
        ("bar(np.log(1))", "bar", 0.0),
        ("baz(jnp.sqrt(9))", "baz", 3.0),
        ("no_arg()", "no_arg", None),
        ("onlyname", "onlyname", None),
        ("0.5", None, 0.5),
        ("1", None, 1),
        ("-1.0", None, -1.0),
    ],
)
def test_parse_function_string_valid(input_str, expected_name, expected_value):
    name, value = parse_function_string(input_str)
    assert name == expected_name
    if expected_value is None:
        assert value is None
    else:
        assert pytest.approx(value) == expected_value


@pytest.mark.parametrize(
    "input_str",
    [
        "func(unknown_var)",
    ],
)
def test_parse_function_string_invalid(input_str):
    with pytest.raises(ValueError):
        parse_function_string(input_str)


# ------------------------
# Tests for parse_initialization
# ------------------------


def test_parse_initialization_constant_with_value():
    # Parse the string and get back the init function
    init_fn = parse_initialization("constant(2.5)")

    # The flax constant initializer is actually a closure capturing 'val'
    # Use inspect.getclosurevars to grab that captured value
    closure_vars = inspect.getclosurevars(init_fn)
    # Check that:
    # 1) the function name matches flax's inner function name
    assert init_fn.__name__ == constant(2.5).__name__

    # 2) the captured 'val' in the closure is 2.5
    assert closure_vars.nonlocals.get("value") == pytest.approx(2.5)


def test_parse_initialization_orthogonal():
    # Parse the string and get back the init function
    init_fn = parse_initialization("orthogonal")
    assert init_fn.__name__ == orthogonal().__name__


def test_parse_initialization_constant_without_value():
    # Parse the string and get back the init function
    with pytest.raises(TypeError):
        parse_initialization("constant")


def test_parse_initialization_invalid_name():
    with pytest.raises(ValueError):
        parse_initialization("unknown_init(1.0)")


def test_parse_initialization_zeros():
    # flax exposes ``zeros`` as an initializer, not a factory: Ajax's registry wraps it.
    init_fn = parse_initialization("zeros")
    out = init_fn(jax.random.PRNGKey(0), (3, 4), jnp.float32)
    np.testing.assert_array_equal(out, 0.0)


@pytest.mark.parametrize("spec, value", [("0.5", 0.5), ("1", 1.0), ("-1.0", -1.0)])
def test_parse_initialization_number_is_a_constant(spec, value):
    out = parse_initialization(spec)(jax.random.PRNGKey(0), (2, 3), jnp.float32)
    np.testing.assert_array_equal(out, value)


def test_parse_initialization_passes_callables_through():
    init_fn = orthogonal(1.0)
    assert parse_initialization(init_fn) is init_fn


def test_parse_initialization_trunc_normal_fan_in():
    key, shape = jax.random.PRNGKey(0), (64, 32)
    expected = variance_scaling(1.0, "fan_in", "truncated_normal")(key, shape)
    np.testing.assert_array_equal(
        parse_initialization("trunc_normal_fan_in")(key, shape), expected
    )
    scaled = variance_scaling(0.25, "fan_in", "truncated_normal")(key, shape)
    np.testing.assert_array_equal(
        parse_initialization("trunc_normal_fan_in(0.5)")(key, shape), scaled
    )
    np.testing.assert_array_equal(
        parse_initialization("trunc_normal_fan_in(0)")(key, shape), 0.0
    )
    with pytest.raises(ValueError, match=">= 0"):
        trunc_normal_fan_in(-1.0)


def test_get_initializer_lists_the_registry_on_unknown_names():
    with pytest.raises(ValueError, match="trunc_normal_fan_in"):
        get_initializer("no_such_initializer")


# ------------------------
# Tests for parse_activation
# ------------------------


@pytest.mark.parametrize(
    "name, at_one, at_minus_one",
    [("silu", 0.7310586, -0.2689414), ("mish", 0.8650984, -0.3034015)],
)
def test_parse_activation_world_model_activations(name, at_one, at_minus_one):
    act = parse_activation(name)
    np.testing.assert_allclose(
        act(jnp.array([1.0, -1.0])), [at_one, at_minus_one], rtol=1e-6
    )


def test_parse_activation_passes_callables_through():
    assert parse_activation(jax.nn.relu) is jax.nn.relu


@pytest.mark.parametrize("activation", ["no_such_activation", 3])
def test_parse_activation_rejects_unknown(activation):
    with pytest.raises(ValueError):
        parse_activation(activation)
