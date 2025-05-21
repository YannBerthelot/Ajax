import inspect

import optax
import pytest
from flax.linen.initializers import constant, orthogonal

from ajax.networks.utils import get_adam_tx, parse_function_string, parse_initialization


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
    with pytest.raises(ValueError, match="Unrecognized initialization name"):
        parse_initialization("unknown_init(1.0)")
