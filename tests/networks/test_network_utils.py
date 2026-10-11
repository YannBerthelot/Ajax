import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.linen.initializers import orthogonal, variance_scaling
from flax.serialization import to_state_dict

from ajax.networks.utils import (
    get_adam_tx,
    get_initializer,
    parse_activation,
    parse_function_string,
    parse_initialization,
    trunc_normal_fan_in,
)
from ajax.state import OptimizerConfig


def _one_step(tx, params, grads):
    state = tx.init(params)
    updates, _ = tx.update(grads, state, params)
    return jax.tree.map(lambda p, u: p + u, params, updates)


def test_zero_weight_decay_is_plain_adam():
    params = {"w": jnp.ones(3)}
    grads = {"w": jnp.zeros(3)}
    new = _one_step(get_adam_tx(1e-2), params, grads)
    assert jnp.allclose(new["w"], 1.0)  # zero grad, no decay -> unchanged


def test_positive_weight_decay_shrinks_params_with_zero_grad():
    params = {"w": jnp.ones(3)}
    grads = {"w": jnp.zeros(3)}
    new = _one_step(get_adam_tx(1e-2, weight_decay=0.1), params, grads)
    assert jnp.allclose(new["w"], 1.0 - 1e-2 * 0.1)


def test_weight_decay_composes_with_clipping_and_rejects_negative():
    tx = get_adam_tx(1e-2, max_grad_norm=0.5, clipped=True, weight_decay=0.1)
    params = {"w": jnp.ones(3)}
    new = _one_step(tx, params, {"w": jnp.full(3, 100.0)})
    assert jnp.all(new["w"] < 1.0) and jnp.all(jnp.isfinite(new["w"]))
    with pytest.raises(ValueError, match="non-negative"):
        get_adam_tx(1e-2, weight_decay=-1.0)


def test_optimizer_config_round_trips_into_get_adam_tx():
    cfg = OptimizerConfig(learning_rate=1e-3, weight_decay=0.05)
    tx = get_adam_tx(**to_state_dict(cfg))
    new = _one_step(tx, {"w": jnp.ones(2)}, {"w": jnp.zeros(2)})
    assert jnp.allclose(new["w"], 1.0 - 1e-3 * 0.05)


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
        # Malformed specs must not silently fall back to the default scale.
        "orthogonal(2.0",
        "orthogonal(2.0)junk",
        "lecun_uniform extra",
        "np.sqrt(2)",
    ],
)
def test_parse_function_string_invalid(input_str):
    with pytest.raises(ValueError):
        parse_function_string(input_str)


# ------------------------
# Tests for parse_initialization
# ------------------------


def test_parse_initialization_constant_with_value():
    out = parse_initialization("constant(2.5)")(
        jax.random.PRNGKey(0), (3, 4), jnp.float32
    )
    np.testing.assert_array_equal(out, 2.5)


def test_parse_initialization_orthogonal():
    """Orthonormal columns: ``W^T W = I``, as flax's ``orthogonal()``."""
    key, shape = jax.random.PRNGKey(0), (8, 4)
    w = parse_initialization("orthogonal")(key, shape, jnp.float32)
    np.testing.assert_allclose(w.T @ w, np.eye(4), atol=1e-5)
    np.testing.assert_array_equal(w, orthogonal()(key, shape, jnp.float32))


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


def test_parse_initialization_of_none_is_the_default():
    default = orthogonal(1.0)
    assert parse_initialization(None, default=default) is default
    assert parse_initialization("zeros", default=default) is not default
    with pytest.raises(ValueError, match="no default"):
        parse_initialization(None)


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
    with pytest.raises(ValueError, match="'mish', 'relu'"):  # it lists the names
        parse_activation(activation)
