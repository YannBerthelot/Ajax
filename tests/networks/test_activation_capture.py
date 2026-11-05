"""Reading a network's hidden layers through flax's ``capture_intermediates``.

Contract (ajax.networks.utils.Activation, and capture_intermediates' own
definition: a module's output): an architecture's activation is a layer of
its own, without parameters, so a capture reports each hidden layer after
its nonlinearity, and an ensemble critic reports one row per member. The
expected values are worked out from the kernels, with NumPy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from ajax.networks.networks import Encoder, MultiCritic
from ajax.networks.utils import Activation

INPUTS = jnp.array([[1.0, -2.0], [0.5, 0.25], [-1.5, 0.75]])


def is_activation(module, method_name):
    del method_name
    return isinstance(module, Activation)


def rectified(kernel, bias):
    return np.maximum(np.asarray(INPUTS) @ np.asarray(kernel) + np.asarray(bias), 0.0)


def test_capture_reads_a_hidden_layer_after_its_nonlinearity():
    """Catches a layer read before its activation: a ReLU layer reports
    negative values, and no unit ever reads as dead."""
    encoder = Encoder(input_architecture=("3", "relu"))
    variables = encoder.init(jax.random.PRNGKey(0), INPUTS)
    dense = variables["params"]["network"]["layers_0"]

    _, captured = encoder.apply(
        variables,
        INPUTS,
        capture_intermediates=is_activation,
        mutable=["intermediates"],
    )

    expected = rectified(dense["kernel"], dense["bias"])
    assert (expected == 0).any(), "the inputs must rectify some unit to show it"
    outputs = captured["intermediates"]["network"]["layers_1"]["__call__"][0]
    np.testing.assert_allclose(outputs, expected, rtol=1e-6)


def test_an_activation_keeps_the_parameter_names_of_its_network():
    """Catches an activation layer taking parameters or moving its
    neighbours' names, which would break every saved checkpoint."""
    encoder = Encoder(input_architecture=("4", "tanh", "4", "tanh"))
    variables = encoder.init(jax.random.PRNGKey(0), INPUTS)

    names = sorted("/".join(path) for path in flatten_dict(variables["params"]))

    assert names == [
        "network/layers_0/bias",
        "network/layers_0/kernel",
        "network/layers_2/bias",
        "network/layers_2/kernel",
    ]


def test_an_ensemble_capture_keeps_each_members_activations():
    """Catches the members' activations shared or swapped: every critic of
    an ensemble would then report the first one's representation."""
    critic = MultiCritic(input_architecture=("3", "relu"), num=2)
    variables = critic.init(jax.random.PRNGKey(0), INPUTS)
    dense = variables["params"]["ensemble"]["encoder"]["network"]["layers_0"]

    _, captured = critic.apply(
        variables,
        INPUTS,
        capture_intermediates=is_activation,
        mutable=["intermediates"],
    )

    tree = captured["intermediates"]["ensemble"]["encoder"]["network"]
    outputs = tree["layers_1"]["__call__"][0]
    assert outputs.shape == (2, 3, 3)
    for member in range(2):
        expected = rectified(dense["kernel"][member], dense["bias"][member])
        np.testing.assert_allclose(outputs[member], expected, rtol=1e-6)
