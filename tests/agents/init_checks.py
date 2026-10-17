"""Checks that a layer starts as an agent's reference code starts it."""

import numpy as np


def assert_torch_linear(name: str, layer: dict, fan_in: int) -> None:
    """torch's (and Torch7's) nn.Linear default: kernel and bias uniform in
    +-1/sqrt(fan_in), so non-zero, the kernel with a uniform's variance."""
    bound = 1 / np.sqrt(fan_in)
    kernel, bias = np.asarray(layer["kernel"]), np.asarray(layer["bias"])
    assert np.abs(kernel).max() <= bound and np.abs(bias).max() <= bound, name
    assert np.abs(kernel).max() > 0.5 * bound and np.any(bias), name
    if kernel.size >= 256:  # enough draws for the variance, bound**2 / 3
        np.testing.assert_allclose(kernel.var(), bound**2 / 3, rtol=0.3, err_msg=name)
