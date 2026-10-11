"""flatten_dict turns nested auxiliaries into one scalar per metric name."""

import jax
import jax.numpy as jnp
import pytest

from ajax.log import flatten_dict


def test_single_element_leaves_become_scalars():
    flat = flatten_dict(
        {
            "value": {"critic_loss": jnp.ones((1,)), "q": jnp.float32(2.0)},
            "timestep": jnp.ones((1,)),
            "name": "sac",
        }
    )
    assert flat["value/critic_loss"].shape == ()
    assert flat["value/q"].shape == ()
    assert flat["timestep"].shape == ()
    assert flat["name"] == "sac"


@pytest.mark.parametrize("traced", [False, True], ids=["eager", "jit"])
def test_a_wider_leaf_raises_instead_of_keeping_its_first_element(traced):
    def flatten(x):
        return flatten_dict({"value": {"critic_loss": x}})

    flatten = jax.jit(flatten) if traced else flatten
    with pytest.raises(ValueError, match="value/critic_loss"):
        flatten(jnp.arange(3.0))
