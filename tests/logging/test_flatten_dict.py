"""flatten_dict turns nested auxiliaries into one scalar per metric name."""

import jax
import jax.numpy as jnp
import pytest
from flax import struct

from ajax.log import aux_metrics, flatten_dict


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


@struct.dataclass
class _Value:
    critic_loss: jax.Array


@struct.dataclass
class _Logs:
    value: _Value
    extensions: dict


def test_the_extensions_training_metrics_keep_their_own_names():
    """An update's ``extensions`` field is logged by its names, beside the
    groups' ``<group>/<metric>``."""
    logs = _Logs(_Value(jnp.ones((1,))), {"ibrl/expert_wins": jnp.full((1,), 0.5)})
    flat = aux_metrics(logs)
    assert sorted(flat) == ["ibrl/expert_wins", "value/critic_loss"]
    assert (
        flat["ibrl/expert_wins"].shape == () and float(flat["ibrl/expert_wins"]) == 0.5
    )
