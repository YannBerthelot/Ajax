import jax
import jax.numpy as jnp
import pytest
from flax.core import FrozenDict

from ajax.distillation import value_distillation_loss_function
from ajax.networks.networks import Actor, Critic
from ajax.networks.utils import get_adam_tx
from ajax.state import LoadedTrainState


def compare_frozen_dicts(dict1: FrozenDict, dict2: FrozenDict) -> bool:
    """
    Compares two FrozenDicts to check if they are equal.

    Args:
        dict1 (FrozenDict): The first FrozenDict.
        dict2 (FrozenDict): The second FrozenDict.

    Returns:
        bool: True if the FrozenDicts are equal, False otherwise.
    """
    for key in dict1.keys():
        if key not in dict2:
            return False
        value1, value2 = dict1[key], dict2[key]
        if isinstance(value1, FrozenDict) and isinstance(value2, FrozenDict):
            if not compare_frozen_dicts(value1, value2):
                return False
        elif not jnp.allclose(value1, value2):
            return False
    return True


@pytest.fixture
def sample_input():
    return jnp.ones((4, 10))  # Example input with batch size 4 and feature size 10


@pytest.fixture
def actor_architecture():
    return ["16", "relu"]


@pytest.fixture
def critic_architecture():
    return ["16", "relu"]


@pytest.fixture
def teacher_actor(actor_architecture):
    actor = Actor(
        input_architecture=actor_architecture,
        action_dim=4,
        continuous=True,
    )
    params = actor.init(jax.random.PRNGKey(0), jnp.ones((4, 8)))
    return LoadedTrainState.create(
        params=FrozenDict(params),
        tx=None,
        apply_fn=actor.apply,
    )


@pytest.fixture
def student_actor(actor_architecture):
    actor = Actor(
        input_architecture=actor_architecture,
        action_dim=4,
        continuous=True,
    )
    params = actor.init(jax.random.PRNGKey(1), jnp.ones((4, 8)))
    return LoadedTrainState.create(
        params=FrozenDict(params),
        tx=None,
        apply_fn=actor.apply,
    )


@pytest.fixture
def teacher_critic(critic_architecture):
    critic = Critic(input_architecture=critic_architecture)
    params = critic.init(jax.random.PRNGKey(0), jnp.ones((4, 10)))
    tx = get_adam_tx(learning_rate=1e-3)
    return LoadedTrainState.create(
        params=FrozenDict(params),
        tx=tx,
        apply_fn=critic.apply,
    )


@pytest.fixture
def student_critic(critic_architecture):
    critic = Critic(input_architecture=critic_architecture)
    params = critic.init(jax.random.PRNGKey(1), jnp.ones((4, 10)))
    tx = get_adam_tx(learning_rate=1e-3)
    return LoadedTrainState.create(
        params=FrozenDict(params),
        tx=tx,
        apply_fn=critic.apply,
    )


@pytest.fixture
def teacher_values():
    return jnp.array([[1.0], [2.0], [3.0], [4.0]])  # Example teacher values


def test_value_distillation_loss_function(student_critic, teacher_values, sample_input):
    loss, _ = value_distillation_loss_function(
        student_params=student_critic.params,
        student_critic_state=student_critic,
        teacher_values=teacher_values,
        student_inputs=sample_input,
    )

    assert jnp.isfinite(loss), "Loss should be a finite value."
    assert loss > 0, "Loss should be positive."


# def test_value_distillation(student_critic, teacher_values, sample_input):
#     updated_student_critic = value_distillation(
#         student_critic_state=student_critic,
#         teacher_values=teacher_values,
#         student_inputs=sample_input,
#     )

#     assert isinstance(
#         updated_student_critic, LoadedTrainState
#     ), "Student critic state should be updated."
#     assert not compare_frozen_dicts(
#         FrozenDict(updated_student_critic.params), FrozenDict(student_critic.params)
#     ), "Student critic parameters should be updated."
