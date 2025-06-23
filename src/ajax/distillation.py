from typing import Any

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax.tree_util import Partial as partial

from ajax.state import LoadedTrainState


def value_distillation_loss_function(
    student_params: FrozenDict,
    student_critic_state: LoadedTrainState,
    teacher_values: jnp.ndarray,
    student_inputs: jnp.ndarray,
):
    """
    The student_inputs have to already be normalized if the student expectes normalization.
    """
    student_values = student_critic_state.apply_fn(
        student_params, student_inputs, mutable=False
    )
    loss = jnp.mean((student_values - teacher_values) ** 2)
    return loss, None  # placeholder aux


def actor_distillation_loss_function(
    student_params: FrozenDict,
    student_critic_state: LoadedTrainState,
    teacher_values: jnp.ndarray,
    student_inputs: jnp.ndarray,
):
    """
    The student_inputs have to already be normalized if the student expectes normalization.
    """
    student_values = student_critic_state.apply_fn(
        student_params, student_inputs, mutable=False
    )
    loss_mean = jnp.mean((student_values.mean() - teacher_values.mean()) ** 2)
    loss_std = jnp.mean((student_values.mean() - teacher_values.mean()) ** 2)
    loss = loss_mean + loss_std
    return loss, None  # placeholder aux


@partial(jax.jit, static_argnames="num_epochs")
def value_distillation(
    student_critic_state: LoadedTrainState,
    teacher_values: jnp.ndarray,
    student_inputs: jnp.ndarray,
    num_epochs: int = 1,
) -> LoadedTrainState:
    """
    The student_inputs have to already be normalized if the student expectes normalization.
    """

    def value_distillation_epoch(student_critic_state: LoadedTrainState, _: Any):
        (loss, aux), grads = jax.value_and_grad(
            value_distillation_loss_function, has_aux=True
        )(
            student_critic_state.params,
            student_critic_state,
            teacher_values,
            student_inputs,
        )
        # jax.debug.print("value loss {x}", x=loss)
        return student_critic_state.apply_gradients(grads=grads), None

    new_student_critic_state, _ = jax.lax.scan(
        f=value_distillation_epoch,
        init=student_critic_state,
        xs=None,
        length=num_epochs,
    )
    return new_student_critic_state


@partial(jax.jit, static_argnames="num_epochs")
def policy_distillation(
    student_critic_state: LoadedTrainState,
    teacher_values: jnp.ndarray,
    student_inputs: jnp.ndarray,
    num_epochs: int = 1,
    actor: bool = False,
) -> LoadedTrainState:
    """
    The student_inputs have to already be normalized if the student expectes normalization.
    """

    def actor_distillation_epoch(student_critic_state: LoadedTrainState, _: Any):
        (loss, aux), grads = jax.value_and_grad(
            actor_distillation_loss_function, has_aux=True
        )(
            student_critic_state.params,
            student_critic_state,
            teacher_values,
            student_inputs,
        )
        # jax.debug.print("policy loss {x}", x=loss)
        return student_critic_state.apply_gradients(grads=grads), None

    new_student_critic_state, _ = jax.lax.scan(
        f=actor_distillation_epoch,
        init=student_critic_state,
        xs=None,
        length=num_epochs,
    )
    return new_student_critic_state
