import jax
import jax.numpy as jnp
import optax
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
    # TODO : switch to KL div
    return loss, None  # placeholder aux


@partial(jax.jit, static_argnames=("num_epochs", "mode", "distillation_lr"))
def distillation(
    student_critic_state: LoadedTrainState,
    teacher_values: jnp.ndarray,
    student_inputs: jnp.ndarray,
    mode: str,
    num_epochs: int = 1,
    distillation_lr: float = 1e-4,
) -> LoadedTrainState:
    """
    The student_inputs have to already be normalized if the student expectes normalization.
    """
    # Temporary optimizer just for distillation
    tx = optax.adam(distillation_lr)
    opt_state = tx.init(student_critic_state.params)

    loss_fn = (
        value_distillation_loss_function
        if mode == "critic"
        else actor_distillation_loss_function
    )

    def distill_epoch(carry, _):
        params, opt_state = carry

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, student_critic_state, teacher_values, student_inputs
        )

        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), None

    (final_params, _), _ = jax.lax.scan(
        f=distill_epoch,
        init=(student_critic_state.params, opt_state),
        xs=None,
        length=num_epochs,
    )
    return student_critic_state.replace(params=final_params)
