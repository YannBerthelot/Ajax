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
    vmap: bool = True,  # just for debug purposes, TODO : remove later
):
    """
    The student_inputs have to already be normalized if the student expectes normalization.
    """
    student_values = student_critic_state.apply_fn(
        student_params, student_inputs, mutable=False
    )
    teacher_values = (
        jnp.min(teacher_values, axis=0, keepdims=True)
        if jnp.ndim(teacher_values) > 2
        else teacher_values
    )

    loss = jnp.mean((student_values - teacher_values) ** 2)

    return loss, None  # placeholder aux


def actor_distillation_loss_function(
    student_params: FrozenDict,
    student_critic_state: LoadedTrainState,
    teacher_values: jnp.ndarray,
    student_inputs: jnp.ndarray,
    vmap: bool = True,
):
    """
    The student_inputs have to already be normalized if the student expectes normalization.
    """
    student_values = student_critic_state.apply_fn(
        student_params, student_inputs, mutable=False
    )
    # loss_mean = jnp.mean((student_values.mean() - teacher_values.mean()) ** 2)
    # loss_std = jnp.mean((student_values.mean() - teacher_values.mean()) ** 2)

    # kl_div = _kl_divergence_transformed_transformed(
    #     student_values, teacher_values
    # ).mean()

    wasserstein = (
        (teacher_values.unsquashed_mean() - student_values.unsquashed_mean()) ** 2
        + (teacher_values.unsquashed_stddev() - student_values.unsquashed_stddev()) ** 2
    ).mean()

    # loss = loss_mean + loss_std
    loss = wasserstein
    # TODO : switch to KL div

    return loss, None  # placeholder aux


@partial(jax.jit, static_argnames=("num_epochs", "mode", "distillation_lr"))
def distillation(
    student: LoadedTrainState,
    teacher_values: jnp.ndarray,
    student_inputs: jnp.ndarray,
    mode: str,
    num_epochs: int = 1,
    distillation_lr: float = 1e-4,
    vmap: bool = False,
) -> tuple[LoadedTrainState, jnp.ndarray]:
    """
    The student_inputs have to already be normalized if the student expectes normalization.
    """
    # Temporary optimizer just for distillation
    tx = optax.adam(distillation_lr)
    opt_state = tx.init(student.params)

    loss_fn = (
        value_distillation_loss_function
        if mode == "critic"
        else actor_distillation_loss_function
    )

    def distill_epoch(carry, _):
        params, opt_state = carry

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, student, teacher_values, student_inputs, vmap
        )

        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), loss

    (final_params, _), final_loss = jax.lax.scan(
        f=distill_epoch,
        init=(student.params, opt_state),
        xs=None,
        length=num_epochs,
    )

    final_loss = final_loss[-1]
    return student.replace(params=final_params), final_loss
