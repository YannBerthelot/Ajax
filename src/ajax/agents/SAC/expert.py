"""SAC's expert plumbing outside the Extension phases: observation
augmentation and the expert replay prefill."""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from ajax.environments.interaction import collect_experience_from_expert_policy
from ajax.environments.utils import check_env_is_gymnax, maybe_append_train_frac
from ajax.state import EnvironmentConfig
from ajax.types import BufferType


def insert_expert_action(observations: jax.Array, a_expert: jax.Array) -> jax.Array:
    """The update's augmented layout: [env_obs | a_expert | train_frac]."""
    a_expert = jax.lax.stop_gradient(a_expert)
    return jnp.concatenate(
        [observations[..., :-1], a_expert, observations[..., -1:]], axis=-1
    )


def augment_obs_if_needed(
    observations: jax.Array,
    raw_observations: jax.Array,
    expert_policy,
    augment: bool,
) -> jax.Array:
    """Append the expert's action at ``raw_observations`` to ``observations``
    (:func:`insert_expert_action`)."""
    if not augment or expert_policy is None:
        return observations
    return insert_expert_action(observations, expert_policy(raw_observations))


def collect_and_store_expert_transitions(
    expert_policy: Callable,
    env_args: EnvironmentConfig,
    buffer: BufferType,
    buffer_state: Any,
    rng: jax.Array,
    n_steps: int,
) -> Any:
    """Collect ``n_steps`` expert steps per env and store them in the replay
    buffer row by row as the live collector does: one step of every env
    per add (the buffer's add batch is the env axis), with the expert's
    actions at s and s' as ``a_expert`` and ``next_a_expert``."""
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    transitions = collect_experience_from_expert_policy(
        expert_policy=expert_policy,
        rng=rng,
        mode=mode,
        env_args=env_args,
        n_timesteps=n_steps,
    )
    schema = buffer_state.experience
    widened = schema["obs"].shape[-1] == transitions.obs.shape[-1] + 1
    train_frac = 0.0 if widened else None

    def add_step(buffer_state: Any, step: Any) -> tuple[Any, None]:
        # The expert collector's next observations are the final ones at
        # every end, the live collector's (bootstrap_obs) at time limits
        # only: they differ at terminations alone, which SAC's target masks.
        row = {
            "obs": maybe_append_train_frac(step.obs, train_frac=train_frac),
            "action": step.action,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "next_obs": maybe_append_train_frac(step.next_obs, train_frac=train_frac),
            "raw_obs": step.raw_obs,
            "is_expert": jnp.ones_like(step.reward),
            "a_expert": step.a_expert,
            "next_a_expert": step.next_a_expert,
        }
        row = {k: v.astype(schema[k].dtype) for k, v in row.items()}
        return buffer.add(buffer_state, row), None

    buffer_state, _ = jax.lax.scan(add_step, buffer_state, transitions)
    return buffer_state
