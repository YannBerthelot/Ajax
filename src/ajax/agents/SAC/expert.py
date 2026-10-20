"""SAC's expert plumbing outside the Extension phases: the expert replay
prefill (observations are augmented by ``interaction.augment_obs``)."""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from ajax.environments.interaction import collect_experience_from_expert_policy
from ajax.state import EnvironmentConfig
from ajax.types import BufferType


def collect_and_store_expert_transitions(
    expert_policy: Callable,
    env_args: EnvironmentConfig,
    buffer: BufferType,
    buffer_state: Any,
    rng: jax.Array,
    n_steps: int,
    augment_obs_with_expert_state: bool = False,
    with_train_frac: bool = False,
) -> Any:
    """Collect ``n_steps`` expert steps per env and store them in the replay
    buffer row by row as the live collector does: one step of every env
    per add (the buffer's add batch is the env axis), with the expert's
    actions at s and s' as ``a_expert`` and ``next_a_expert``, and the
    collector's observation layout (the training fraction, 0 before
    training, then the expert's state)."""
    mode = env_args.mode
    transitions = collect_experience_from_expert_policy(
        expert_policy=expert_policy,
        rng=rng,
        mode=mode,
        env_args=env_args,
        n_timesteps=n_steps,
        augment_obs_with_expert_state=augment_obs_with_expert_state,
        train_frac=0.0 if with_train_frac else None,
    )
    schema = buffer_state.experience

    def add_step(buffer_state: Any, step: Any) -> tuple[Any, None]:
        # The expert collector's next observations are the final ones at
        # every end, the live collector's (bootstrap_obs) at time limits
        # only: they differ at terminations alone, which SAC's target masks.
        row = {
            "obs": step.obs,
            "action": step.action,
            "reward": step.reward,
            "terminated": step.terminated,
            "truncated": step.truncated,
            "next_obs": step.next_obs,
            "raw_obs": step.raw_obs,
            "is_expert": jnp.ones_like(step.reward),
            "a_expert": step.a_expert,
            "next_a_expert": step.next_a_expert,
            "raw_next_obs": step.raw_next_obs,
            # Unknown, as the live collector's: no Cal-QL floor on the row.
            "return_to_go": jnp.full_like(step.reward, -jnp.inf),
        }
        # The fields the buffer holds (init_buffer).
        row = {k: v.astype(schema[k].dtype) for k, v in row.items() if k in schema}
        return buffer.add(buffer_state, row), None

    buffer_state, _ = jax.lax.scan(add_step, buffer_state, transitions)
    return buffer_state
