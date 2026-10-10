"""SAC's expert plumbing outside the Extension phases: observation
augmentation, expert diagnostics, the expert replay prefill and the
Bellman critic pretraining (``use_bellman_critic_pretrain``)."""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import collect_experience_from_expert_policy
from ajax.environments.utils import check_env_is_gymnax, maybe_append_train_frac
from ajax.networks.networks import predict_value
from ajax.state import EnvironmentConfig
from ajax.types import BufferType


def augment_obs_if_needed(
    observations: jax.Array,
    raw_observations: jax.Array,
    expert_policy,
    augment: bool,
) -> jax.Array:
    """Append expert action to obs at runtime. Layout: [env_obs | a_expert | train_frac]."""
    if not augment or expert_policy is None:
        return observations
    a_expert = jax.lax.stop_gradient(expert_policy(raw_observations))
    return jnp.concatenate(
        [observations[..., :-1], a_expert, observations[..., -1:]], axis=-1
    )


def compute_expert_diagnostics(
    critic_state,
    observations: jax.Array,
    q_min: jax.Array,
    a_expert: jax.Array,
    pi_loc: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Expert policy diagnostics: Q(s, a_expert), ||pi - a_expert||^2, above-expert fraction.

    Returns (q_expert_mean, l2_expert_mean, above_expert_frac).
    """
    q_expert = jnp.min(
        predict_value(
            critic_state=critic_state,
            critic_params=critic_state.params,
            x=jnp.concatenate([observations, a_expert], axis=-1),
        ),
        axis=0,
    )
    above_expert_frac = jnp.mean((q_min >= q_expert).astype(jnp.float32))
    l2_expert = jnp.sum((jnp.tanh(pi_loc) - a_expert) ** 2, axis=-1, keepdims=True)
    return q_expert.mean(), l2_expert.mean(), above_expert_frac


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


@partial(
    jax.jit,
    static_argnames=[
        "recurrent",
        "gamma",
        "reward_scale",
        "n_steps",
        "buffer",
        "update_value_fn",
        "update_target_fn",
    ],
)
def pretrain_critic_bellman(
    agent_state,
    recurrent: bool,
    gamma: float,
    reward_scale: float,
    buffer: BufferType,
    n_steps: int = 5_000,
    update_value_fn: Optional[Callable] = None,
    update_target_fn: Optional[Callable] = None,
):
    """Bellman-bootstrapped critic pretraining on the expert buffer.

    Requires update_value_fn and update_target_fn to be passed in,
    avoiding circular imports with train_SAC.
    """

    def critic_pretrain_step(carry, _):
        agent_state = carry
        sample_key, rng = jax.random.split(agent_state.rng)
        agent_state = agent_state.replace(rng=rng)
        (
            observations,
            terminated,
            _,
            next_observations,
            rewards,
            actions,
            raw_observations,
            _,
        ) = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, sample_key
        )
        dones = terminated
        agent_state, _ = update_value_fn(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            dones=dones,
            agent_state=agent_state,
            recurrent=recurrent,
            gamma=gamma,
            reward_scale=reward_scale,
        )
        agent_state = update_target_fn(agent_state, tau=5e-4)
        return agent_state, None

    agent_state, _ = jax.lax.scan(
        critic_pretrain_step, agent_state, None, length=n_steps
    )
    return agent_state
