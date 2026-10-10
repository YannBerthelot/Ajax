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
    """Collect expert transitions and store them in the replay buffer."""
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    transitions = collect_experience_from_expert_policy(
        expert_policy=expert_policy,
        rng=rng,
        mode=mode,
        env_args=env_args,
        n_timesteps=n_steps,
    )

    flat = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]) if x is not None else None,
        transitions,
        is_leaf=lambda x: x is None,
    )

    buffer_obs_dim = buffer_state.experience["obs"].shape[-1]
    expert_obs_dim = flat.obs.shape[-1]
    train_frac = 0.0 if buffer_obs_dim == expert_obs_dim + 1 else None
    flat_obs = maybe_append_train_frac(flat.obs, train_frac=train_frac)
    # The expert collector's next observations are the final ones at every
    # end, the live collector's (bootstrap_obs) at time limits only: they
    # differ at terminations alone, which SAC's target masks.
    flat_next_obs = maybe_append_train_frac(
        flat.next_obs.astype(jnp.float32), train_frac=train_frac
    )
    flat_raw_obs = flat.raw_obs if flat.raw_obs is not None else flat.obs

    n_total = flat_obs.shape[0]

    def add_one(buffer_state, i):
        def take(x):
            return jnp.take(x, i, axis=0, mode="clip")[None]

        _transition = {
            "obs": take(flat_obs),
            "action": take(flat.action),
            "reward": take(flat.reward),
            "terminated": take(flat.terminated),
            "truncated": take(flat.truncated),
            "next_obs": take(flat_next_obs),
            "raw_obs": take(flat_raw_obs),
            "is_expert": take(jnp.ones_like(flat_obs[..., :1])),
        }
        return buffer.add(buffer_state, _transition), None

    buffer_state, _ = jax.lax.scan(add_one, buffer_state, jnp.arange(n_total))
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
