"""TD3 deterministic actor and Deterministic distribution wrapper.

Matches the original TD3 architecture (Fujimoto et al., 2018, Table 1):
state -> Dense -> ReLU -> Dense -> ReLU -> Dense(action_dim) -> tanh.

No LayerNorm, no log_std head. Final layer uses uniform(-3e-3, 3e-3) init
per the paper / original DDPG init recipe.
"""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant
from flax.serialization import to_state_dict

from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.networks.networks import MultiCritic, init_network_state
from ajax.networks.utils import (
    get_adam_tx,
    parse_architecture,
    uniform_init,
)
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import ActivationFunction


class Deterministic:
    """Tiny distrax-compatible wrapper for deterministic policies.

    Exposes `.mean()` so existing code (compute_imitation_score, the policy
    loss, the target action computation) keeps working without branches.
    """

    def __init__(self, action: jax.Array):
        self._action = action

    def mean(self) -> jax.Array:
        return self._action

    def sample(self, seed=None, sample_shape=()) -> jax.Array:
        del seed, sample_shape
        return self._action

    def sample_and_log_prob(self, seed=None) -> Tuple[jax.Array, jax.Array]:
        del seed
        return self._action, jnp.zeros(self._action.shape[:-1] + (1,))

    def entropy(self) -> jax.Array:
        return jnp.zeros(self._action.shape[:-1])

    def mode(self) -> jax.Array:
        return self._action

    def stddev(self) -> jax.Array:
        return jnp.zeros_like(self._action)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """For BC pretraining: a delta distribution's log-density is +∞ at
        x=action, -∞ elsewhere. We use the negative-MSE proxy so that
        maximizing log_prob is equivalent to MSE-minimizing the actor toward
        the target action. Shape: (..., action_dim) -> (..., action_dim)
        (per-dim, summed by caller)."""
        return -((x - self._action) ** 2)


class DeterministicActor(nn.Module):
    """TD3 deterministic actor: hidden_arch + Dense(action_dim) + tanh."""

    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int

    def setup(self):
        self.hidden = nn.Sequential(parse_architecture(self.input_architecture))
        # Final layer init: uniform(-3e-3, 3e-3) -- standard DDPG/TD3 final-layer init.
        self.head = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(3e-3),
            bias_init=constant(0.0),
            name="head",
        )

    def __call__(self, obs: jax.Array, raw_obs=None) -> Deterministic:
        del raw_obs
        h = self.hidden(obs)
        return Deterministic(jnp.tanh(self.head(h)))


def get_initialized_td3_actor_critic(
    key: jax.Array,
    env_config: EnvironmentConfig,
    actor_optimizer_config: OptimizerConfig,
    critic_optimizer_config: OptimizerConfig,
    network_config: NetworkConfig,
    num_critics: int = 2,
    pid_actor_config: Optional[object] = None,
    action_dim_override: Optional[int] = None,
) -> Tuple[LoadedTrainState, LoadedTrainState]:
    """TD3-specific init: deterministic actor + standard MultiCritic ensemble.

    Falls back to PIDActor when pid_actor_config is provided (the PID actor
    already returns a deterministic action via its `.mean()`).
    """
    action_dim = (
        action_dim_override
        if action_dim_override is not None
        else get_action_dim(env_config.env, env_config.env_params)
    )

    if pid_actor_config is not None:
        from ajax.modules.pid_actor import PIDActorNetwork

        actor = PIDActorNetwork(
            input_architecture=network_config.actor_architecture,
            action_dim=action_dim,
            obs_current_idx=pid_actor_config.obs_current_idx,
            obs_target_idx=pid_actor_config.obs_target_idx,
            obs_derivative_idx=pid_actor_config.obs_derivative_idx,
            penultimate_normalization=network_config.penultimate_normalization,
        )
    else:
        actor = DeterministicActor(
            input_architecture=network_config.actor_architecture,
            action_dim=action_dim,
        )
    critic = MultiCritic(
        input_architecture=network_config.critic_architecture,
        penultimate_normalization=network_config.penultimate_normalization,
        num=num_critics,
        kernel_init=network_config.critic_kernel_init,
        bias_init=network_config.critic_bias_init,
        encoder_kernel_init=network_config.encoder_kernel_init,
        encoder_bias_init=network_config.encoder_bias_init,
    )

    actor_tx = get_adam_tx(**to_state_dict(actor_optimizer_config))
    critic_tx = get_adam_tx(**to_state_dict(critic_optimizer_config))
    actor_key, critic_key = jax.random.split(key)

    observation_shape, action_shape = get_state_action_shapes(env_config.env)
    if action_dim_override is not None:
        action_shape = (action_dim_override,)
    init_obs = jnp.zeros((env_config.n_envs, *observation_shape))
    init_action = jnp.zeros((env_config.n_envs, *action_shape))

    actor_state = init_network_state(
        init_x=init_obs,
        network=actor,
        key=actor_key,
        tx=actor_tx,
        recurrent=network_config.lstm_hidden_size is not None,
        lstm_hidden_size=network_config.lstm_hidden_size,
        n_envs=env_config.n_envs,
        lr_schedule=actor_optimizer_config.learning_rate,
    )
    critic_state = init_network_state(
        init_x=jnp.hstack([init_obs, init_action]),
        network=critic,
        key=critic_key,
        tx=critic_tx,
        recurrent=network_config.lstm_hidden_size is not None,
        lstm_hidden_size=network_config.lstm_hidden_size,
        n_envs=env_config.n_envs,
        lr_schedule=critic_optimizer_config.learning_rate,
    )
    return actor_state, critic_state
