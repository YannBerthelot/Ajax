from collections.abc import Sequence
from typing import Optional, Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.linen.normalization import _l2_normalize
from flax.serialization import to_state_dict

from ajax.agents.SAC.utils import SquashedNormal
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.networks.scanned_rnn import ScannedRNN
from ajax.networks.utils import (
    get_adam_tx,
    parse_architecture,
    parse_initialization,
)
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import ActivationFunction, HiddenState, InitializationFunction

"""
Heavy inspiration from https://github.com/Howuhh/SAC-n-jax/blob/main/SAC_n_jax_flax.py
"""


# class Encoder(nn.Module):
#     input_architecture: Sequence[Union[str, ActivationFunction]]
#     penultimate_normalization: bool = False
#     kernel_init: Optional[str] = None
#     bias_init: Optional[str] = None

#     def setup(self):
#         layers = parse_architecture(
#             self.input_architecture, self.kernel_init, self.bias_init
#         )
#         self.encoder = nn.Sequential(layers)

#     @nn.compact
#     def __call__(self, input):
#         features = self.encoder(input)
#         if self.penultimate_normalization:
#             return _l2_normalize(features, axis=1)
#         return features


class Encoder(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    penultimate_normalization: bool = False
    kernel_init: Optional[str] = None
    bias_init: Optional[str] = None

    @nn.compact
    def __call__(self, input):
        layers = parse_architecture(
            self.input_architecture, self.kernel_init, self.bias_init
        )
        encoder = nn.Sequential(layers)
        features = encoder(input)
        if self.penultimate_normalization:
            return _l2_normalize(features, axis=1)
        return features


class Actor(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int
    continuous: bool = False
    squash: bool = False
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None
    bounds: Optional[Tuple] = None

    def setup(self):
        # Initialize the Encoder as a submodule
        self.encoder = Encoder(
            input_architecture=self.input_architecture,
            penultimate_normalization=self.penultimate_normalization,
            kernel_init=self.encoder_kernel_init,
            bias_init=self.encoder_bias_init,
        )
        if self.kernel_init is None:
            kernel_init = orthogonal(1.0)
        else:
            kernel_init = parse_initialization(self.kernel_init)
        if self.bias_init is None:
            bias_init = constant(0.0)
        else:
            bias_init = parse_initialization(self.bias_init)

        if self.continuous:
            self.mean = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=bias_init,
            )
            # self.log_std = nn.Dense(
            #     self.action_dim,
            #     kernel_init=kernel_init,
            #     bias_init=bias_init,
            # )
            self.log_std = self.param(
                "log_std",
                nn.initializers.zeros,  # initialize all stds at 0
                (self.action_dim,),  # shape of the parameter
            )
        else:
            self.model = nn.Sequential(
                [
                    nn.Dense(
                        self.action_dim,
                        kernel_init=kernel_init,
                        bias_init=bias_init,
                    ),
                    distrax.Categorical,
                ],
            )

    @nn.compact
    def __call__(self, obs) -> distrax.Distribution:
        # Use the Encoder submodule
        embedding = self.encoder(obs)
        embedding = nn.LayerNorm()(embedding)
        if self.continuous:
            mean = self.mean(embedding)
            log_std = jnp.clip(self.log_std, -20, 2)
            std = jnp.exp(log_std)
            return (
                distrax.Normal(mean, std)
                if not self.squash
                else SquashedNormal(mean, std)
            )

        return self.model(embedding)


class Critic(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
        self.encoder = Encoder(
            input_architecture=self.input_architecture,
            penultimate_normalization=self.penultimate_normalization,
        )
        if self.kernel_init is None:
            kernel_init = orthogonal(1.0)
        else:
            kernel_init = parse_initialization(self.kernel_init)
        if self.bias_init is None:
            bias_init = constant(0.0)
        else:
            bias_init = parse_initialization(self.bias_init)

        self.model = nn.Dense(
            1,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        embedding = self.encoder(x)
        return self.model(embedding)


class MultiCritic(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    num: int = 1
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num,
        )

        return ensemble(
            self.input_architecture,
            self.penultimate_normalization,
            self.kernel_init,
            self.bias_init,
            self.encoder_kernel_init,
            self.encoder_bias_init,
        )(*args, **kwargs)


def get_initialized_actor_critic(
    key: jax.Array,
    env_config: EnvironmentConfig,
    actor_optimizer_config: OptimizerConfig,
    critic_optimizer_config: OptimizerConfig,
    network_config: NetworkConfig,
    continuous: bool = False,
    action_value: bool = False,
    squash: bool = False,
    num_critics: int = 1,
    actor_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
    critic_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None,
) -> Tuple[LoadedTrainState, LoadedTrainState]:
    """Create actor and critic adapted to the environment and following the\
          given architectures
    """
    action_dim = get_action_dim(env_config.env, env_config.env_params)
    # higher_bound = (
    #     env_config.env.action_space(env_config.env_params).high
    #     if "action_space" in dir(env_config.env)
    #     else jnp.inf
    # )
    # lower_bound = (
    #     env_config.env.action_space(env_config.env_params).low
    #     if "action_space" in dir(env_config.env)
    #     else -jnp.inf
    # )
    # if isinstance(higher_bound, jnp.ndarray):
    #     bounds = (
    #         tuple((low, high) for low, high in zip(lower_bound, higher_bound))
    #         if continuous and squash
    #         else None
    #     )
    # else:
    #     bounds = ((lower_bound, higher_bound),)
    actor = Actor(
        input_architecture=network_config.actor_architecture,
        action_dim=action_dim,
        continuous=continuous,
        squash=squash,
        penultimate_normalization=network_config.penultimate_normalization,
        kernel_init=actor_kernel_init,
        bias_init=actor_bias_init,
        encoder_kernel_init=encoder_kernel_init,
        encoder_bias_init=encoder_bias_init,
        # bounds=bounds,
    )
    critic = MultiCritic(
        input_architecture=network_config.critic_architecture,
        penultimate_normalization=network_config.penultimate_normalization,
        num=num_critics,
        kernel_init=critic_kernel_init,
        bias_init=critic_bias_init,
        encoder_kernel_init=encoder_kernel_init,
        encoder_bias_init=encoder_bias_init,
    )
    actor_tx = get_adam_tx(**to_state_dict(actor_optimizer_config))
    critic_tx = get_adam_tx(**to_state_dict(critic_optimizer_config))

    actor_key, critic_key = jax.random.split(key)
    observation_shape, action_shape = get_state_action_shapes(
        env_config.env,
    )
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
        init_x=jnp.hstack([init_obs, init_action]) if action_value else init_obs,
        network=critic,
        key=critic_key,
        tx=critic_tx,
        recurrent=network_config.lstm_hidden_size is not None,
        lstm_hidden_size=network_config.lstm_hidden_size,
        n_envs=env_config.n_envs,
        lr_schedule=critic_optimizer_config.learning_rate,
    )

    return actor_state, critic_state


def init_hidden_state(
    lstm_hidden_size: int,
    n_envs: int,
    rng: jax.random.PRNGKey,
) -> HiddenState:
    """Initialize the hidden state for the recurrent layer of the network."""
    # rng, _rng = jax.random.split(rng)
    return ScannedRNN(lstm_hidden_size).initialize_carry(rng, n_envs)


def init_network_state(
    init_x, network, key, tx, recurrent, lstm_hidden_size, n_envs, lr_schedule
):
    params = FrozenDict(network.init(key, init_x))
    if recurrent:
        _, hidden_state_key = jax.random.split(key)
        hidden_state = init_hidden_state(lstm_hidden_size, n_envs, hidden_state_key)
    else:
        hidden_state = None

    return LoadedTrainState.create(
        params=params,
        tx=tx,
        apply_fn=network.apply,
        hidden_state=hidden_state,
        recurrent=recurrent,
        target_params=params,
    )


def predict_value(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
) -> jax.Array:
    return critic_state.apply_fn(critic_params, x)
