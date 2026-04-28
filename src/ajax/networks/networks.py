from collections.abc import Callable, Sequence
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
from ajax.modules.pid_actor import PIDActorConfig, PIDActorNetwork
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


class Encoder(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    penultimate_normalization: bool = False
    kernel_init: Optional[str] = None
    bias_init: Optional[str] = None
    # When True, skip both LayerNorm and L2 normalization at the encoder
    # output. Required for identity-pass-through to be learnable, e.g.
    # when the actor input includes the expert action and BC must be
    # able to copy it to the output. Default False preserves the
    # existing LayerNorm behaviour for all other call sites.
    disable_output_norm: bool = False

    def setup(self):
        layers = parse_architecture(
            self.input_architecture, self.kernel_init, self.bias_init
        )
        self.network = nn.Sequential(layers)
        self.norm = nn.LayerNorm()

    def __call__(self, x):
        features = self.network(x)
        if self.disable_output_norm:
            return features
        if self.penultimate_normalization:
            return _l2_normalize(features, axis=1)
        return self.norm(features)


class Actor(nn.Module):
    """
    Standard SAC actor. Expert guidance is handled at the loss level, not in the
    architecture. This keeps the policy unconstrained and theoretically clean.
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int
    continuous: bool = False
    squash: bool = False
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
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
                name="mean",
            )
            # State-dependent log_std: kernel_init=zeros means output equals
            # bias at initialization regardless of input, giving a clean
            # starting std of exp(-1) ≈ 0.37 — enough for meaningful
            # exploration without destabilizing early training.
            self.log_std = nn.Dense(
                self.action_dim,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.constant(-1.0),
                name="log_std",
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

    def __call__(self, obs, raw_obs=None) -> distrax.Distribution:
        embedding = self.encoder(obs)
        if self.continuous:
            mean = self.mean(embedding)
            log_std = jnp.clip(self.log_std(embedding), -20, 2)
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
        kernel_init = (
            orthogonal(1.0)
            if self.kernel_init is None
            else parse_initialization(self.kernel_init)
        )
        bias_init = (
            constant(0.0)
            if self.bias_init is None
            else parse_initialization(self.bias_init)
        )
        self.model = nn.Dense(
            1,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.model(self.encoder(x))


class MultiCritic(nn.Module):
    """
    Ensemble of critics. Using num=4 is recommended for this setting:
    the min aggregation over 4 critics is significantly more conservative
    than over 2, directly reducing overestimation bias without requiring
    more gradient updates per step.
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    num: int = 4
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
    num_critics: int = 4,
    actor_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
    critic_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None,
    expert_policy: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    residual: bool = False,  # kept for API compatibility, ignored
    fixed_alpha: bool = False,  # kept for API compatibility, ignored
    max_timesteps: Optional[int] = None,
    extra_obs_dim: int = 0,
    pid_actor_config: Optional[PIDActorConfig] = None,
    action_dim_override: Optional[int] = None,
) -> Tuple[LoadedTrainState, LoadedTrainState]:
    """
    Create actor and critic networks.

    extra_obs_dim: extra dimensions appended to obs at runtime before the
    network forward pass. Set to action_dim (2 for the plane) when
    augment_obs_with_expert_action=True, so the network is initialised with
    the correct input size matching what augment_obs_if_needed produces.

    pid_actor_config: when set, uses PIDActorNetwork instead of Actor. The
    network predicts PID gains from the full observation; the policy mean is
    then computed as gains @ pid_terms (error and optionally derivative).
    Fully compatible with residual RL.

    All other expert-guidance is handled at the loss level (value
    constraint, online BC) so the architecture is always a plain Actor/MultiCritic.
    """
    action_dim = (
        action_dim_override
        if action_dim_override is not None
        else get_action_dim(env_config.env, env_config.env_params)
    )

    if pid_actor_config is not None:
        actor = PIDActorNetwork(
            input_architecture=network_config.actor_architecture,
            action_dim=action_dim,
            obs_current_idx=pid_actor_config.obs_current_idx,
            obs_target_idx=pid_actor_config.obs_target_idx,
            obs_derivative_idx=pid_actor_config.obs_derivative_idx,
            penultimate_normalization=network_config.penultimate_normalization,
        )
    else:
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

    observation_shape, action_shape = get_state_action_shapes(env_config.env)

    # Inflate obs dim for train_frac and/or obs augmentation
    obs_extra = (1 if max_timesteps is not None else 0) + extra_obs_dim
    if obs_extra > 0:
        _obs_shape = list(observation_shape)
        _obs_shape[-1] += obs_extra
        observation_shape = tuple(_obs_shape)

    init_obs = jnp.zeros((env_config.n_envs, *observation_shape))
    if action_dim_override is not None:
        action_shape = (action_dim_override,)
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


def get_initialized_critic(
    key: jax.Array,
    env_config: EnvironmentConfig,
    critic_optimizer_config: OptimizerConfig,
    network_config: NetworkConfig,
    num_critics: int = 2,
    max_timesteps: Optional[int] = None,
    extra_obs_dim: int = 0,
) -> LoadedTrainState:
    """Initialize a standalone critic network (no actor). Same architecture as
    the online critic so predict_value can be called on its params directly."""
    critic = MultiCritic(
        input_architecture=network_config.critic_architecture,
        penultimate_normalization=network_config.penultimate_normalization,
        num=num_critics,
    )

    critic_tx = get_adam_tx(**to_state_dict(critic_optimizer_config))

    observation_shape, action_shape = get_state_action_shapes(env_config.env)

    obs_extra = (1 if max_timesteps is not None else 0) + extra_obs_dim
    if obs_extra > 0:
        _obs_shape = list(observation_shape)
        _obs_shape[-1] += obs_extra
        observation_shape = tuple(_obs_shape)

    init_obs = jnp.zeros((env_config.n_envs, *observation_shape))
    init_action = jnp.zeros((env_config.n_envs, *action_shape))

    return init_network_state(
        init_x=jnp.hstack([init_obs, init_action]),
        network=critic,
        key=key,
        tx=critic_tx,
        recurrent=network_config.lstm_hidden_size is not None,
        lstm_hidden_size=network_config.lstm_hidden_size,
        n_envs=env_config.n_envs,
        lr_schedule=critic_optimizer_config.learning_rate,
    )


def init_hidden_state(
    lstm_hidden_size: int,
    n_envs: int,
    rng: jax.random.PRNGKey,
) -> HiddenState:
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
    # Agent-side obs normalisation: x = concat([obs, action]); we slice
    # the leading obs_dim, normalise it, and recombine. The action stays
    # raw (already in [-1, 1]). Stats live on critic_state.obs_norm_info,
    # synced from CollectorState after every online collection step.
    obs_norm_info = getattr(critic_state, "obs_norm_info", None)
    if obs_norm_info is not None and obs_norm_info.var is not None:
        from ajax.agents.obs_norm import apply_obs_norm
        obs_dim = obs_norm_info.mean.shape[-1]
        obs_part = x[..., :obs_dim]
        act_part = x[..., obs_dim:]
        obs_part = apply_obs_norm(obs_part, obs_norm_info)
        x = jnp.concatenate([obs_part, act_part], axis=-1)
    return critic_state.apply_fn(critic_params, x)
