from collections.abc import Sequence
from typing import NamedTuple, Optional, Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.linen.normalization import _l2_normalize
from flax.serialization import to_state_dict

from ajax.agents.SAC.utils import SquashedNormal
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.modules.pid_actor import PIDActorConfig, PIDActorNetwork
from ajax.networks.memory import (
    MemoryCell,
    MemoryConfig,
    init_carry,
    parse_memory_config,
)
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


class CNNEncoder(nn.Module):
    """Conv encoder for image observations packed flat as ``(H*W*C + extra_dim,)``.

    The flat layout (rather than native NHWC) keeps Ajax's existing collector
    and command-augmentation paths working unchanged: extra trailing scalar
    dims (e.g. UDRL's (d_r, d_h) command) are concatenated to the embedding
    AFTER the convolutions, so the conv stack only sees the image.

    The image is assumed to be stored in NHWC order: a Flatten wrapper in
    user code must transpose a stack-first array (T, H, W) into (H, W, T)
    before flattening so reshape(H, W, C) recovers the right layout.
    """

    image_shape: Tuple[int, int, int]  # (H, W, C)
    extra_obs_dim: int = 0
    channels: Tuple[int, ...] = (16, 32)
    kernel_sizes: Tuple[int, ...] = (4, 3)
    strides: Tuple[int, ...] = (2, 2)
    feature_dim: int = 128

    @nn.compact
    def __call__(self, x):
        H, W, C = self.image_shape
        img_flat = H * W * C
        if self.extra_obs_dim > 0:
            img = x[..., :img_flat]
            extra = x[..., img_flat:]
        else:
            img = x
            extra = None
        img = img.reshape(*x.shape[:-1], H, W, C)
        for c, k, s in zip(self.channels, self.kernel_sizes, self.strides):
            # Each kernel/stride entry may be an int (square) or an (h, w)
            # tuple -- the latter for non-square images (e.g. octax's 64x32).
            ks = k if isinstance(k, tuple) else (k, k)
            st = s if isinstance(s, tuple) else (s, s)
            img = nn.Conv(c, kernel_size=ks, strides=st)(img)
            img = nn.relu(img)
        # Flatten the spatial+channel dims while preserving leading batch dims.
        img = img.reshape(*img.shape[:-3], -1)
        feat = nn.Dense(self.feature_dim)(img)
        feat = nn.relu(feat)
        if extra is not None:
            feat = jnp.concatenate([feat, extra], axis=-1)
        return feat


class CNNSpec(NamedTuple):
    """CNN encoder architecture: conv stack + projection width.

    Defaults match :class:`CNNEncoder`'s own defaults, so ``CNNSpec()``
    reproduces the legacy encoder. Each ``kernel_sizes`` / ``strides``
    entry may be an ``int`` (square) or an ``(h, w)`` tuple. A
    ``NamedTuple`` -> hashable, so it is safe as a flax module field and
    as a :class:`NetworkConfig` field.
    """

    channels: Tuple[int, ...] = (16, 32)
    kernel_sizes: Tuple = (4, 3)
    strides: Tuple = (2, 2)
    feature_dim: int = 128


def build_cnn_encoder(image_shape, extra_obs_dim=0, cnn_spec=None):
    """Build a :class:`CNNEncoder` from an optional :class:`CNNSpec`.

    ``cnn_spec=None`` -> ``CNNEncoder``'s default architecture. The single
    place that turns a (shape, spec) pair into an encoder, so every
    network module wires the CNN identically.
    """
    spec = cnn_spec if cnn_spec is not None else CNNSpec()
    return CNNEncoder(
        image_shape=image_shape,
        extra_obs_dim=extra_obs_dim,
        channels=spec.channels,
        kernel_sizes=spec.kernel_sizes,
        strides=spec.strides,
        feature_dim=spec.feature_dim,
    )


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
    # When True, log_std is a single learnable scalar Param per action
    # dim (state-independent), matching brax PPO's ``noise_std_type=
    # 'scalar'``. When False (default), log_std is a Dense layer over
    # the encoder features (state-dependent, the SAC convention).
    # ``log_std_init`` sets the initial value of log_std (in either
    # mode). brax PPO defaults to ``init_noise_std=1.0`` (a softplus-
    # parametrised scalar around std=1.31); the equivalent in log-space
    # is ``log_std_init=0.0`` giving std=1.0. The legacy Ajax default
    # (``-1.0``) gives std=exp(-1)≈0.37, which is ~3.5× quieter than
    # brax — explored less and contributed to PPO policy collapse on
    # bounded continuous control envs (mujoco_playground manip).
    log_std_state_independent: bool = False
    log_std_init: float = -1.0
    # Mean-head kernel init. None keeps the legacy ``orthogonal(0.01)``
    # default (small initial mean, used by SAC/old PPO/etc). Setting to
    # a string ("lecun_uniform", "orthogonal", ...) parses via
    # ``parse_initialization``; setting to a callable uses it directly.
    # brax PPO uses ``lecun_uniform`` for the mean head, which gives an
    # initial mean magnitude ~17x bigger than ``orthogonal(0.01)`` --
    # essential for the eval (deterministic mean) action to be non-zero
    # at the start of training. When the mean stays at ~0 (the legacy
    # default), eval = tanh(near_zero) ≈ near_zero and the arm doesn't
    # move; combined with the wide std=1 exploration noise this produces
    # the "train_reward >> eval_reward" pathology on continuous control.
    mean_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # When True, skip the LayerNorm at the encoder output. Ajax's
    # Encoder applies ``nn.LayerNorm()`` to the final hidden features
    # before the policy/value heads; brax PPO's MLP does not. The
    # LayerNorm silently rescales the encoder output to ~N(0,1) PRE
    # the heads, which interacts badly with brax-tuned lecun_uniform
    # head init (the heads expect un-normalised inputs).
    disable_encoder_output_norm: bool = False
    # Optional CNN encoder. When `cnn_image_shape` is provided, the encoder
    # treats obs as `(*batch, H*W*C + cnn_extra_obs_dim)` flat: the image
    # portion is reshaped to NHWC, run through a small conv stack, then
    # any trailing scalar dims (e.g. UDRL command) are concatenated to the
    # embedding before the heads. None keeps the legacy MLP encoder.
    cnn_image_shape: Optional[Tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
    cnn_spec: Optional[CNNSpec] = None
    # Optional memory block between encoder and heads. When set, __call__
    # takes time-major (T, B, obs) plus (hidden_state, done) and returns
    # (distribution, new_hidden_state). None keeps the network feedforward
    # and the call signature unchanged.
    memory: Optional[MemoryConfig] = None

    def setup(self):
        if self.memory is not None:
            self.memory_cell = MemoryCell(self.memory)
        if self.cnn_image_shape is not None:
            self.encoder = build_cnn_encoder(
                self.cnn_image_shape, self.cnn_extra_obs_dim, self.cnn_spec
            )
        else:
            self.encoder = Encoder(
                input_architecture=self.input_architecture,
                penultimate_normalization=self.penultimate_normalization,
                kernel_init=self.encoder_kernel_init,
                bias_init=self.encoder_bias_init,
                disable_output_norm=self.disable_encoder_output_norm,
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
            if self.mean_kernel_init is None:
                _mean_kernel_init = orthogonal(0.01)
            elif callable(self.mean_kernel_init):
                _mean_kernel_init = self.mean_kernel_init
            else:
                _mean_kernel_init = parse_initialization(self.mean_kernel_init)
            self.mean = nn.Dense(
                self.action_dim,
                kernel_init=_mean_kernel_init,
                bias_init=bias_init,
                name="mean",
            )
            # log_std head: two flavours.
            #  * Dense over the encoder features (state-dependent, SAC's
            #    convention). kernel_init=zeros means output equals bias
            #    at init -- clean starting std=exp(log_std_init).
            #  * Scalar Param per action dim (state-independent, brax PPO's
            #    convention). Lives outside the encoder so the policy's
            #    exploration profile is a pure global scalar that doesn't
            #    couple to the value-estimating features. flax requires
            #    such params to be declared in setup() (or in a method
            #    wrapped with @nn.compact); we use the former.
            if not self.log_std_state_independent:
                self.log_std = nn.Dense(
                    self.action_dim,
                    kernel_init=nn.initializers.zeros,
                    bias_init=nn.initializers.constant(self.log_std_init),
                    name="log_std",
                )
            else:
                self.log_std_param = self.param(
                    "log_std_param",
                    nn.initializers.constant(self.log_std_init),
                    (self.action_dim,),
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

    def _distribution(self, embedding) -> distrax.Distribution:
        if self.continuous:
            mean = self.mean(embedding)
            if self.log_std_state_independent:
                # Single learnable scalar per action dim, declared in
                # setup(); broadcast to mean.shape for the elementwise
                # std computation.
                log_std = jnp.clip(
                    jnp.broadcast_to(self.log_std_param, mean.shape),
                    -20,
                    2,
                )
            else:
                log_std = jnp.clip(self.log_std(embedding), -20, 2)
            std = jnp.exp(log_std)
            return (
                distrax.Normal(mean, std)
                if not self.squash
                else SquashedNormal(mean, std)
            )
        return self.model(embedding)

    def __call__(self, obs, hidden_state=None, done=None):
        embedding = self.encoder(obs)
        if self.memory is not None:
            if hidden_state is None or done is None:
                raise ValueError(
                    "Recurrent Actor requires hidden_state and done flags."
                )
            hidden_state, embedding = self.memory_cell(hidden_state, embedding, done)
            return self._distribution(embedding), hidden_state
        return self._distribution(embedding)


class Critic(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None
    # See Actor.disable_encoder_output_norm for rationale; same field
    # for the critic so PPO can disable LayerNorm on both heads.
    disable_encoder_output_norm: bool = False
    # When set, the encoder is a `CNNEncoder` over flat image obs rather
    # than the MLP `Encoder`. See `NetworkConfig.cnn_image_shape`. Valid
    # only for state-value critics (obs-only input); an action-value
    # critic concatenates the action, which the CNN reshape does not
    # expect, so SAC-style Q(s,a) critics keep the MLP encoder.
    cnn_image_shape: Optional[Tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
    cnn_spec: Optional[CNNSpec] = None
    # Optional memory block; see Actor.memory. When set, __call__ takes
    # time-major (T, B, features) plus (hidden_state, done) and returns
    # (values, new_hidden_state).
    memory: Optional[MemoryConfig] = None
    # A recurrent Q-critic's current action: the input's last ``query_dim``
    # features skip the memory, which reads the rest (the observation and
    # the previous action, :func:`action_value_input`) through its own input
    # projection; the encoder then runs after the memory, as the head over
    # its output and the action. A queried action never feeds the carry (Ni
    # et al. 2022's recurrent critic; cited from memory, unverified).
    query_dim: int = 0

    def setup(self):
        if self.memory is not None:
            self.memory_cell = MemoryCell(self.memory)
        if self.cnn_image_shape is not None:
            self.encoder = build_cnn_encoder(
                self.cnn_image_shape, self.cnn_extra_obs_dim, self.cnn_spec
            )
        else:
            self.encoder = Encoder(
                input_architecture=self.input_architecture,
                penultimate_normalization=self.penultimate_normalization,
                kernel_init=self.encoder_kernel_init,
                bias_init=self.encoder_bias_init,
                disable_output_norm=self.disable_encoder_output_norm,
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

    def __call__(self, x: jax.Array, hidden_state=None, done=None):
        if self.memory is None:
            return self.model(self.encoder(x))
        if hidden_state is None or done is None:
            raise ValueError("Recurrent Critic requires hidden_state and done flags.")
        if not self.query_dim:
            hidden_state, feat = self.memory_cell(hidden_state, self.encoder(x), done)
            return self.model(feat), hidden_state
        x, query = x[..., : -self.query_dim], x[..., -self.query_dim :]
        hidden_state, feat = self.memory_cell(hidden_state, x, done)
        feat = self.encoder(jnp.concatenate([feat, query], axis=-1))
        return self.model(feat), hidden_state

    def apply_encoder(self, x: jax.Array) -> jax.Array:
        """Expose the encoder's features alone, without the value head.

        Used by auxiliary representation-shaping losses (VAE, RSSM) that
        need to push gradients into the shared encoder while owning
        their own decoder/prior/posterior heads.
        """
        return self.encoder(x)


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
    disable_encoder_output_norm: bool = False
    cnn_image_shape: Optional[Tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
    cnn_spec: Optional[CNNSpec] = None
    # Optional memory block; each ensemble member owns its carry, stacked
    # on a leading axis: hidden_state leaves are (num, batch, hidden).
    memory: Optional[MemoryConfig] = None
    query_dim: int = 0  # see Critic.query_dim

    def setup(self):
        # x (and done) are broadcast across the ensemble; the carry is
        # mapped over its leading (num,) axis so each member evolves its
        # own memory. NOTE: the recurrent in_axes spec matches __call__'s
        # 3-arg signature; apply_encoder (1 arg) is only used by the
        # VAE/RSSM auxiliaries, which never combine with memory.
        in_axes = (None, 0, None) if self.memory is not None else None
        Vmapped = nn.vmap(
            target=Critic,
            in_axes=in_axes,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num,
            methods=("__call__", "apply_encoder"),
        )
        self.ensemble = Vmapped(
            input_architecture=self.input_architecture,
            penultimate_normalization=self.penultimate_normalization,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            encoder_kernel_init=self.encoder_kernel_init,
            encoder_bias_init=self.encoder_bias_init,
            disable_encoder_output_norm=self.disable_encoder_output_norm,
            cnn_image_shape=self.cnn_image_shape,
            cnn_extra_obs_dim=self.cnn_extra_obs_dim,
            cnn_spec=self.cnn_spec,
            memory=self.memory,
            query_dim=self.query_dim,
        )

    def __call__(self, x: jax.Array, hidden_state=None, done=None):
        if self.memory is not None:
            # Returns (values, new_hidden_state): values (num, T, B, 1),
            # hidden leaves (num, B, hidden).
            return self.ensemble(x, hidden_state, done)
        return self.ensemble(x)

    def apply_encoder_ensemble(self, x: jax.Array) -> jax.Array:
        """Per-ensemble-member encoder features, shape ``(num, ..., d)``.

        Auxiliary modules (VAE, RSSM) consume these features to drive
        gradients back through the shared encoder. Callers typically
        average across the ensemble axis since the encoder sees
        identical inputs and only diverges via its init RNG.
        """
        return self.ensemble.apply_encoder(x)


def action_value_input(
    obs: jax.Array, action: jax.Array, previous_action: Optional[jax.Array] = None
) -> jax.Array:
    """A Q-critic's input: ``(obs, action)``, or, given the previous
    action, the recurrent critic's ``(obs, previous_action, action)``: its
    memory reads the observation and the previous action, the current
    action joins at its head (``Critic.query_dim``)."""
    if previous_action is None:
        return jnp.concatenate([obs, action], axis=-1)
    return jnp.concatenate([obs, previous_action, action], axis=-1)


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
    max_timesteps: Optional[int] = None,
    extra_obs_dim: int = 0,
    pid_actor_config: Optional[PIDActorConfig] = None,
    action_dim_override: Optional[int] = None,
    cnn_image_shape: Optional[Tuple[int, int, int]] = None,
    log_std_state_independent: bool = False,
    log_std_init: float = -1.0,
    mean_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    disable_encoder_output_norm: bool = False,
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

    memory = parse_memory_config(network_config.memory)
    if memory is not None:
        if pid_actor_config is not None:
            raise NotImplementedError("PIDActorNetwork does not support memory yet.")
        if network_config.penultimate_normalization:
            # The encoder's l2-normalization is hardcoded to axis=1, which
            # is the batch axis on time-major (T, B, F) inputs.
            raise NotImplementedError(
                "penultimate_normalization is incompatible with memory."
            )

    # Resolve the CNN encoder spec: an explicit arg (UDRL passes one)
    # takes precedence, else fall back to the NetworkConfig field (the
    # path SAC/PPO use). A CNN critic is only valid for state-value
    # critics; an action-value critic (SAC's Q(s,a)) keeps the MLP.
    if cnn_image_shape is None:
        cnn_image_shape = network_config.cnn_image_shape
    critic_cnn_image_shape = None if action_value else cnn_image_shape
    cnn_spec = network_config.cnn_spec  # conv architecture (None -> default)

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
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=extra_obs_dim,
            cnn_spec=cnn_spec,
            log_std_state_independent=log_std_state_independent,
            log_std_init=log_std_init,
            mean_kernel_init=mean_kernel_init,
            disable_encoder_output_norm=disable_encoder_output_norm,
            memory=memory,
        )
    critic = MultiCritic(
        input_architecture=network_config.critic_architecture,
        penultimate_normalization=network_config.penultimate_normalization,
        num=num_critics,
        kernel_init=critic_kernel_init,
        bias_init=critic_bias_init,
        encoder_kernel_init=encoder_kernel_init,
        encoder_bias_init=encoder_bias_init,
        disable_encoder_output_norm=disable_encoder_output_norm,
        cnn_image_shape=critic_cnn_image_shape,
        cnn_extra_obs_dim=network_config.cnn_extra_obs_dim,
        cnn_spec=cnn_spec,
        memory=memory,
        query_dim=action_dim if action_value and memory is not None else 0,
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

    # Flax network.init only reads init_x's shape to infer param shapes;
    # the leading batch dim can be 1. Allocating (n_envs, ...) just
    # materialised an n_envs× larger zero tensor for no benefit, and
    # matters when n_envs is large or obs are high-dim (images).
    init_obs = jnp.zeros((1, *observation_shape))
    if action_dim_override is not None:
        action_shape = (action_dim_override,)
    init_action = jnp.zeros((1, *action_shape))

    actor_state = init_network_state(
        init_x=init_obs,
        network=actor,
        key=actor_key,
        tx=actor_tx,
        memory=memory,
        n_envs=env_config.n_envs,
    )
    critic_state = init_network_state(
        init_x=(
            action_value_input(
                init_obs, init_action, None if memory is None else init_action
            )
            if action_value
            else init_obs
        ),
        network=critic,
        key=critic_key,
        tx=critic_tx,
        memory=memory,
        n_envs=env_config.n_envs,
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

    init_obs = jnp.zeros((1, *observation_shape))
    init_action = jnp.zeros((1, *action_shape))

    return init_network_state(
        init_x=jnp.hstack([init_obs, init_action]),
        network=critic,
        key=key,
        tx=critic_tx,
        memory=parse_memory_config(network_config.memory),
        n_envs=env_config.n_envs,
    )


def init_network_carry(network, memory: MemoryConfig, key: jax.Array, batch_size: int):
    """Fresh carry for ``network``. Ensembles (modules exposing ``num``)
    get one carry per member, stacked on a leading (num,) axis."""
    carry = init_carry(memory, key, batch_size)
    num = getattr(network, "num", None)
    if num is not None:
        carry = jax.tree.map(lambda x: jnp.repeat(x[None], num, axis=0), carry)
    return carry


def init_network_state(
    init_x: jax.Array,
    network: nn.Module,
    key: jax.Array,
    tx: optax.GradientTransformation,
    n_envs: int = 1,
    memory: Optional[MemoryConfig] = None,
) -> LoadedTrainState:
    """Initialise ``network`` on ``init_x`` into a :class:`LoadedTrainState`.

    With a ``memory`` config the network is recurrent: it is initialised on
    a single-step time-major sequence and the state carries a fresh carry
    batch-sized to ``n_envs``.
    """
    if memory is None:
        params = FrozenDict(network.init(key, init_x))
        hidden_state = None
    else:
        init_key, carry_key = jax.random.split(key)
        # The live carry is batch-sized to n_envs; the init forward pass
        # uses a carry matching init_x's (possibly smaller) batch — param
        # shapes don't depend on the batch dim, so init with batch=1 obs
        # (see get_initialized_actor_critic) stays cheap.
        hidden_state = init_network_carry(network, memory, carry_key, n_envs)
        init_carry_batch = init_network_carry(
            network, memory, carry_key, init_x.shape[0]
        )
        # Recurrent networks consume time-major (T, B, ...) sequences;
        # initialize with a single-step sequence.
        params = FrozenDict(
            network.init(
                init_key,
                init_x[None, ...],
                hidden_state=init_carry_batch,
                done=jnp.zeros((1, init_x.shape[0]), dtype=bool),
            )
        )
    return LoadedTrainState.create(
        params=params,
        tx=tx,
        apply_fn=network.apply,
        hidden_state=hidden_state,
        recurrent=memory is not None,
        target_params=params,
    )


def _apply_critic_obs_norm(critic_state: LoadedTrainState, x: jax.Array) -> jax.Array:
    """Normalise the obs slice of a critic input (obs or concat(obs, action));
    the action slice stays raw. No-op when normalisation is disabled."""
    obs_norm_info = getattr(critic_state, "obs_norm_info", None)
    if obs_norm_info is None or obs_norm_info.var is None:
        return x
    from ajax.agents.obs_norm import apply_obs_norm

    obs_dim = obs_norm_info.mean.shape[-1]
    obs_part = apply_obs_norm(x[..., :obs_dim], obs_norm_info)
    return jnp.concatenate([obs_part, x[..., obs_dim:]], axis=-1)


def predict_value_sequence(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
    resets: jax.Array,
    initial_hidden: HiddenState,
) -> Tuple[jax.Array, HiddenState]:
    """Run a recurrent critic over a time-major sequence.

    Args:
        x: (T, B, features) critic input (obs, or concat(obs, action)).
        resets: (T, B) episode-start flags aligned with ``x`` (resets[t]
            means x[t] is the first observation of a new episode).
        initial_hidden: carry valid for x[0]; leaves are (num, B, hidden)
            for critic ensembles.

    Returns:
        (values, final_hidden): values (num, T, B, 1); final_hidden is the
        carry after consuming the whole sequence.
    """
    x = _apply_critic_obs_norm(critic_state, x)
    return critic_state.apply_fn(
        critic_params, x, hidden_state=initial_hidden, done=resets
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
