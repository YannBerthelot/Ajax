from collections.abc import Sequence
from typing import Any, NamedTuple, Optional, Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.serialization import to_state_dict

from ajax.environments.obs_norm import apply_obs_norm
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.modules.pid_actor import PIDActor, PIDPolicy, check_pid_actor
from ajax.networks.distributions import SquashedNormal
from ajax.networks.memory import (
    Carry,
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
from ajax.types import ActivationFunction, InitializationFunction


class Encoder(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None
    # Opt-in LayerNorm on the output features. Off by default: the SAC,
    # TD3, PPO and DQN papers use plain MLPs. Ignored under
    # penultimate_normalization, which takes precedence.
    layer_norm: bool = False

    def setup(self):
        layers = parse_architecture(
            self.input_architecture, self.kernel_init, self.bias_init
        )
        self.network = nn.Sequential(layers)
        if self.layer_norm:
            self.norm = nn.LayerNorm()

    def __call__(self, x):
        features = self.network(x)
        if self.penultimate_normalization:
            # Each feature vector (last axis) to unit L2 norm, whatever the
            # leading batch or time axes: flax's _l2_normalize formula and
            # eps, bitwise equal to its axis=1 on (B, F) inputs.
            return features * jax.lax.rsqrt(
                (features * features).sum(axis=-1, keepdims=True) + 1e-12
            )
        if self.layer_norm:
            return self.norm(features)
        return features


class CNNSpec(NamedTuple):
    """CNN encoder architecture: conv stack + projection width.

    ``CNNSpec()`` is :class:`CNNEncoder`'s default architecture. Each
    ``kernel_sizes`` / ``strides`` entry may be an ``int`` (square) or an
    ``(h, w)`` tuple. A ``NamedTuple`` -> hashable, so it is safe as a flax
    module field and as a :class:`NetworkConfig` field.
    """

    channels: Tuple[int, ...] = (16, 32)
    kernel_sizes: Tuple = (4, 3)
    strides: Tuple = (2, 2)
    feature_dim: int = 128


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
    spec: Optional[CNNSpec] = None  # None: CNNSpec()

    @nn.compact
    def __call__(self, x):
        spec = self.spec if self.spec is not None else CNNSpec()
        H, W, C = self.image_shape
        img_flat = H * W * C
        if x.shape[-1] != img_flat + self.extra_obs_dim:
            raise ValueError(
                f"CNNEncoder reads {img_flat} image dims {self.image_shape} and"
                f" {self.extra_obs_dim} trailing dims, got {x.shape[-1]} dims."
            )
        if self.extra_obs_dim > 0:
            img = x[..., :img_flat]
            extra = x[..., img_flat:]
        else:
            img = x
            extra = None
        img = img.reshape(*x.shape[:-1], H, W, C)
        for c, k, s in zip(
            spec.channels, spec.kernel_sizes, spec.strides, strict=False
        ):
            # Each kernel/stride entry may be an int (square) or an (h, w)
            # tuple -- the latter for non-square images (e.g. octax's 64x32).
            ks = k if isinstance(k, tuple) else (k, k)
            st = s if isinstance(s, tuple) else (s, s)
            img = nn.Conv(c, kernel_size=ks, strides=st)(img)
            img = nn.relu(img)
        # Flatten the spatial+channel dims while preserving leading batch dims.
        img = img.reshape(*img.shape[:-3], -1)
        feat = nn.Dense(spec.feature_dim)(img)
        feat = nn.relu(feat)
        if extra is not None:
            feat = jnp.concatenate([feat, extra], axis=-1)
        return feat


class Actor(nn.Module):
    """The stochastic policy of SAC, REDQ, ASAC, AVG, PPO, APO and UDRL: a
    Normal (or tanh-squashed) head for continuous actions, a Categorical one
    for discrete. Expert guidance acts at the loss level, not in the
    architecture, which keeps the policy unconstrained.
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int
    continuous: bool = False
    squash: bool = False
    penultimate_normalization: bool = False
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # When True, log_std is a single learnable scalar Param per action
    # dim (state-independent), matching brax PPO's ``noise_std_type=
    # 'scalar'``. When False (default), log_std is a Dense layer over
    # the encoder features (state-dependent, the SAC convention).
    # ``log_std_init`` sets the initial value of log_std (in either
    # mode). brax PPO defaults to ``init_noise_std=1.0`` (a softplus-
    # parametrised scalar around std=1.31); the equivalent in log-space
    # is ``log_std_init=0.0`` giving std=1.0. The Ajax default
    # (``-1.0``) gives std=exp(-1)≈0.37, which is ~3.5× quieter than
    # brax — explored less and contributed to PPO policy collapse on
    # bounded continuous control envs (mujoco_playground manip).
    log_std_state_independent: bool = False
    log_std_init: float = -1.0
    # Mean-head kernel init (a ``parse_initialization`` string or a
    # callable). None keeps ``orthogonal(0.01)``, a small initial mean.
    # brax PPO uses ``lecun_uniform`` for the mean head, which gives an
    # initial mean magnitude ~17x bigger than ``orthogonal(0.01)`` --
    # essential for the eval (deterministic mean) action to be non-zero
    # at the start of training. When the mean stays at ~0 (the
    # default), eval = tanh(near_zero) ≈ near_zero and the arm doesn't
    # move; combined with the wide std=1 exploration noise this produces
    # the "train_reward >> eval_reward" pathology on continuous control.
    mean_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # Opt-in LayerNorm on the encoder's output (see Encoder.layer_norm).
    encoder_layer_norm: bool = False
    # Optional CNN encoder over flat image obs (see `CNNEncoder`); None
    # keeps the MLP encoder.
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
            self.encoder = CNNEncoder(
                self.cnn_image_shape, self.cnn_extra_obs_dim, self.cnn_spec
            )
        else:
            self.encoder = Encoder(
                input_architecture=self.input_architecture,
                penultimate_normalization=self.penultimate_normalization,
                kernel_init=self.encoder_kernel_init,
                layer_norm=self.encoder_layer_norm,
            )
        kernel_init = orthogonal(1.0)
        bias_init = parse_initialization(self.bias_init, default=constant(0.0))

        if self.continuous:
            self.mean = nn.Dense(
                self.action_dim,
                kernel_init=parse_initialization(
                    self.mean_kernel_init, default=orthogonal(0.01)
                ),
                bias_init=bias_init,
                name="mean",
            )
            # The state-dependent head's zero kernel outputs its bias at
            # init: std=exp(log_std_init). The state-independent Param
            # lives outside the encoder so exploration does not couple to
            # the value-estimating features; flax requires it declared in
            # setup() (or an @nn.compact method).
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
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # Opt-in LayerNorm on the encoder's output (see Encoder.layer_norm).
    encoder_layer_norm: bool = False
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
    # features skip the encoder and the memory, which read the rest (the
    # observation and the previous action, :func:`action_value_input`), and
    # join the memory's output at a head of the encoder's architecture. A
    # queried action never feeds the carry (Ni et al. 2022's recurrent
    # critic, embedders before the memory, the action at the head after it;
    # cited from memory, unverified).
    query_dim: int = 0

    def setup(self):
        if self.memory is not None:
            self.memory_cell = MemoryCell(self.memory)
            if self.query_dim:
                self.head = Encoder(
                    input_architecture=self.input_architecture,
                    kernel_init=self.encoder_kernel_init,
                    layer_norm=self.encoder_layer_norm,
                )
        if self.cnn_image_shape is not None:
            self.encoder = CNNEncoder(
                self.cnn_image_shape, self.cnn_extra_obs_dim, self.cnn_spec
            )
        else:
            self.encoder = Encoder(
                input_architecture=self.input_architecture,
                penultimate_normalization=self.penultimate_normalization,
                kernel_init=self.encoder_kernel_init,
                layer_norm=self.encoder_layer_norm,
            )
        self.model = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=parse_initialization(self.bias_init, default=constant(0.0)),
        )

    def __call__(self, x: jax.Array, hidden_state=None, done=None):
        if self.memory is None:
            return self.model(self.encoder(x))
        if hidden_state is None or done is None:
            raise ValueError("Recurrent Critic requires hidden_state and done flags.")
        query = None
        if self.query_dim:
            x, query = x[..., : -self.query_dim], x[..., -self.query_dim :]
        hidden_state, feat = self.memory_cell(hidden_state, self.encoder(x), done)
        if query is not None:
            feat = self.head(jnp.concatenate([feat, query], axis=-1))
        return self.model(feat), hidden_state

    def apply_encoder(self, x: jax.Array) -> jax.Array:
        """The encoder's features alone, without the value head (read by
        :class:`ajax.extensions.instrumentation.ConditioningMetrics`)."""
        return self.encoder(x)


class MultiCritic(nn.Module):
    """Ensemble of ``num`` critics, vmapped over a leading parameter axis."""

    input_architecture: Sequence[Union[str, ActivationFunction]]
    num: int = 4
    penultimate_normalization: bool = False
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_layer_norm: bool = False
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
        # 3-arg signature; apply_encoder (1 arg) is only used by
        # ConditioningMetrics, which therefore needs a feedforward critic.
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
            bias_init=self.bias_init,
            encoder_kernel_init=self.encoder_kernel_init,
            encoder_layer_norm=self.encoder_layer_norm,
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

    def first_member(self, x: jax.Array):
        """The first member's values alone, shape ``(1, ..., 1)``, run as a
        one-member ensemble on its slice of the parameters: TD3's actor
        reads Q1 only, so the other members' passes are not run.
        Feedforward only (a recurrent member would need its carry)."""
        first = jax.tree.map(lambda p: p[:1], self.variables["params"])
        return self.clone(num=1).apply({"params": first}, x)

    def apply_encoder_ensemble(self, x: jax.Array) -> jax.Array:
        """Per-ensemble-member encoder features, shape ``(num, ..., d)``
        (:class:`ajax.extensions.instrumentation.ConditioningMetrics`)."""
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
    max_timesteps: Optional[int] = None,
    extra_obs_dim: int = 0,
    action_dim_override: Optional[int] = None,
    actor: Optional[nn.Module] = None,
) -> Tuple[LoadedTrainState, LoadedTrainState]:
    """
    Create actor and critic networks, their initializers and log-std head
    as ``network_config`` sets them; a prebuilt ``actor`` (TD3's
    deterministic one) replaces the :class:`Actor` built here.

    extra_obs_dim: dimensions the agent appends to obs at run time (UDRL's
    command; SAC's expert action and expert state when it augments with
    them), so the network is initialised on the input width it will be fed.

    A :class:`PIDPolicy` as ``network_config.actor_architecture`` makes the
    actor a :class:`PIDActor` (see :mod:`ajax.modules.pid_actor`).
    """
    action_dim = (
        action_dim_override
        if action_dim_override is not None
        else get_action_dim(env_config.env, env_config.env_params)
    )

    memory = parse_memory_config(network_config.memory)

    # A CNN critic is only valid for state-value critics; an action-value
    # critic (SAC's Q(s,a)) keeps the MLP.
    cnn_image_shape = network_config.cnn_image_shape
    critic_cnn_image_shape = None if action_value else cnn_image_shape
    cnn_spec = network_config.cnn_spec
    # Dims appended to the env's observation at run time: the train
    # fraction and the agent's augmentation (UDRL's command, an expert's
    # action). A CNN passes them through after the convolutions, with the
    # env's own trailing dims, the same way in the actor and the critic.
    obs_extra = (1 if max_timesteps is not None else 0) + extra_obs_dim
    cnn_extra_obs_dim = network_config.cnn_extra_obs_dim + obs_extra

    if actor is not None:
        pass
    elif isinstance(network_config.actor_architecture, PIDPolicy):
        check_pid_actor(continuous, memory)
        actor = PIDActor(
            policy=network_config.actor_architecture,
            action_dim=action_dim,
            squash=squash,
            log_std_init=network_config.log_std_init,
        )
    else:
        actor = Actor(
            input_architecture=network_config.actor_architecture,
            action_dim=action_dim,
            continuous=continuous,
            squash=squash,
            penultimate_normalization=network_config.penultimate_normalization,
            bias_init=network_config.actor_bias_init,
            encoder_kernel_init=network_config.encoder_kernel_init,
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=cnn_extra_obs_dim,
            cnn_spec=cnn_spec,
            log_std_state_independent=network_config.log_std_state_independent,
            log_std_init=network_config.log_std_init,
            mean_kernel_init=network_config.mean_kernel_init,
            encoder_layer_norm=network_config.encoder_layer_norm,
            memory=memory,
        )
    critic = MultiCritic(
        input_architecture=network_config.critic_architecture,
        penultimate_normalization=network_config.penultimate_normalization,
        num=num_critics,
        bias_init=network_config.critic_bias_init,
        encoder_kernel_init=network_config.encoder_kernel_init,
        encoder_layer_norm=network_config.encoder_layer_norm,
        cnn_image_shape=critic_cnn_image_shape,
        cnn_extra_obs_dim=cnn_extra_obs_dim,
        cnn_spec=cnn_spec,
        memory=memory,
        query_dim=action_dim if action_value and memory is not None else 0,
    )

    actor_tx = get_adam_tx(**to_state_dict(actor_optimizer_config))
    critic_tx = get_adam_tx(**to_state_dict(critic_optimizer_config))
    actor_key, critic_key = jax.random.split(key)

    observation_shape, action_shape = get_state_action_shapes(env_config.env)

    if obs_extra > 0:
        _obs_shape = list(observation_shape)
        _obs_shape[-1] += obs_extra
        observation_shape = tuple(_obs_shape)

    # Flax network.init only reads init_x's shape, so a batch of 1 does:
    # n_envs rows would allocate an n_envs-times larger zero tensor for
    # nothing, which matters with many envs or image observations.
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
    the online critic so predict_value can be called on its params directly.
    Always feedforward: its one reader (MCPretrain's frozen expert critic)
    only calls predict_value."""
    critic = MultiCritic(
        input_architecture=network_config.critic_architecture,
        penultimate_normalization=network_config.penultimate_normalization,
        encoder_layer_norm=network_config.encoder_layer_norm,
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
        params: FrozenDict[str, Any] = FrozenDict(network.init(key, init_x))
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
    the action slice stays raw (``apply_obs_norm``). No-op when
    normalisation is disabled."""
    return apply_obs_norm(x, getattr(critic_state, "obs_norm_info", None))


def predict_value_sequence(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
    resets: jax.Array,
    initial_hidden: Carry,
) -> Tuple[jax.Array, Carry]:
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


def phi_star_critic_state(critic_state: LoadedTrainState) -> LoadedTrainState:
    """The critic state the frozen expert critic phi*'s parameters
    (``expert_critic_params``) are applied with: the online critic's
    network without the agent's observation statistics, as MC pretraining
    and PhiRefresh fit phi* on the observations as stored."""
    if getattr(critic_state, "obs_norm_info", None) is None:
        return critic_state
    return critic_state.replace(obs_norm_info=None)


def predict_value(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
    *,
    first_member: bool = False,
) -> jax.Array:
    """The critic's values on ``x``, or only its first member's
    (:meth:`MultiCritic.first_member`) when ``first_member``."""
    x = _apply_critic_obs_norm(critic_state, x)
    if first_member:
        return critic_state.apply_fn(critic_params, x, method=MultiCritic.first_member)
    return critic_state.apply_fn(critic_params, x)


def state_value(agent_state: Any, obs: jax.Array, action: jax.Array) -> jax.Array:
    """``V(obs)`` of the agent's critic, its ensemble's mean (``action``
    unused): an evaluation's value estimate (:class:`ajax.evaluate.ValueEstimate`)."""
    del action
    critic = agent_state.critic_state
    return predict_value(critic, critic.params, obs).mean(0).squeeze(-1)


def action_value(agent_state: Any, obs: jax.Array, action: jax.Array) -> jax.Array:
    """``Q(obs, action)`` of the agent's critic, its ensemble's mean: an
    evaluation's value estimate (:class:`ajax.evaluate.ValueEstimate`)."""
    critic = agent_state.critic_state
    x = action_value_input(obs, action.astype(obs.dtype))
    return predict_value(critic, critic.params, x).mean(0).squeeze(-1)
