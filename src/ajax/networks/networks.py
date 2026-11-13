import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Optional, Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.linen.initializers import constant, lecun_normal, orthogonal
from flax.serialization import to_state_dict

from ajax.controller import (
    feature_width,
    observation_layout,
    policy_action_dim,
    reads_previous_action,
)
from ajax.environments.obs_norm import apply_obs_norm
from ajax.environments.utils import get_state_action_shapes
from ajax.extensions.observe import pid_terms_at
from ajax.modules.pid_actor import PIDActor, PIDPolicy, check_pid_actor
from ajax.networks.distributions import squash
from ajax.networks.memory import (
    Carry,
    MemoryCell,
    MemoryConfig,
    init_carry,
    parse_memory_config,
)
from ajax.networks.utils import (
    SHARED_ENCODER_MODULES,
    TORCH_LINEAR,
    dense,
    get_adam_tx,
    parse_architecture,
    parse_initialization,
    uniform_init,
)
from ajax.random_keys import derive_key
from ajax.state import (
    LOG_STD_BOUNDS,
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    check_log_std_bounds,
    network_variables,
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
    module field and as a :class:`NetworkConfig` field. ``padding`` is
    flax's ``nn.Conv`` padding; ``layer_norm`` puts a LayerNorm after every
    convolution and the projection, before its ReLU; ``kernel_init`` names
    every layer's initializer as :func:`~ajax.networks.utils.dense` reads
    it, ``"torch_linear"`` drawing kernel and bias uniform in
    ``+-1/sqrt(fan_in)`` (torch's and Torch7's default); None is flax's
    lecun_normal with zero biases.
    """

    channels: Tuple[int, ...] = (16, 32)
    kernel_sizes: Tuple = (4, 3)
    strides: Tuple = (2, 2)
    feature_dim: int = 128
    padding: str = "SAME"
    layer_norm: bool = False
    kernel_init: Optional[str] = None


#: Mnih et al. (2015)'s network: google-deepmind/dqn @9d9b1d1
#: (dqn/convnet_atari3.lua:10-13; no padding, convnet.lua:27-33).
MNIH_CNN = CNNSpec((32, 64, 64), (8, 4, 3), (4, 2, 1), 512, "VALID")
#: MinAtar's network (Young & Tian, 2019): one 3x3 convolution of 16 channels,
#: no padding, then 128 units (kenjyoung/MinAtar @4241f24, examples/dqn.py:72,
#: :79-80), the one purejaxql's MinAtar PQN also builds (mttga/purejaxql
#: @47af6d7, purejaxql/pqn_minatar.py:38-50).
MINATAR_CNN = CNNSpec((16,), (3,), (1,), 128, "VALID")


def reference_cnn(image_shape: Tuple[int, int, int]) -> CNNSpec:
    """Mnih et al.'s network when the image fits it, 36 pixels a side or
    more (its three VALID convolutions leave nothing of less), else
    MinAtar's, made for MinAtar's 10 x 10 images."""
    if min(image_shape[:2]) < 3:
        raise ValueError(
            f"No reference network reads a {image_shape[:2]} image (MinAtar's"
            " needs 3 pixels a side); pass a cnn_spec."
        )
    return MNIH_CNN if min(image_shape[:2]) >= 36 else MINATAR_CNN


def _conv(
    features: int, kernel: Tuple, strides: Tuple, spec: CNNSpec, in_channels: int
) -> nn.Conv:
    """``spec``'s convolution; torch_linear's bound reads the fan-in, the
    kernel's area times the input channels."""
    bias_init = constant(0.0)
    if spec.kernel_init == TORCH_LINEAR:
        bound = 1.0 / math.sqrt(kernel[0] * kernel[1] * in_channels)
        kernel_init = bias_init = uniform_init(bound)
    else:
        kernel_init = parse_initialization(spec.kernel_init, lecun_normal())
    return nn.Conv(
        features,
        kernel_size=kernel,
        strides=strides,
        padding=spec.padding,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )


def _maybe_norm(spec: CNNSpec, x: jax.Array) -> jax.Array:
    return nn.LayerNorm()(x) if spec.layer_norm else x


def observation_encoder(
    network: nn.Module,
    architecture: Optional[Sequence[Union[str, ActivationFunction]]] = None,
) -> nn.Module:
    """The layers ``network`` (an Actor, a Critic or a MultiCritic) reads
    its observations with, before any memory: a :class:`CNNEncoder` for
    image observations, else the MLP :class:`Encoder` of ``architecture``
    (None: the network's ``input_architecture``). Built in the network's
    setup, so it is the network's own submodule. A network with an input
    embedding (Ni et al.'s layout) reads its parts through a
    :class:`PartEmbedding` instead."""
    if getattr(network, "input_embedding", ()):
        return PartEmbedding(network.input_embedding, network.embedding_kernel_init)
    if network.cnn_image_shape is not None:
        return CNNEncoder(
            network.cnn_image_shape, network.cnn_extra_obs_dim, network.cnn_spec
        )
    return Encoder(
        input_architecture=(
            network.input_architecture if architecture is None else architecture
        ),
        penultimate_normalization=network.penultimate_normalization,
        kernel_init=network.encoder_kernel_init,
        layer_norm=network.encoder_layer_norm,
    )


#: Ni et al.'s input embedding as a network reads it
#: (NetworkConfig.input_embedding): per part, its name, its columns as
#: (start, stop) ranges of the observation, and its embedding's width.
EmbeddingParts = Tuple[Tuple[str, Tuple[Tuple[int, int], ...], int], ...]


def take_columns(x: jax.Array, ranges: Tuple[Tuple[int, int], ...]) -> jax.Array:
    """The columns of x in ``ranges``, joined in order."""
    return jnp.concatenate([x[..., start:stop] for start, stop in ranges], axis=-1)


class PartEmbedding(nn.Module):
    """Each part of the input through its own Dense layer and ReLU, the
    results joined in order: Ni et al. (2022)'s embedders before their
    memory, one torch Linear and ReLU per input, joined as (action, reward,
    observation) (pomdp-baselines e7c19c3, utils/helpers.py:201-226;
    policies/models/recurrent_actor.py:36-46, :101-109)."""

    parts: EmbeddingParts
    kernel_init: Optional[Union[str, InitializationFunction]] = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        embeddings = []
        for name, ranges, width in self.parts:
            layer = dense(width, self.kernel_init, None, lecun_normal(), name=name)
            embeddings.append(nn.relu(layer(take_columns(x, ranges))))
        return jnp.concatenate(embeddings, axis=-1)


def shortcut_layer(network: nn.Module) -> Optional[nn.Module]:
    """Ni et al.'s shortcut of ``network`` (its ``shortcut_width``), built in
    its setup; None without one."""
    if not network.shortcut_width:
        return None
    return dense(
        network.shortcut_width,
        network.embedding_kernel_init,
        None,
        lecun_normal(),
        name="shortcut",
    )


def observation_ranges(parts: EmbeddingParts) -> Tuple[Tuple[int, int], ...]:
    """The columns of the observation part of an input embedding."""
    for name, ranges, _ in parts:
        if name == "observation":
            return ranges
    raise ValueError("An input embedding with a shortcut needs an 'observation' part.")


def join_after_memory(
    network: nn.Module,
    features: jax.Array,
    observation: jax.Array,
    query: Optional[jax.Array] = None,
) -> jax.Array:
    """What a network's layers after its memory read: the memory's features
    joined with Ni et al.'s shortcut, the current observation (with the
    action asked about, ``query``) through one Dense layer and ReLU
    (pomdp-baselines e7c19c3, recurrent_actor.py:76-80, :136-140;
    recurrent_critic.py:70-82, :164-182), or, without a shortcut, with the
    action asked about alone. ``observation`` is what the memory read, the
    observation part's columns of it are the shortcut's."""
    if network.shortcut_width:
        current = take_columns(observation, observation_ranges(network.input_embedding))
        if query is not None:
            current = jnp.concatenate([current, query], axis=-1)
        return jnp.concatenate([features, nn.relu(network.shortcut(current))], axis=-1)
    if query is not None:
        return jnp.concatenate([features, query], axis=-1)
    return features


def head_layers(
    architecture: Sequence[Union[str, ActivationFunction]],
    kernel_init: Optional[Union[str, InitializationFunction]],
    encoder_kernel_init: Optional[Union[str, InitializationFunction]],
) -> Callable[[jax.Array], jax.Array]:
    """The hidden layers a network's output layers read its trunk (encoder,
    then memory block) through (NetworkConfig.head_architecture), started
    with ``kernel_init``, else the encoder's; none: the features as they
    are. Built in the network's setup, so they are the network's own."""
    if not architecture:
        return lambda features: features
    if kernel_init is None:
        kernel_init = encoder_kernel_init
    return nn.Sequential(parse_architecture(architecture, kernel_init))


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
            img = _conv(c, ks, st, spec, img.shape[-1])(img)
            img = nn.relu(_maybe_norm(spec, img))
        # Flatten the spatial+channel dims while preserving leading batch dims.
        img = img.reshape(*img.shape[:-3], -1)
        feat = dense(spec.feature_dim, spec.kernel_init, None, lecun_normal())(img)
        feat = nn.relu(_maybe_norm(spec, feat))
        if extra is not None:
            feat = jnp.concatenate([feat, extra], axis=-1)
        return feat


def check_log_std_head(
    state_independent: bool,
    log_std_init: Optional[Union[float, str]],
    log_std_kernel_init: Optional[Union[str, InitializationFunction]],
) -> None:
    """Refuse a log-std head whose start is set twice or not at all: the
    learned parameter needs a float ``log_std_init`` and has no kernel; the
    Dense head takes ``log_std_init`` as its bias (a float: a constant; a
    string: an initializer) unless its kernel init is "torch_linear", whose
    layer draws the bias (``log_std_init`` None)."""
    if state_independent:
        if not isinstance(log_std_init, (int, float)) or log_std_kernel_init:
            raise ValueError(
                "A state-independent log-std is one parameter: it needs a"
                " float log_std_init and takes no log_std_kernel_init."
            )
    elif (log_std_kernel_init == TORCH_LINEAR) != (log_std_init is None):
        raise ValueError(
            f"A {TORCH_LINEAR!r} log-std head draws its bias: log_std_init must"
            " be None with it, and set otherwise."
        )


class Actor(nn.Module):
    """The stochastic policy of SAC, REDQ, ASAC, AVG, PPO, APO and UDRL: a
    Normal head for continuous actions (:class:`Squashed` bounds it), a
    Categorical one for discrete. Expert guidance acts at the loss level,
    not in the architecture, which keeps the policy unconstrained.
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int
    continuous: bool = False
    penultimate_normalization: bool = False
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # The log-std head, every std starting at exp(log_std_init). False (the
    # default here): a Dense layer over the encoder's features
    # (state-dependent, SAC's). True: one learned log-std per action
    # dimension. Either is clipped to ``log_std_bounds`` (None: not
    # clipped). PPO's default is True, log_std_init 0 and no clip, as its
    # reference code learns it: OpenAI Baselines @ea25b9e (ppo2's
    # common/distributions.py:96-106; ppo1's ppo1/mlp_policy.py:36-39,
    # gaussian_fixed_var=True), CleanRL @fe8d8a0
    # (ppo_continuous_action.py:129) and PureJaxRL @31756b1
    # (ppo_continuous_action.py:41).
    log_std_state_independent: bool = False
    # The head's output read as a log-variance, std = exp(out / 2), after
    # ``log_std_bounds`` (NetworkConfig.log_variance_head).
    log_variance_head: bool = False
    # None only for a "torch_linear" log-std head, whose layer draws its
    # bias: the starting std is then the layer's, not one constant.
    log_std_init: Optional[Union[float, str]] = -1.0
    log_std_bounds: Optional[Tuple[float, float]] = LOG_STD_BOUNDS
    # The output layers' kernel inits (``parse_initialization`` strings or
    # callables, or "torch_linear"), so an agent can start them as its
    # reference code does. None keeps orthogonal(0.01) for the logits (the
    # comment in setup cites it) and a zero kernel for the state-dependent
    # log-std head (std=exp(log_std_init) at init); the mean's is
    # ``mean_kernel_init`` below.
    logits_kernel_init: Optional[Union[str, InitializationFunction]] = None
    log_std_kernel_init: Optional[Union[str, InitializationFunction]] = None
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
    # The mean head's output through tanh, so the Gaussian's mean, not its
    # samples, stays in [-1, 1]: APO's reference code's head, xtma/rlpyt
    # @e28bb20 (rlpyt/models/pg/mujoco_ff_model.py:23, :40-41,
    # mu_nonlinearity=torch.nn.Tanh).
    tanh_mean: bool = False
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
    # The hidden layers the output layers read the memory's (or the
    # encoder's) features through (NetworkConfig.head_architecture).
    head_architecture: Sequence[Union[str, ActivationFunction]] = ()
    head_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # Ni et al.'s layout (NetworkConfig.input_embedding): the memory reads
    # the parts' embeddings, and input_architecture runs after it (as
    # ``head``) on its output joined with the shortcut (shortcut_width,
    # join_after_memory).
    input_embedding: EmbeddingParts = ()
    shortcut_width: int = 0
    embedding_kernel_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
        check_log_std_bounds(self.log_std_bounds)
        if self.memory is not None:
            self.memory_cell = MemoryCell(self.memory)
        self.encoder = observation_encoder(self)
        if self.input_embedding:
            self.head = Encoder(
                self.input_architecture, kernel_init=self.encoder_kernel_init
            )
            self.shortcut = shortcut_layer(self)
        self.head_layers = head_layers(
            self.head_architecture, self.head_kernel_init, self.encoder_kernel_init
        )
        # Discrete logits start near 0 (a near-uniform policy) unless
        # logits_kernel_init sets them, as in OpenAI Baselines @ea25b9e
        # (common/policies.py:49, init_scale=0.01), CleanRL @fe8d8a0
        # (ppo.py:115), PureJaxRL @31756b1 (ppo.py:33) and SB3 2.5.0
        # (common/policies.py:620), and like the mean head below.
        if self.continuous:
            check_log_std_head(
                self.log_std_state_independent,
                self.log_std_init,
                self.log_std_kernel_init,
            )
            self.mean = dense(
                self.action_dim,
                self.mean_kernel_init,
                self.bias_init,
                orthogonal(0.01),
                name="mean",
            )
            # The state-dependent head's zero kernel outputs its bias at
            # init: std=exp(log_std_init). The state-independent Param
            # lives outside the encoder so exploration does not couple to
            # the value-estimating features; flax requires it declared in
            # setup() (or an @nn.compact method).
            if not self.log_std_state_independent:
                # A float log_std_init is the bias's constant; a string is
                # its initializer (Ni et al.'s uniform, actor.py:118-121).
                bias_init = self.log_std_init
                if isinstance(bias_init, (int, float)):
                    bias_init = constant(bias_init)
                self.log_std = dense(
                    self.action_dim,
                    self.log_std_kernel_init,
                    bias_init,
                    nn.initializers.zeros,
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
                    dense(
                        self.action_dim,
                        self.logits_kernel_init,
                        self.bias_init,
                        orthogonal(0.01),
                    ),
                    distrax.Categorical,
                ],
            )

    def _distribution(self, embedding) -> distrax.Distribution:
        embedding = self.head_layers(embedding)
        if self.continuous:
            mean = self.mean(embedding)
            if self.tanh_mean:
                mean = jnp.tanh(mean)
            if self.log_std_state_independent:
                log_std = jnp.broadcast_to(self.log_std_param, mean.shape)
            else:
                log_std = self.log_std(embedding)
            if self.log_std_bounds is not None:
                log_std = jnp.clip(log_std, *self.log_std_bounds)
            std = jnp.exp(0.5 * log_std if self.log_variance_head else log_std)
            return distrax.Normal(mean, std)
        return self.model(embedding)

    def __call__(self, obs, hidden_state=None, done=None):
        embedding = self.encoder(obs)
        if self.memory is not None:
            if hidden_state is None or done is None:
                raise ValueError(
                    "Recurrent Actor requires hidden_state and done flags."
                )
            hidden_state, embedding = self.memory_cell(hidden_state, embedding, done)
            if self.input_embedding:
                embedding = self.head(join_after_memory(self, embedding, obs))
            return self._distribution(embedding), hidden_state
        return self._distribution(embedding)


class Squashed(nn.Module):
    """``actor`` with its policy bounded by ``tanh`` (:func:`squash`): any
    actor, stochastic or deterministic, feedforward or recurrent. The two
    share one scope, so the parameters are ``actor``'s alone."""

    actor: nn.Module
    log_det_eps: float = 0.0  # Squash's: 0 keeps the exact log-det
    sigma_point_mean: bool = False  # Squash's: False, tanh of the latent's mean

    def setup(self):
        nn.share_scope(self, self.actor)

    def __call__(self, *args, **kwargs):
        out = self.actor(*args, **kwargs)
        if isinstance(out, tuple):  # a recurrent actor's (policy, carry)
            policy = squash(out[0], self.log_det_eps, self.sigma_point_mean)
            return (policy, *out[1:])
        return squash(out, self.log_det_eps, self.sigma_point_mean)


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
    # The value layer's kernel init (as Actor's output layers'); None keeps
    # orthogonal(1.0).
    value_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # The head's outputs: one value, or a distributional critic's quantiles
    # of the return (TQC's), whose mean __call__ returns.
    n_quantiles: int = 1
    # A member of a MultiCritic sharing its encoder (share_encoder): it reads
    # the encoder's features and is its value layer alone, or, for a
    # Q-critic (query_dim), the features with the action through a head of
    # its own architecture, as SAC-AE's critic (the action joins after the
    # encoder).
    reads_features: bool = False
    # The hidden layers the value layer reads the features through, after
    # the memory and the action (NetworkConfig.head_architecture).
    head_architecture: Sequence[Union[str, ActivationFunction]] = ()
    head_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # Ni et al.'s layout, as Actor's: with an input embedding, the memory
    # reads its parts' embeddings and ``head`` (input_architecture) reads
    # the memory's output joined with the shortcut, which a Q-critic's reads
    # with the action asked about (Ni et al.'s critic, recurrent_critic.py:
    # 70-99).
    input_embedding: EmbeddingParts = ()
    shortcut_width: int = 0
    embedding_kernel_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
        self.head_layers = head_layers(
            self.head_architecture, self.head_kernel_init, self.encoder_kernel_init
        )
        if self.reads_features:
            if self.query_dim or self.input_embedding:
                self.head = Encoder(
                    input_architecture=self.input_architecture,
                    penultimate_normalization=self.penultimate_normalization,
                    kernel_init=self.encoder_kernel_init,
                    layer_norm=self.encoder_layer_norm,
                )
            self.model = dense(
                self.n_quantiles,
                self.value_kernel_init,
                self.bias_init,
                orthogonal(1.0),
            )
            return
        if self.memory is not None:
            self.memory_cell = MemoryCell(self.memory)
            if self.query_dim or self.input_embedding:
                self.head = Encoder(
                    input_architecture=self.input_architecture,
                    kernel_init=self.encoder_kernel_init,
                    layer_norm=self.encoder_layer_norm,
                )
            self.shortcut = shortcut_layer(self)
        self.encoder = observation_encoder(self)
        self.model = dense(
            self.n_quantiles, self.value_kernel_init, self.bias_init, orthogonal(1.0)
        )

    def __call__(self, x: jax.Array, hidden_state=None, done=None):
        """The value: the head's output, or the mean of its quantiles."""
        out = self.quantiles(x, hidden_state, done)
        if self.n_quantiles == 1:
            return out
        if self.memory is None:
            return out.mean(-1, keepdims=True)
        values, hidden_state = out
        return values.mean(-1, keepdims=True), hidden_state

    def quantiles(self, x: jax.Array, hidden_state=None, done=None):
        """The head's ``n_quantiles`` outputs (and the carry, with memory)."""
        if self.reads_features:
            if self.query_dim or self.input_embedding:
                x = self.head(x)
            return self.model(self.head_layers(x))
        if self.memory is None:
            return self.model(self.head_layers(self.encoder(x)))
        if hidden_state is None or done is None:
            raise ValueError("Recurrent Critic requires hidden_state and done flags.")
        query = None
        if self.query_dim:
            x, query = x[..., : -self.query_dim], x[..., -self.query_dim :]
        hidden_state, feat = self.memory_cell(hidden_state, self.encoder(x), done)
        if query is not None or self.input_embedding:
            feat = self.head(join_after_memory(self, feat, x, query))
        return self.model(self.head_layers(feat)), hidden_state

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
    value_kernel_init: Optional[Union[str, InitializationFunction]] = None
    n_quantiles: int = 1  # see Critic.n_quantiles
    # One encoder and memory block for every member, outside the ensemble
    # (NetworkConfig.share_encoder), its parameters under "encoder" and
    # "memory_cell" as the actor's would be; the members are value layers
    # on its features. Its one carry keeps the ensemble layout, with one
    # member: hidden_state leaves are (1, batch, hidden).
    share_encoder: bool = False
    # The shared encoder's layers: the actor's architecture, which the actor
    # reads it with. A Q-critic's members keep input_architecture for their
    # heads over its features and the action. None: input_architecture.
    encoder_architecture: Optional[Sequence[Union[str, ActivationFunction]]] = None
    # Each member's hidden layers before its value layer (Critic's).
    head_architecture: Sequence[Union[str, ActivationFunction]] = ()
    head_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # Ni et al.'s layout (Critic's). With share_encoder, the input embedding,
    # the memory and the shortcut are the members' one trunk, as Ni et al.'s
    # qf1 and qf2 read one RNN (recurrent_critic.py:95-99, :184-186).
    input_embedding: EmbeddingParts = ()
    shortcut_width: int = 0
    embedding_kernel_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
        if self.share_encoder:
            if self.memory is not None:
                self.memory_cell = MemoryCell(self.memory)
            self.encoder = observation_encoder(self, self.encoder_architecture)
            self.shortcut = shortcut_layer(self)
        # x (and done) are broadcast across the ensemble; the carry is
        # mapped over its leading (num,) axis so each member evolves its
        # own memory. NOTE: the recurrent in_axes spec matches __call__'s
        # 3-arg signature; apply_encoder (1 arg) is only used by
        # ConditioningMetrics, which therefore needs a feedforward critic.
        member_memory = None if self.share_encoder else self.memory
        in_axes = (None, 0, None) if member_memory is not None else None
        Vmapped = nn.vmap(
            target=Critic,
            in_axes=in_axes,
            out_axes=0,
            # "intermediates": a capture reads each member's own activations.
            variable_axes={"params": 0, "batch_stats": 0, "intermediates": 0},
            split_rngs={"params": True},
            axis_size=self.num,
            methods=("__call__", "apply_encoder", "quantiles"),
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
            memory=member_memory,
            query_dim=self.query_dim,
            value_kernel_init=self.value_kernel_init,
            n_quantiles=self.n_quantiles,
            reads_features=self.share_encoder,
            head_architecture=self.head_architecture,
            head_kernel_init=self.head_kernel_init,
            input_embedding=self.input_embedding,
            shortcut_width=0 if self.share_encoder else self.shortcut_width,
            embedding_kernel_init=self.embedding_kernel_init,
        )

    def __call__(self, x: jax.Array, hidden_state=None, done=None):
        if self.share_encoder:
            features, hidden_state = self._shared_features(x, hidden_state, done)
            values = self.ensemble(features)
            return values if self.memory is None else (values, hidden_state)
        if self.memory is not None:
            # Returns (values, new_hidden_state): values (num, T, B, 1),
            # hidden leaves (num, B, hidden).
            return self.ensemble(x, hidden_state, done)
        return self.ensemble(x)

    def quantiles(self, x: jax.Array, hidden_state=None, done=None):
        """Every member's quantiles, the ensemble axis first."""
        if self.share_encoder:
            features, hidden_state = self._shared_features(x, hidden_state, done)
            quantiles = self.ensemble.quantiles(features)
            return quantiles if self.memory is None else (quantiles, hidden_state)
        if self.memory is not None:
            return self.ensemble.quantiles(x, hidden_state, done)
        return self.ensemble.quantiles(x)

    def _shared_features(self, x: jax.Array, hidden_state, done) -> tuple:
        """The shared encoder's features at x, through the memory block if
        there is one, and the memory's next carry (None without one). A
        Q-critic's action (the last query_dim inputs) skips the encoder and
        the memory and joins their features, so asking about another action
        never changes the memory (Critic.query_dim)."""
        action = None
        if self.query_dim:
            x, action = x[..., : -self.query_dim], x[..., -self.query_dim :]
        features = self.encoder(x)
        if self.memory is not None:
            if hidden_state is None or done is None:
                raise ValueError(
                    "Recurrent Critic requires hidden_state and done flags."
                )
            carry = jax.tree.map(lambda leaf: leaf[0], hidden_state)
            carry, features = self.memory_cell(carry, features, done)
            hidden_state = jax.tree.map(lambda leaf: leaf[None], carry)
        return join_after_memory(self, features, x, action), hidden_state

    def apply_encoder_ensemble(self, x: jax.Array) -> jax.Array:
        """Per-ensemble-member encoder features, shape ``(num, ..., d)``
        (:class:`ajax.extensions.instrumentation.ConditioningMetrics`); a
        shared encoder's, on an ensemble axis of one."""
        if self.share_encoder:
            if self.query_dim:
                x = x[..., : -self.query_dim]
            return self.encoder(x)[None]
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


def check_share_encoder(
    network_config: NetworkConfig, action_value: bool, observers: Sequence = ()
) -> None:
    """Refuse a shared encoder (``network_config.share_encoder``) the
    networks cannot have, the run's observation features being
    ``observers``."""
    if not network_config.share_encoder:
        return
    if isinstance(network_config.actor_architecture, PIDPolicy):
        raise ValueError("share_encoder needs an encoder: a PID actor has none.")
    memory = network_config.memory is not None
    if action_value and memory and not reads_previous_action(observers):
        raise ValueError(
            "A memory shared with a Q-critic reads the previous action, as the"
            " critic's does (Ni et al. 2022), so the actor must read it too:"
            " add PreviousAction() (ajax.extensions.observe) to the extensions."
        )
    architectures = (
        network_config.actor_architecture,
        network_config.critic_architecture,
    )
    if any("batchrenorm" in architecture for architecture in architectures):
        raise ValueError(
            "A shared encoder's running statistics are the critic's, which the"
            " actor's own training pass would also move: share_encoder takes no"
            " 'batchrenorm' layer."
        )
    if action_value:
        # The encoder is the actor's architecture; the critic's own layers
        # read its features with the action.
        return
    if network_config.cnn_image_shape is None and tuple(
        network_config.actor_architecture
    ) != tuple(network_config.critic_architecture):
        raise ValueError(
            "With share_encoder one encoder reads for both networks, so"
            " actor_architecture and critic_architecture must be the same, got"
            f" {network_config.actor_architecture} and"
            f" {network_config.critic_architecture}."
        )


def embedding_parts(
    network_config: NetworkConfig, env_config: EnvironmentConfig, extra_obs_dim: int
) -> EmbeddingParts:
    """``network_config.input_embedding`` as the networks read it
    (:data:`EmbeddingParts`): each part's columns in the observation with
    its features (``env_config.observers``); the "observation" part takes
    every column no other part names."""
    if not network_config.input_embedding:
        return ()
    if network_config.memory is None:
        raise ValueError(
            "input_embedding is Ni et al.'s layout around a memory block: it"
            " needs memory."
        )
    if network_config.cnn_image_shape is not None or extra_obs_dim:
        raise ValueError(
            "input_embedding reads flat observations with their features alone:"
            " no CNN, no command appended."
        )
    env_width = get_state_action_shapes(env_config.env)[0][-1]
    layout = observation_layout(env_width, env_config.observers, env_config)
    named: dict[str, Tuple[int, int]] = {}
    for name, _ in network_config.input_embedding:
        if name == "observation":
            continue
        try:
            columns = layout.slice(name)
        except KeyError as error:
            raise ValueError(
                f"input_embedding's part {name!r} is an observation feature's"
                " columns: add the feature (ajax.extensions.observe) to the"
                " extensions."
            ) from error
        named[name] = (columns.start, columns.stop)
    rest = []
    start = 0
    for begin, end in sorted(named.values()):
        if begin > start:
            rest.append((start, begin))
        start = end
    if start < layout.width:
        rest.append((start, layout.width))
    parts = []
    for name, width in network_config.input_embedding:
        ranges = tuple(rest) if name == "observation" else (named[name],)
        parts.append((name, ranges, width))
    return tuple(parts)


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
    extra_obs_dim: int = 0,
    actor: Optional[nn.Module] = None,
) -> Tuple[LoadedTrainState, LoadedTrainState]:
    """
    Create actor and critic networks, their initializers and log-std head
    as ``network_config`` sets them; a prebuilt ``actor`` (TD3's
    deterministic one) replaces the :class:`Actor` built here. ``squash``
    wraps a continuous actor in :class:`Squashed` (a discrete actor keeps
    its Categorical).

    extra_obs_dim: dimensions the agent appends to obs at run time (UDRL's
    command), so the network is initialised on the input width it will be
    fed; the observation features (``env_config.observers``) join them.

    A :class:`PIDPolicy` as ``network_config.actor_architecture`` makes the
    actor a :class:`PIDActor` (see :mod:`ajax.modules.pid_actor`). The
    actions are the policy's (:func:`ajax.controller.policy_action_dim`:
    gain mode's gains).

    With ``network_config.share_encoder`` the two read one encoder (and
    memory block): the critic's, which its optimiser steps; the actor's
    state holds its heads and a copy of it (``shared_params``).
    """
    check_share_encoder(network_config, action_value, env_config.observers)
    action_dim = policy_action_dim(env_config)

    memory = parse_memory_config(network_config.memory)
    parts = embedding_parts(network_config, env_config, extra_obs_dim)
    actor_shortcut, critic_shortcut = network_config.shortcut_widths

    # A CNN critic is only valid for state-value critics; an action-value
    # critic (SAC's Q(s,a)) keeps the MLP.
    cnn_image_shape = network_config.cnn_image_shape
    critic_cnn_image_shape = None if action_value else cnn_image_shape
    cnn_spec = network_config.cnn_spec
    # Dims appended to the env's observation at run time: UDRL's command
    # and the observation features. A CNN passes them through after the
    # convolutions, with the env's own trailing dims, the same way in the
    # actor and the critic.
    obs_extra = extra_obs_dim + feature_width(env_config)  # with the features
    cnn_extra_obs_dim = network_config.cnn_extra_obs_dim + obs_extra

    if actor is not None:
        pass
    elif isinstance(network_config.actor_architecture, PIDPolicy):
        check_pid_actor(continuous, memory)
        # The agent's mean and log-std settings (kernel inits, log_std_init)
        # are for its Gaussian MLP actor's head, often its reference code's:
        # the PID actor in its place has one learned log-std parameter and
        # always starts it at its own PID_LOG_STD_INIT, whatever the agent.
        policy = network_config.actor_architecture
        actor = PIDActor(
            policy=policy,
            action_dim=action_dim,
            tanh_mean=network_config.tanh_mean,
            log_std_bounds=network_config.log_std_bounds,
            terms_at=pid_terms_at(env_config, policy.n_errors),
        )
    else:
        actor = Actor(
            input_architecture=network_config.actor_architecture,
            action_dim=action_dim,
            continuous=continuous,
            penultimate_normalization=network_config.penultimate_normalization,
            bias_init=network_config.actor_bias_init,
            encoder_kernel_init=network_config.encoder_kernel_init,
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=cnn_extra_obs_dim,
            cnn_spec=cnn_spec,
            log_std_state_independent=network_config.log_std_state_independent,
            log_variance_head=network_config.log_variance_head,
            log_std_init=network_config.log_std_init,
            log_std_bounds=network_config.log_std_bounds,
            mean_kernel_init=network_config.mean_kernel_init,
            logits_kernel_init=network_config.logits_kernel_init,
            log_std_kernel_init=network_config.log_std_kernel_init,
            tanh_mean=network_config.tanh_mean,
            encoder_layer_norm=network_config.encoder_layer_norm,
            memory=memory,
            head_architecture=network_config.head_architecture,
            head_kernel_init=network_config.head_kernel_init,
            input_embedding=parts,
            shortcut_width=actor_shortcut,
            embedding_kernel_init=network_config.embedding_kernel_init,
        )
    if squash and continuous:  # a discrete policy has no range to bound
        actor = Squashed(
            actor,
            network_config.squash_log_det_eps,
            network_config.squash_sigma_point_mean,
        )
    # A shared encoder is the actor's (check_share_encoder refuses a PID actor).
    actor_architecture = network_config.actor_architecture
    shared_architecture = (
        None
        if isinstance(actor_architecture, PIDPolicy) or not network_config.share_encoder
        else actor_architecture
    )
    # The critics read one encoder when the actor shares it too.
    members_share = network_config.share_encoder or network_config.critics_share_encoder
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
        # The action joins after a recurrent or a shared encoder.
        query_dim=(
            action_dim if action_value and (memory is not None or members_share) else 0
        ),
        value_kernel_init=network_config.value_kernel_init,
        n_quantiles=network_config.critic_quantiles,
        share_encoder=members_share,
        encoder_architecture=shared_architecture,
        head_architecture=network_config.head_architecture,
        head_kernel_init=network_config.head_kernel_init,
        input_embedding=parts,
        shortcut_width=critic_shortcut,
        embedding_kernel_init=network_config.embedding_kernel_init,
    )

    actor_tx = get_adam_tx(**to_state_dict(actor_optimizer_config))
    critic_tx = get_adam_tx(**to_state_dict(critic_optimizer_config), per_member=True)
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
    if env_config.policy_action_dim is not None:
        action_shape = (env_config.policy_action_dim,)
    init_action = jnp.zeros((1, *action_shape))

    actor_state = init_network_state(
        init_x=init_obs,
        network=actor,
        key=actor_key,
        tx=actor_tx,
        memory=memory,
        n_envs=env_config.n_envs,
    )
    # A recurrent Q-critic's memory reads the action taken before each
    # observation: in the observation where it carries it (PreviousAction),
    # else appended to it.
    in_observation = reads_previous_action(env_config.observers)
    appends_previous_action = action_value and memory is not None and not in_observation
    critic_state = init_network_state(
        init_x=(
            action_value_input(
                init_obs, init_action, init_action if appends_previous_action else None
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
    critic_state = critic_state.replace(previous_action_in_observation=in_observation)
    if network_config.share_encoder:
        if "encoder" not in actor_state.params["params"]:
            raise ValueError(
                "share_encoder needs an actor reading its observations through"
                f" an encoder; {type(actor).__name__} has none."
            )
        actor_state = _borrow_encoder(
            actor_state, shared_encoder_params(critic_state.params)
        )
    return actor_state, critic_state


def get_initialized_critic(
    key: jax.Array,
    env_config: EnvironmentConfig,
    critic_optimizer_config: OptimizerConfig,
    network_config: NetworkConfig,
    num_critics: int = 2,
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
        head_architecture=network_config.head_architecture,
    )

    critic_tx = get_adam_tx(**to_state_dict(critic_optimizer_config), per_member=True)

    observation_shape, action_shape = get_state_action_shapes(env_config.env)

    obs_extra = extra_obs_dim + feature_width(env_config)  # with the features
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


def shared_encoder_params(critic_params: Any) -> FrozenDict:
    """The parameters of the encoder (and memory block) a critic shares with
    its actor, taken from the critic's parameters."""
    owned = critic_params["params"]
    return FrozenDict(
        {name: owned[name] for name in SHARED_ENCODER_MODULES if name in owned}
    )


def with_critic_encoder(
    actor_state: LoadedTrainState, critic_params: Any
) -> LoadedTrainState:
    """``actor_state`` reading the encoder in ``critic_params``, for an actor
    sharing the critic's encoder; an actor with its own is returned as is. A
    loss differentiated with respect to ``critic_params`` then trains the
    encoder through the actor too."""
    if actor_state.shared_params is None:
        return actor_state
    return actor_state.replace(shared_params=shared_encoder_params(critic_params))


def mirror_shared_encoder(agent_state: Any) -> Any:
    """``agent_state`` with its actor's copy of the shared encoder taken from
    the critic as it is now: the training loop calls it after anything that
    steps the critic (each update, the extensions' pretraining), so acting
    and evaluating read the current encoder. Unchanged without one."""
    actor_state: Any = getattr(agent_state, "actor_state", None)
    if getattr(actor_state, "shared_params", None) is None:
        return agent_state
    critic_params = agent_state.critic_state.params
    return agent_state.replace(
        actor_state=with_critic_encoder(actor_state, critic_params)
    )


def take_actor_encoder(agent_state: Any) -> Any:
    """``agent_state`` with the critic's shared encoder replaced by the
    actor's copy, after something trained the encoder through the actor
    alone (cloning pretraining); the target critic's too, which starts as
    the critic. Unchanged without a shared encoder."""
    actor_state = agent_state.actor_state
    if actor_state.shared_params is None:
        return agent_state
    critic_state = agent_state.critic_state

    def with_encoder(params: Any) -> FrozenDict:
        owned = {**params["params"], **actor_state.shared_params}
        return FrozenDict({**params, "params": owned})

    critic_state = critic_state.replace(
        params=with_encoder(critic_state.params),
        target_params=with_encoder(critic_state.target_params),
    )
    return agent_state.replace(critic_state=critic_state)


def _borrow_encoder(
    actor_state: LoadedTrainState, encoder: FrozenDict
) -> LoadedTrainState:
    """``actor_state`` without its own encoder, reading ``encoder`` instead:
    its parameters, and its optimiser's state, rebuilt on what it still
    owns (its heads)."""
    own = {
        name: value
        for name, value in actor_state.params["params"].items()
        if name not in encoder
    }
    params: FrozenDict = FrozenDict({**actor_state.params, "params": own})
    return LoadedTrainState.create(
        params=params,
        tx=actor_state.tx,
        apply_fn=actor_state.apply_fn,
        hidden_state=actor_state.hidden_state,
        target_params=params,
        batch_stats=actor_state.batch_stats,
    ).replace(shared_params=encoder)


def init_network_carry(network, memory: MemoryConfig, key: jax.Array, batch_size: int):
    """Fresh carry for ``network``. Ensembles (modules exposing ``num``)
    get one carry per member, stacked on a leading (num,) axis; members
    sharing one memory block (MultiCritic.share_encoder) have one carry,
    on an axis of one."""
    carry = init_carry(memory, key, batch_size)
    num = getattr(network, "num", None)
    if getattr(network, "share_encoder", False):
        num = 1
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
        variables = dict(network.init(key, init_x))
        hidden_state = None
    else:
        # The parameters initialise on key as without memory, so a layer
        # both networks have starts from the same weights; the carry draws
        # on a key derived from it.
        carry_key = derive_key(key, "memory_carry")
        # The live carry is batch-sized to n_envs; the init forward pass
        # uses a carry matching init_x's (possibly smaller) batch — param
        # shapes don't depend on the batch dim, so init with batch=1 obs
        # (see get_initialized_actor_critic) stays cheap.
        hidden_state = init_network_carry(network, memory, carry_key, n_envs)
        init_carry_batch = init_network_carry(
            network, memory, carry_key, init_x.shape[0]
        )
        variables = dict(
            network.init(
                key,
                init_x[None, ...],
                hidden_state=init_carry_batch,
                done=jnp.zeros((1, init_x.shape[0]), dtype=bool),
            )
        )
    # Running statistics (BatchRenorm's) live apart from the trained params.
    batch_stats = variables.pop("batch_stats", None)
    params: FrozenDict[str, Any] = FrozenDict(variables)
    return LoadedTrainState.create(
        params=params,
        tx=tx,
        apply_fn=network.apply,
        hidden_state=hidden_state,
        recurrent=memory is not None,
        target_params=params,
        batch_stats=None if batch_stats is None else FrozenDict(batch_stats),
    )


def in_critic_units(critic_state: LoadedTrainState, actions: jax.Array) -> jax.Array:
    """``actions`` as the critic reads them: ``critic_state.action_units``'
    affine map of the agent's [-1, 1] actions, or the actions themselves."""
    units = getattr(critic_state, "action_units", None)
    if units is None:
        return actions
    offset, scale = (jnp.asarray(u, actions.dtype) for u in units)
    return offset + scale * actions


def _apply_critic_action_units(
    critic_state: LoadedTrainState, x: jax.Array
) -> jax.Array:
    """The action slice of a Q-critic's input ``concat(obs, action)`` in the
    critic's units (:func:`in_critic_units`): every caller passes the
    agent's [-1, 1] actions."""
    units = getattr(critic_state, "action_units", None)
    if units is None:
        return x
    k = len(units[0])
    return jnp.concatenate(
        [x[..., :-k], in_critic_units(critic_state, x[..., -k:])], axis=-1
    )


def _apply_critic_obs_norm(critic_state: LoadedTrainState, x: jax.Array) -> jax.Array:
    """Normalise the obs slice of a critic input (obs or concat(obs, action));
    the action slice stays raw (``apply_obs_norm``). No-op when
    normalisation is disabled."""
    return apply_obs_norm(x, getattr(critic_state, "obs_norm_info", None))


def critic_input(critic_state: LoadedTrainState, x: jax.Array) -> jax.Array:
    """``x`` (obs, or ``concat(obs, action)`` with the agent's action) as a
    feedforward critic reads it: its observations normalised, its action in
    its units."""
    return _apply_critic_action_units(
        critic_state, _apply_critic_obs_norm(critic_state, x)
    )


def predict_value_sequence(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
    resets: jax.Array,
    initial_hidden: Carry,
    *,
    quantiles: bool = False,
) -> Tuple[jax.Array, Carry]:
    """Run a recurrent critic over a time-major sequence (``quantiles``: a
    distributional critic's quantiles, last axis, in place of the values).

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
        network_variables(critic_state, critic_params),
        x,
        hidden_state=initial_hidden,
        done=resets,
        method="quantiles" if quantiles else None,
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
    quantiles: bool = False,
) -> jax.Array:
    """The critic's values at ``x``: obs, or ``concat(obs, action)`` with the
    agent's [-1, 1] action, which a critic with ``action_units`` reads in its
    units (TD3's, in the env's). ``quantiles``: a distributional critic's
    quantiles (last axis) in place of their mean, the value."""
    x = critic_input(critic_state, x)
    variables = network_variables(critic_state, critic_params)
    if quantiles:
        return critic_state.apply_fn(variables, x, method="quantiles")
    return critic_state.apply_fn(variables, x)


def get_policy_updating_batch_statistics(
    actor_state: LoadedTrainState, params: Any, obs: jax.Array
) -> Tuple[Any, Optional[Any]]:
    """The actor's distribution at obs in training mode, and its batch
    normalisation's running statistics after this pass (apply_train).

    In training mode a batch normalisation layer (CrossQ's batch
    renormalisation) normalises with the batch's own statistics and updates
    its running averages; acting and the targets use the running averages
    instead. A feedforward actor only."""
    obs = apply_obs_norm(obs, getattr(actor_state, "obs_norm_info", None))
    return actor_state.apply_train(params, obs)


def predict_value_updating_batch_statistics(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
) -> Tuple[jax.Array, Optional[FrozenDict]]:
    """The critic's values at x in training mode, and its batch
    normalisation's running statistics after this pass
    (LoadedTrainState.apply_train); see get_policy_updating_batch_statistics
    for what training mode changes."""
    return critic_state.apply_train(critic_params, critic_input(critic_state, x))


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
