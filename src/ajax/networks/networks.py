import math
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional, Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.linen.initializers import constant, lecun_normal, orthogonal
from flax.serialization import to_state_dict

from ajax.environments.obs_norm import apply_obs_norm
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.modules.pid_actor import (
    PID_LOG_STD_INIT,
    PIDActor,
    PIDPolicy,
    check_pid_actor,
)
from ajax.networks.distributions import squash
from ajax.networks.memory import (
    Carry,
    MemoryCell,
    MemoryConfig,
    init_carry,
    parse_memory_config,
)
from ajax.networks.utils import (
    TORCH_LINEAR,
    dense,
    get_adam_tx,
    parse_architecture,
    parse_initialization,
    uniform_init,
)
from ajax.state import (
    LOG_STD_BOUNDS,
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
    check_log_std_bounds,
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
    log_std_init: Optional[float],
    log_std_kernel_init: Optional[Union[str, InitializationFunction]],
) -> None:
    """Refuse a log-std head whose start is set twice or not at all: the
    learned parameter needs ``log_std_init`` and has no kernel; the Dense
    head takes ``log_std_init`` as its bias unless its kernel init is
    "torch_linear", whose layer draws the bias (``log_std_init`` None)."""
    if state_independent:
        if log_std_init is None or log_std_kernel_init is not None:
            raise ValueError(
                "A state-independent log-std is one parameter: it needs a"
                " float log_std_init and takes no log_std_kernel_init."
            )
    elif (log_std_kernel_init == TORCH_LINEAR) != (log_std_init is None):
        raise ValueError(
            f"A {TORCH_LINEAR!r} log-std head draws its bias: log_std_init must"
            " be None with it, and a float otherwise."
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
    log_std_init: Optional[float] = -1.0
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

    def setup(self):
        check_log_std_bounds(self.log_std_bounds)
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
                self.log_std = dense(
                    self.action_dim,
                    self.log_std_kernel_init,
                    None if self.log_std_init is None else constant(self.log_std_init),
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
            return self._distribution(embedding), hidden_state
        return self._distribution(embedding)


class Squashed(nn.Module):
    """``actor`` with its policy bounded by ``tanh`` (:func:`squash`): any
    actor, stochastic or deterministic, feedforward or recurrent. The two
    share one scope, so the parameters are ``actor``'s alone."""

    actor: nn.Module
    log_det_eps: float = 0.0  # Squash's: 0 keeps the exact log-det

    def setup(self):
        nn.share_scope(self, self.actor)

    def __call__(self, *args, **kwargs):
        out = self.actor(*args, **kwargs)
        if isinstance(out, tuple):  # a recurrent actor's (policy, carry)
            return (squash(out[0], self.log_det_eps), *out[1:])
        return squash(out, self.log_det_eps)


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
        self.model = dense(1, self.value_kernel_init, self.bias_init, orthogonal(1.0))

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
    value_kernel_init: Optional[Union[str, InitializationFunction]] = None

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
            value_kernel_init=self.value_kernel_init,
        )

    def __call__(self, x: jax.Array, hidden_state=None, done=None):
        if self.memory is not None:
            # Returns (values, new_hidden_state): values (num, T, B, 1),
            # hidden leaves (num, B, hidden).
            return self.ensemble(x, hidden_state, done)
        return self.ensemble(x)

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
    deterministic one) replaces the :class:`Actor` built here. ``squash``
    wraps a continuous actor in :class:`Squashed` (a discrete actor keeps
    its Categorical).

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
        # The PID actor's log-std is one learned parameter: the agent's
        # log-std kernel init is for its Dense head and does not apply, and
        # a start that head's layer would draw (None) is the PID actor's own.
        log_std_init = network_config.log_std_init
        actor = PIDActor(
            policy=network_config.actor_architecture,
            action_dim=action_dim,
            tanh_mean=network_config.tanh_mean,
            log_std_init=PID_LOG_STD_INIT if log_std_init is None else log_std_init,
            log_std_bounds=network_config.log_std_bounds,
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
        )
    if squash and continuous:  # a discrete policy has no range to bound
        actor = Squashed(actor, network_config.squash_log_det_eps)
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
        value_kernel_init=network_config.value_kernel_init,
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

    critic_tx = get_adam_tx(**to_state_dict(critic_optimizer_config), per_member=True)

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
) -> jax.Array:
    """The critic's values at ``x``: obs, or ``concat(obs, action)`` with the
    agent's [-1, 1] action, which a critic with ``action_units`` reads in its
    units (TD3's, in the env's)."""
    x = _apply_critic_action_units(
        critic_state, _apply_critic_obs_norm(critic_state, x)
    )
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
