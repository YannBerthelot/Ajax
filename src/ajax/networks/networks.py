from collections.abc import Callable, Sequence
from typing import NamedTuple, Optional, Tuple, Union

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

    def setup(self):
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

    def __call__(self, obs, raw_obs=None) -> distrax.Distribution:
        embedding = self.encoder(obs)
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

    def setup(self):
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

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.model(self.encoder(x))

    def apply_encoder(self, x: jax.Array) -> jax.Array:
        """Expose the encoder's features alone, without the value head.

        Used by auxiliary representation-shaping losses (VAE, RSSM) that
        need to push gradients into the shared encoder while owning
        their own decoder/prior/posterior heads.
        """
        return self.encoder(x)


class SharedActorCritic(nn.Module):
    """Shared-encoder actor-critic (one backbone, two heads).

    For agents where the policy and value share a single feature
    extractor (Atari-style PPO, A2C, IMPALA). Contrasts with the
    legacy Ajax pattern of two independent ``Actor`` + ``Critic``
    networks (used by SAC, brax-tuned PPO on continuous-control).

    Surface mirrors :class:`Actor` for the policy head (continuous +
    squash + log_std modes + mean init) so the shared variant is a
    drop-in replacement at the agent level; the value head is a
    single Dense(1) sharing the encoder features.

    Returns ``(distribution, value)`` from a single forward pass.
    Callers that need only one output can do
    ``dist, _ = net.apply(params, obs)`` or use the targeted
    :meth:`apply_actor` / :meth:`apply_value` methods.
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int
    continuous: bool = True
    squash: bool = False
    penultimate_normalization: bool = False
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None
    actor_kernel_init: Optional[Union[str, InitializationFunction]] = None
    actor_bias_init: Optional[Union[str, InitializationFunction]] = None
    critic_kernel_init: Optional[Union[str, InitializationFunction]] = None
    critic_bias_init: Optional[Union[str, InitializationFunction]] = None
    mean_kernel_init: Optional[Union[str, InitializationFunction]] = None
    log_std_state_independent: bool = False
    log_std_init: float = -1.0
    disable_encoder_output_norm: bool = False
    cnn_image_shape: Optional[Tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
    cnn_spec: Optional[CNNSpec] = None

    def setup(self):
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
        # Value head
        v_kernel = (
            orthogonal(1.0)
            if self.critic_kernel_init is None
            else parse_initialization(self.critic_kernel_init)
        )
        v_bias = (
            constant(0.0)
            if self.critic_bias_init is None
            else parse_initialization(self.critic_bias_init)
        )
        self.value_head = nn.Dense(
            1,
            kernel_init=v_kernel,
            bias_init=v_bias,
            name="value_head",
        )
        # Actor head: continuous or discrete
        a_bias = (
            constant(0.0)
            if self.actor_bias_init is None
            else parse_initialization(self.actor_bias_init)
        )
        if self.continuous:
            if self.mean_kernel_init is None:
                m_kernel = orthogonal(0.01)
            elif callable(self.mean_kernel_init):
                m_kernel = self.mean_kernel_init
            else:
                m_kernel = parse_initialization(self.mean_kernel_init)
            self.mean = nn.Dense(
                self.action_dim,
                kernel_init=m_kernel,
                bias_init=a_bias,
                name="mean",
            )
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
            a_kernel = (
                orthogonal(1.0)
                if self.actor_kernel_init is None
                else parse_initialization(self.actor_kernel_init)
            )
            self.model = nn.Sequential(
                [
                    nn.Dense(self.action_dim, kernel_init=a_kernel, bias_init=a_bias),
                    distrax.Categorical,
                ]
            )

    def _heads(self, emb):
        value = self.value_head(emb)
        if self.continuous:
            mean = self.mean(emb)
            if self.log_std_state_independent:
                log_std = jnp.clip(
                    jnp.broadcast_to(self.log_std_param, mean.shape),
                    -20,
                    2,
                )
            else:
                log_std = jnp.clip(self.log_std(emb), -20, 2)
            std = jnp.exp(log_std)
            dist = (
                SquashedNormal(mean, std) if self.squash else distrax.Normal(mean, std)
            )
        else:
            dist = self.model(emb)
        return dist, value

    def __call__(self, obs):
        emb = self.encoder(obs)
        return self._heads(emb)

    def apply_actor(self, obs):
        """Run encoder + actor head only (value head's params still
        live in the same pytree but aren't applied here)."""
        dist, _ = self._heads(self.encoder(obs))
        return dist

    def apply_value(self, obs):
        """Run encoder + value head only."""
        _, value = self._heads(self.encoder(obs))
        return value

    def apply_encoder(self, obs):
        """Expose the encoder features alone."""
        return self.encoder(obs)


class MultiHeadCritic(Critic):
    """Critic with a shared encoder and one or more *additional* value heads.

    A drop-in subclass of :class:`Critic` that keeps the original
    ``__call__(x) -> primary head`` for backward compatibility, and
    exposes named extra heads via :meth:`apply_head` and
    :meth:`apply_all_heads`. All heads read from the same encoder
    output, so gradients on any head shape the shared encoder.

    Use case
    --------
    Multi-objective value learning where one head trains by some
    primary signal (e.g. SAC's TD target on Q) and additional heads
    train by complementary signals (e.g. an analytical safety predicate
    on a state-value head, used by SafeSAC's shield). Sharing the
    encoder means the safety geometry is preserved under online TD
    updates because both losses compete for the same parameters,
    rather than the safety prior living in a separate frozen module
    that the task critic is only loosely coupled to via distillation.

    Composability with :class:`MultiCritic`
    ---------------------------------------
    ``MultiCritic`` ``vmap``s a target module across an ensemble axis.
    Pass ``MultiHeadCritic`` as the target and each ensemble member
    will carry its own copy of every head. For SAC's twin-Q ensemble
    one can either (a) read a particular head from a particular
    ensemble member, or (b) aggregate the head across the ensemble
    (e.g. ``min`` over Q for the conservative target, ``mean`` over
    V_safety for the shield).

    Parameters
    ----------
    extra_head_names : Tuple[str, ...]
        Names of additional heads beyond the primary one.
    extra_head_dims  : Tuple[int, ...]
        Output dimensionalities, parallel to ``extra_head_names``.
        Use ``1`` for scalar value heads.

    Examples
    --------
    >>> critic = MultiHeadCritic(
    ...     input_architecture=("256", "relu", "256", "relu"),
    ...     extra_head_names=("v_safety",),
    ...     extra_head_dims=(1,),
    ... )
    >>> q = critic.apply(params, obs)  # primary head
    >>> v_safety = critic.apply(params, obs, head="v_safety", method=critic.apply_head)
    >>> all_heads = critic.apply(params, obs, method=critic.apply_all_heads)
    >>> # all_heads = {"primary": ..., "v_safety": ...}
    """

    extra_head_names: Tuple[str, ...] = ()
    extra_head_dims: Tuple[int, ...] = ()

    def setup(self):
        super().setup()  # builds self.encoder + self.model (primary head)
        if len(self.extra_head_names) != len(self.extra_head_dims):
            raise ValueError(
                "extra_head_names and extra_head_dims must have the same "
                "length, got "
                f"{len(self.extra_head_names)} and {len(self.extra_head_dims)}"
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
        # Flax registers submodule attributes by name. Use setattr with
        # a stable ``head_<name>`` prefix (a dict-of-modules attribute
        # would not be auto-registered).
        for name, dim in zip(self.extra_head_names, self.extra_head_dims):
            setattr(
                self,
                self._extra_attr(name),
                nn.Dense(dim, kernel_init=kernel_init, bias_init=bias_init),
            )

    @staticmethod
    def _extra_attr(name: str) -> str:
        return f"head_{name}"

    def _extra_head(self, name: str):
        return getattr(self, self._extra_attr(name))

    def __call__(self, x: jax.Array) -> jax.Array:
        # Backward-compat: return the primary head's output only. We
        # also evaluate the extra heads on a dummy zero so Flax sees
        # them during init and registers their params; the result is
        # multiplied by 0 and added so the forward output is unchanged.
        feat = self.encoder(x)
        primary = self.model(feat)
        if self.is_initializing() and self.extra_head_names:
            for name in self.extra_head_names:
                _ = self._extra_head(name)(feat)
        return primary

    def apply_head(self, x: jax.Array, head: str) -> jax.Array:
        """Run the encoder + a specific named head.

        ``head="primary"`` (or any name not in ``extra_head_names``)
        returns the primary head's output. Otherwise the named extra
        head's output.
        """
        feat = self.encoder(x)
        if head in self.extra_head_names:
            return self._extra_head(head)(feat)
        return self.model(feat)

    def apply_all_heads(self, x: jax.Array) -> dict:
        """Run the encoder once and return all heads' outputs as a dict
        keyed by head name. The primary head is keyed under
        ``"primary"``."""
        feat = self.encoder(x)
        out = {"primary": self.model(feat)}
        for name in self.extra_head_names:
            out[name] = self._extra_head(name)(feat)
        return out

    def apply_encoder(self, x: jax.Array) -> jax.Array:
        """Expose the shared encoder's features alone, without any head.

        Used by auxiliary representation-shaping losses (VAE, RSSM)
        that need to push gradients into the shared encoder while
        owning their own decoder/prior/posterior heads. Returning the
        normalised feature vector keeps the latent geometry identical
        to what the Q heads and ``v_safety`` head see.
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

    def setup(self):
        Vmapped = nn.vmap(
            target=Critic,
            in_axes=None,
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
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.ensemble(x)

    def apply_encoder_ensemble(self, x: jax.Array) -> jax.Array:
        """Per-ensemble-member encoder features, shape ``(num, ..., d)``.

        Auxiliary modules (VAE, RSSM) consume these features to drive
        gradients back through the shared encoder. Callers typically
        average across the ensemble axis since the encoder sees
        identical inputs and only diverges via its init RNG.
        """
        return self.ensemble.apply_encoder(x)


class MultiHeadMultiCritic(nn.Module):
    """Ensemble of :class:`MultiHeadCritic` (each ensemble member has the
    same set of extra heads).

    Mirrors :class:`MultiCritic` but vmaps over ``MultiHeadCritic``
    instead of ``Critic``. Output of the primary head is shape
    ``(num, ...)`` (same as MultiCritic). The extra heads can be read
    per-ensemble-member via ``apply_head_ensemble`` / ``apply_all_heads_ensemble``.

    Aggregation policy is left to the caller: SAC's twin-Q convention
    is ``min`` over the ensemble for the Bellman target; for a safety
    head the right aggregation is task-dependent (``mean`` for an
    averaged shield value, ``min`` for a conservative one).
    """

    input_architecture: Sequence[Union[str, ActivationFunction]]
    num: int = 4
    extra_head_names: Tuple[str, ...] = ()
    extra_head_dims: Tuple[int, ...] = ()
    penultimate_normalization: bool = False
    kernel_init: Optional[Union[str, InitializationFunction]] = None
    bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None

    def setup(self):
        # Build the vmapped target ONCE with all relevant methods
        # exposed for ensembling. Each ensemble member has its own
        # params (variable_axes) and its own init RNG (split_rngs).
        Vmapped = nn.vmap(
            target=MultiHeadCritic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num,
            methods=("__call__", "apply_head", "apply_all_heads", "apply_encoder"),
        )
        self.ensemble = Vmapped(
            input_architecture=self.input_architecture,
            penultimate_normalization=self.penultimate_normalization,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            encoder_kernel_init=self.encoder_kernel_init,
            encoder_bias_init=self.encoder_bias_init,
            extra_head_names=self.extra_head_names,
            extra_head_dims=self.extra_head_dims,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.ensemble(x)

    def apply_head_ensemble(self, x: jax.Array, head: str) -> jax.Array:
        """Per-ensemble-member output of a specific named head, shape
        ``(num, ...)``."""
        return self.ensemble.apply_head(x, head)

    def apply_all_heads_ensemble(self, x: jax.Array) -> dict:
        """Per-ensemble-member output of every head, returned as a
        ``{head_name -> (num, ...)}`` dict."""
        return self.ensemble.apply_all_heads(x)

    def apply_encoder_ensemble(self, x: jax.Array) -> jax.Array:
        """Per-ensemble-member encoder features, shape ``(num, ..., d)``.

        Auxiliary modules (VAE, RSSM) consume these features to drive
        gradients back through the shared encoder. Callers typically
        aggregate across the ensemble axis (mean) since the encoder
        sees identical inputs and only diverges due to its init RNG.
        """
        return self.ensemble.apply_encoder(x)


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
    cnn_image_shape: Optional[Tuple[int, int, int]] = None,
    extra_critic_head_names: Tuple[str, ...] = (),
    extra_critic_head_dims: Tuple[int, ...] = (),
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
        )
    if extra_critic_head_names:
        # SafeSAC and other multi-objective subclasses want one or more
        # extra value heads sharing the SAC critic's encoder. The
        # ensemble structure (num critics) is preserved; each member
        # carries the same set of heads.
        critic = MultiHeadMultiCritic(
            input_architecture=network_config.critic_architecture,
            penultimate_normalization=network_config.penultimate_normalization,
            num=num_critics,
            extra_head_names=tuple(extra_critic_head_names),
            extra_head_dims=tuple(
                extra_critic_head_dims
                if extra_critic_head_dims
                else (1,) * len(extra_critic_head_names)
            ),
            kernel_init=critic_kernel_init,
            bias_init=critic_bias_init,
            encoder_kernel_init=encoder_kernel_init,
            encoder_bias_init=encoder_bias_init,
        )
    else:
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


def get_initialized_shared_actor_critic(
    key: jax.Array,
    env_config: EnvironmentConfig,
    optimizer_config: OptimizerConfig,
    network_config: NetworkConfig,
    continuous: bool = True,
    squash: bool = False,
    actor_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    actor_bias_init: Optional[Union[str, InitializationFunction]] = None,
    critic_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    critic_bias_init: Optional[Union[str, InitializationFunction]] = None,
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None,
    mean_kernel_init: Optional[Union[str, InitializationFunction]] = None,
    log_std_state_independent: bool = False,
    log_std_init: float = -1.0,
    disable_encoder_output_norm: bool = False,
    max_timesteps: Optional[int] = None,
    extra_obs_dim: int = 0,
    action_dim_override: Optional[int] = None,
    cnn_image_shape: Optional[Tuple[int, int, int]] = None,
) -> LoadedTrainState:
    """Initialise a :class:`SharedActorCritic` (one encoder, two heads)
    with a single optimizer. Returns ONE TrainState (vs the dual
    :func:`get_initialized_actor_critic` which returns two).

    The single TrainState is what enables fused-loss training (brax-
    PPO-style ``policy + vf_coef*value + ent_coef*entropy`` with one
    backward pass + one optimizer step). The caller (PPO with
    ``shared_encoder=True``) sets both ``state.actor_state`` and
    ``state.critic_state`` to point at this same object so the rest
    of the agent infrastructure that expects the dual surface (e.g.
    ``predict_value``, ``get_pi``) keeps working unchanged.
    """
    action_dim = (
        action_dim_override
        if action_dim_override is not None
        else get_action_dim(env_config.env, env_config.env_params)
    )
    if cnn_image_shape is None:
        cnn_image_shape = network_config.cnn_image_shape
    cnn_spec = network_config.cnn_spec
    # The actor_architecture is treated as THE shared encoder arch.
    # (Caller should set actor_architecture == critic_architecture or
    # accept that the shared backbone uses the actor one.)
    net = SharedActorCritic(
        input_architecture=network_config.actor_architecture,
        action_dim=action_dim,
        continuous=continuous,
        squash=squash,
        penultimate_normalization=network_config.penultimate_normalization,
        encoder_kernel_init=encoder_kernel_init,
        encoder_bias_init=encoder_bias_init,
        actor_kernel_init=actor_kernel_init,
        actor_bias_init=actor_bias_init,
        critic_kernel_init=critic_kernel_init,
        critic_bias_init=critic_bias_init,
        mean_kernel_init=mean_kernel_init,
        log_std_state_independent=log_std_state_independent,
        log_std_init=log_std_init,
        disable_encoder_output_norm=disable_encoder_output_norm,
        cnn_image_shape=cnn_image_shape,
        cnn_extra_obs_dim=extra_obs_dim,
        cnn_spec=cnn_spec,
    )
    tx = get_adam_tx(**to_state_dict(optimizer_config))
    observation_shape, _ = get_state_action_shapes(env_config.env)
    obs_extra = (1 if max_timesteps is not None else 0) + extra_obs_dim
    if obs_extra > 0:
        _obs_shape = list(observation_shape)
        _obs_shape[-1] += obs_extra
        observation_shape = tuple(_obs_shape)
    init_obs = jnp.zeros((1, *observation_shape))
    state = init_network_state(
        init_x=init_obs,
        network=net,
        key=key,
        tx=tx,
        recurrent=network_config.lstm_hidden_size is not None,
        lstm_hidden_size=network_config.lstm_hidden_size,
        n_envs=env_config.n_envs,
        lr_schedule=optimizer_config.learning_rate,
    )
    return state


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
