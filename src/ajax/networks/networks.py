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
from ajax.networks.memory import (
    MemoryCell,
    MemoryConfig,
    init_carry,
    resolve_memory_config,
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
            img = nn.Conv(c, kernel_size=(k, k), strides=(s, s))(img)
            img = nn.relu(img)
        # Flatten the spatial+channel dims while preserving leading batch dims.
        img = img.reshape(*img.shape[:-3], -1)
        feat = nn.Dense(self.feature_dim)(img)
        feat = nn.relu(feat)
        if extra is not None:
            feat = jnp.concatenate([feat, extra], axis=-1)
        return feat


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
    # Optional CNN encoder. When `cnn_image_shape` is provided, the encoder
    # treats obs as `(*batch, H*W*C + cnn_extra_obs_dim)` flat: the image
    # portion is reshaped to NHWC, run through a small conv stack, then
    # any trailing scalar dims (e.g. UDRL command) are concatenated to the
    # embedding before the heads. None keeps the legacy MLP encoder.
    cnn_image_shape: Optional[Tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
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
                image_shape=self.cnn_image_shape,
                extra_obs_dim=self.cnn_extra_obs_dim,
            )
        else:
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

    def _distribution(self, embedding) -> distrax.Distribution:
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

    def __call__(self, obs, raw_obs=None, hidden_state=None, done=None):
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
    # Optional memory block; see Actor.memory. When set, __call__ takes
    # time-major (T, B, features) plus (hidden_state, done) and returns
    # (values, new_hidden_state).
    memory: Optional[MemoryConfig] = None

    def setup(self):
        self.encoder = Encoder(
            input_architecture=self.input_architecture,
            penultimate_normalization=self.penultimate_normalization,
        )
        if self.memory is not None:
            self.memory_cell = MemoryCell(self.memory)
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
        feat = self.encoder(x)
        if self.memory is not None:
            if hidden_state is None or done is None:
                raise ValueError(
                    "Recurrent Critic requires hidden_state and done flags."
                )
            hidden_state, feat = self.memory_cell(hidden_state, feat, done)
            return self.model(feat), hidden_state
        return self.model(feat)


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
        if self.memory is not None:
            raise NotImplementedError("MultiHeadCritic does not support memory yet.")
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

    def __call__(self, x: jax.Array, hidden_state=None, done=None) -> jax.Array:
        # hidden_state/done only exist to match Critic's signature;
        # setup() raises when memory is configured, so they are never
        # meaningfully passed.
        del hidden_state, done
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
    # Optional memory block; each ensemble member owns its carry, stacked
    # on a leading axis: hidden_state leaves are (num, batch, hidden).
    memory: Optional[MemoryConfig] = None

    @nn.compact
    def __call__(self, x, hidden_state=None, done=None):
        # x (and done) are broadcast across the ensemble; the carry is
        # mapped over its leading (num,) axis so each member evolves its
        # own memory.
        in_axes = (None, 0, None) if self.memory is not None else None
        ensemble = nn.vmap(
            target=Critic,
            in_axes=in_axes,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num,
        )
        critic = ensemble(
            input_architecture=self.input_architecture,
            penultimate_normalization=self.penultimate_normalization,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            encoder_kernel_init=self.encoder_kernel_init,
            encoder_bias_init=self.encoder_bias_init,
            memory=self.memory,
        )
        if self.memory is not None:
            # Returns (values, new_hidden_state): values (num, T, B, 1),
            # hidden leaves (num, B, hidden).
            return critic(x, hidden_state, done)
        return critic(x)


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
            methods=("__call__", "apply_head", "apply_all_heads"),
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

    memory = resolve_memory_config(
        network_config.memory, network_config.lstm_hidden_size
    )
    if memory is not None:
        if pid_actor_config is not None:
            raise NotImplementedError("PIDActorNetwork does not support memory yet.")
        if extra_critic_head_names:
            raise NotImplementedError("Multi-head critics do not support memory yet.")
        if network_config.penultimate_normalization:
            # The encoder's l2-normalization is hardcoded to axis=1, which
            # is the batch axis on time-major (T, B, F) inputs.
            raise NotImplementedError(
                "penultimate_normalization is incompatible with memory."
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
            cnn_image_shape=cnn_image_shape,
            cnn_extra_obs_dim=extra_obs_dim,
            memory=memory,
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
            memory=memory,
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
        memory=memory,
        n_envs=env_config.n_envs,
        lr_schedule=actor_optimizer_config.learning_rate,
    )
    critic_state = init_network_state(
        init_x=jnp.hstack([init_obs, init_action]) if action_value else init_obs,
        network=critic,
        key=critic_key,
        tx=critic_tx,
        memory=memory,
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
        memory=resolve_memory_config(
            network_config.memory, network_config.lstm_hidden_size
        ),
        n_envs=env_config.n_envs,
        lr_schedule=critic_optimizer_config.learning_rate,
    )


def init_hidden_state(
    lstm_hidden_size: int,
    n_envs: int,
    rng: jax.random.PRNGKey,
) -> HiddenState:
    """Deprecated: legacy GRU carry initializer, kept for backward compat.

    Note the historical field name: it always built a GRU. Prefer
    ``ajax.networks.memory.init_carry`` with an explicit MemoryConfig.
    """
    return init_carry(
        MemoryConfig(kind="gru", hidden_size=lstm_hidden_size), rng, n_envs
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
    init_x,
    network,
    key,
    tx,
    recurrent: bool = False,
    lstm_hidden_size: Optional[int] = None,
    n_envs: int = 1,
    lr_schedule=None,
    memory: Optional[MemoryConfig] = None,
):
    # Legacy path: recurrent=True + lstm_hidden_size built a GRU. The
    # explicit `memory` argument supersedes both.
    memory = resolve_memory_config(memory, lstm_hidden_size if recurrent else None)
    if memory is None:
        params = FrozenDict(network.init(key, init_x))
        hidden_state = None
    else:
        init_key, carry_key = jax.random.split(key)
        hidden_state = init_network_carry(network, memory, carry_key, n_envs)
        # Recurrent networks consume time-major (T, B, ...) sequences;
        # initialize with a single-step sequence.
        params = FrozenDict(
            network.init(
                init_key,
                init_x[None, ...],
                hidden_state=hidden_state,
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
