from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Tuple,
    Union,
)

import flashbax as fbx
import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from gymnax import EnvParams

from ajax.environments.utils import get_env_type
from ajax.networks.memory import MemoryConfig
from ajax.types import EnvStateType, EnvType, InitializationFunction, Replaceable
from ajax.wrappers import NormalizationInfo

if TYPE_CHECKING:
    from ajax.modules.pid_actor import PIDPolicy
    from ajax.networks.networks import CNNSpec


@struct.dataclass
class Transition(Replaceable):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    next_obs: jnp.ndarray
    raw_obs: Optional[jnp.ndarray] = None
    log_prob: Optional[jnp.ndarray] = None
    # Pre-tanh sample of a squashed policy; lets on-policy agents recompute
    # log_prob without arctanh (which is unstable as |action| -> 1).
    raw_action: Optional[jnp.ndarray] = None
    # Expert action computed at collection time, with the correct (stateful)
    # expert internal state. None for methods that don't need it. Read by
    # the residual-RL actor loss so the actor's Q-gradient evaluates at
    # the same a_expert the critic was trained on, instead of recomputing
    # the expert with a fresh zero state on a buffer-sampled obs.
    a_expert: Optional[jnp.ndarray] = None
    # Expert action at the *next* observation s_{t+1}. Needed by the
    # residual-RL TD target so the bootstrap Q is evaluated on the same
    # residual-transformed action distribution the critic was trained
    # on (clip(a_expert + scale * a_pi, -1, 1)). Without this the target
    # is OOD for the critic and training diverges.
    next_a_expert: Optional[jnp.ndarray] = None
    # The environment's own observation at s_{t+1} (the final one where an
    # episode ended), as raw_obs is at s_t: what an expert reads there.
    raw_next_obs: Optional[jnp.ndarray] = None
    # A dataset row's discounted return to its episode's end (-inf where
    # unknown: a collected row), for an extension that reads it (Cal-QL).
    return_to_go: Optional[jnp.ndarray] = None
    # A dataset row's next action, its episode's next row's (NaN where
    # unknown: an episode's end, a collected row), for ReBRAC's TargetBC.
    next_action: Optional[jnp.ndarray] = None

    def __len__(self):
        return self.obs.shape[0] if self.obs.ndim > 0 else 1


def zeros_like_abstract_pytree(abstract_tree: Any) -> Any:
    """Allocate a concrete zero-filled pytree mirroring the per-leaf
    shape/dtype of an abstract pytree (as returned by ``jax.eval_shape``).

    Used by the on-policy agents (PPO / PQN / APO / AVG) to pre-allocate
    the ``last_rollout`` placeholder on :class:`BaseAgentState` so the
    JIT-traced scan body sees a stable pytree carry from iteration zero
    — see :attr:`BaseAgentState.last_rollout`.
    """
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype),
        abstract_tree,
    )


@struct.dataclass
class EnvironmentConfig(Replaceable):
    env: EnvType
    env_params: EnvParams
    n_envs: int
    continuous: bool
    # Simulator steps per agent step (brax / playground EpisodeWrapper, see
    # ``ajax.environments.create``). Static: it fixes shapes and schedules
    # (episode length in agent steps), never a traced value.
    action_repeat: int = struct.field(pytree_node=False, default=1)
    # The agent normalises the observations itself (a replay agent's
    # ``normalize_observations``): the env passes them raw, the collector
    # keeps running statistics (CollectorState.obs_norm_info) and the
    # networks apply the current ones (ajax.environments.obs_norm).
    normalize_obs_in_agent: bool = struct.field(pytree_node=False, default=False)

    @property
    def mode(self) -> str:
        """The backend: "gymnax", or "brax" for brax and playground envs."""
        return get_env_type(self.env)


@struct.dataclass
class RollinEpisodicMeanRewardState(Replaceable):
    """Each env's rolling window of its last episode returns."""

    buffer: jnp.ndarray  # (window_size, n_envs, 1)
    index: jnp.ndarray  # (n_envs, 1): the next slot to write
    count: jnp.ndarray  # (n_envs, 1): filled slots
    sum: jnp.ndarray  # (n_envs, 1)
    last_return: jnp.ndarray
    cumulative_reward: jnp.ndarray


@struct.dataclass
class CollectorState(Replaceable):
    """The variables necessary to interact with the environment and collect the transitions"""

    rng: jax.Array
    _env_state: EnvStateType
    last_obs: jnp.ndarray
    last_terminated: jnp.ndarray
    last_truncated: jnp.ndarray
    episodic_return_state: RollinEpisodicMeanRewardState
    episodic_mean_return: Union[float, jax.Array] = jnp.nan
    timestep: Union[int, jax.Array] = 0
    buffer_state: Optional[fbx.flat_buffer.TrajectoryBufferState] = None
    rollout: Optional[Transition] = None
    max_timesteps: Optional[int] = None
    last_in_box: Optional[jnp.ndarray] = None
    # Batched internal state of a stateful expert policy (e.g. PID integrator).
    # None when no expert is used. Threaded through collection so integral/
    # derivative terms aren't reset every env step.
    expert_state: Optional[Any] = None
    # Agent-side running normalization stats for the FULL augmented obs
    # (env_obs + flatten(expert_state)). Updated only at collection time
    # (online and BC); read-only at apply_fn / eval. Lives in collector
    # so it's part of agent_state and threads through naturally. None
    # disables normalization.
    obs_norm_info: Optional[NormalizationInfo] = None
    # Fraction of envs an expert drove on the latest collection step
    # (EDGEExploration logs it); NaN until the first step.
    last_expert_frac: Union[float, jax.Array] = jnp.nan
    # Per-env step counter within the current episode. Incremented on
    # every env step, reset to 0 on done; None unless an extension reads it
    # (Extension.reads_step_in_episode: JSRLCurriculum, whose expert acts
    # while step_in_episode < H_t). Shape: (n_envs,). Zero-initialized.
    step_in_episode: Optional[jnp.ndarray] = None

    @property
    def train_time_fraction(self) -> Optional[Union[float, jax.Array]]:
        if self.max_timesteps is None:
            return None
        return self.timestep / self.max_timesteps

    @property
    def env_state(self) -> EnvStateType:
        """The environment's own state, what it is stepped and reset with;
        the training fraction is the collector's (``train_time_fraction``)."""
        return self._env_state


@struct.dataclass(kw_only=True)
class LoadedTrainState(TrainState):
    hidden_state: Optional[Any] = None
    recurrent: bool = False
    target_params: Optional[flax.core.FrozenDict] = None
    # Read-only mirror of CollectorState.obs_norm_info, synced after every
    # online collection step. ``get_pi`` / ``predict_value`` read this and
    # normalise obs before the forward pass, so downstream callers stay
    # unchanged. None disables normalisation. Critic networks slice the
    # obs portion of the (obs, action) input before normalising — the
    # action portion stays raw.
    obs_norm_info: Optional[NormalizationInfo] = None
    # The affine map ``offset + scale * a`` a critic applies to the agent's
    # [-1, 1] actions before reading them (recurrent.q_values), as static
    # ``(offsets, scales)`` tuples. An original that acts in the env's units
    # (sfujim/TD3's max_action * tanh) trains its critic on those, and Adam's
    # steps on the action inputs depend on the scale. None: [-1, 1] as is.
    action_units: Optional[tuple] = struct.field(pytree_node=False, default=None)

    def soft_update(self, tau):
        new_target_params = optax.incremental_update(
            self.params,
            self.target_params,
            tau,
        )
        return self.replace(target_params=new_target_params)

    @classmethod
    def create(cls, *, hidden_state=None, apply_fn: Callable, **kwargs):
        instance = super().create(apply_fn=apply_fn, **kwargs)
        recurrent = hidden_state is not None
        return instance.replace(hidden_state=hidden_state, recurrent=recurrent)

    def apply(self, params, *args, **kwargs):
        return self.apply_fn(params, *args, **kwargs)


@struct.dataclass
class BaseAgentState(Replaceable):
    rng: jax.Array
    actor_state: LoadedTrainState
    critic_state: LoadedTrainState
    collector_state: CollectorState
    # The evaluations' key, never advanced: each evaluation folds in its
    # count, ``n_logs`` (``ajax.log.evaluation_keys``), so each draws fresh
    # initial states. Inits split it from the run's key apart from ``rng``,
    # so no evaluation draw repeats a training one, and last, so the training
    # keys are those of the split without it (threefry is partitionable:
    # ``split(k, n + 1)[:n] == split(k, n)``).
    eval_rng: jax.Array
    n_updates: Union[int, jax.Array] = 0
    n_logs: Union[int, jax.Array] = 0
    index: Optional[int] = None
    # Per-extension pytree state (one entry per Extension in the agent's
    # ExtensionStack; `()` -> no extensions / all stateless). See
    # `ajax.extensions.base`.
    ext_state: tuple = ()
    # The on-policy collector's (PPO / PQN / APO / AVG) latest rollout,
    # for measurement extensions (EVarEst-style probes) that need an
    # on-state-visitation batch without a fresh rollout per evaluation;
    # off-policy agents expose their replay buffer instead
    # (``collector_state.buffer_state``). ``None`` (no extra leaves, no
    # memory) unless ``BaseAgentConfig.expose_recent_rollout``; then the
    # iteration's full ``(T, n_envs, ...)`` :class:`Transition` (one
    # rollout of memory), pre-allocated at ``make_train`` time so the scan
    # carry's structure is stable from iteration zero.
    last_rollout: Optional[Any] = None


@struct.dataclass
class BaseAgentConfig(Replaceable):
    # Opt-in for on-policy agents (PPO, PQN, APO, AVG): keep each
    # collection's rollout on ``BaseAgentState.last_rollout`` (see there
    # for the cost). An algorithm-level resource switch, not any
    # Extension's hyperparameter.
    expose_recent_rollout: bool = False


#: SAC's ``(low, high)`` clip of a Gaussian actor's log-std: the default of
#: ``NetworkConfig.log_std_bounds``, which clips every log-std head (the
#: Actor's Dense head or parameter, the PID actor's); PPO's is None.
LOG_STD_BOUNDS: Tuple[float, float] = (-20.0, 2.0)


def check_log_std_bounds(bounds: Optional[Tuple[float, float]]) -> None:
    """Swapped or equal bounds make ``jnp.clip`` return the high one for
    every input: the std would be silently constant."""
    if bounds is not None and not bounds[0] < bounds[1]:
        raise ValueError(
            f"log_std_bounds is (low, high) with low < high, got {bounds}."
        )


@struct.dataclass
class NetworkConfig(Replaceable):
    # Hidden layers, or a PIDPolicy: the actor is then a PID controller
    # (ajax.modules.pid_actor).
    actor_architecture: Union[Tuple[str, ...], "PIDPolicy"]
    critic_architecture: Tuple[str, ...]
    # Pluggable memory block (encoder -> memory -> heads); None keeps the
    # network feedforward. Static (hashable) so trace-time branching on it
    # never leaks into the compiled graph. See ajax.networks.memory.
    memory: Optional["MemoryConfig"] = flax.struct.field(
        pytree_node=False, default=None
    )
    penultimate_normalization: bool = False
    actor_bias_init: Optional[Union[str, InitializationFunction]] = None
    critic_bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # Optional CNN encoder for image observations. When `cnn_image_shape`
    # is set, networks treat obs as a flat `(H*W*C + cnn_extra_obs_dim,)`
    # vector: the image portion is reshaped to NHWC and run through a conv
    # stack (see `CNNEncoder`), any trailing scalar dims are concatenated
    # after. None keeps the MLP encoder.
    cnn_image_shape: Optional[Tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
    # Conv architecture for the CNN encoder (imported for type checking
    # only, to avoid a networks<->state import cycle). None ->
    # CNNEncoder's default architecture.
    cnn_spec: Optional["CNNSpec"] = None
    # The actor head's knobs; Actor (networks.py) documents each and its
    # reference.
    log_std_state_independent: bool = False
    log_std_init: Optional[float] = -1.0
    # Static, not a pytree leaf: jax rebuilds the config (tree.map,
    # unflatten) through __post_init__, whose check must then read the
    # bounds, not a leaf jax put in their place.
    log_std_bounds: Optional[Tuple[float, float]] = flax.struct.field(
        pytree_node=False, default=LOG_STD_BOUNDS
    )
    mean_kernel_init: Optional[Union[str, InitializationFunction]] = None
    logits_kernel_init: Optional[Union[str, InitializationFunction]] = None
    log_std_kernel_init: Optional[Union[str, InitializationFunction]] = None
    # The critic's value (V or Q) layer; Critic (networks.py) documents it.
    value_kernel_init: Optional[Union[str, InitializationFunction]] = None
    tanh_mean: bool = False
    # Opt-in LayerNorm on the shared encoder's output (Encoder in
    # networks.py), for every network built from this config. Off by
    # default: the SAC, TD3, PPO and DQN papers use plain MLPs. Agents
    # without a constructor argument for it take it through
    # ``agent.network_args.replace(encoder_layer_norm=True)``.
    encoder_layer_norm: bool = False
    squash: bool = False
    # The tanh log-det's guard, Squash's ``log_det_eps``: 0 keeps the exact
    # log-det; an agent whose reference adds one inside the log sets it.
    squash_log_det_eps: float = flax.struct.field(pytree_node=False, default=0.0)
    # Read the MLP actor's log-std head as a log-variance: std = exp(out / 2),
    # ``log_std_bounds`` bounding the log-variance, as an agent whose
    # reference does (AVG's) sets it.
    log_variance_head: bool = False

    def __post_init__(self):
        check_log_std_bounds(self.log_std_bounds)


@struct.dataclass
class OptimizerConfig(Replaceable):
    learning_rate: float | Callable[[int], float]
    max_grad_norm: Optional[float] = None
    beta_1: float = 0.9
    beta_2: float = 0.999
    # Adam's epsilon for numerical stability. Default 1e-5, Ajax's own;
    # brax/torch/optax default is 1e-8 and is the standard for PPO (Ajax
    # PPO's ``adam_eps`` defaults to it). Larger eps makes Adam less
    # aggressive on small gradients (e.g. log_std), slowing exploration
    # adaptation.
    eps: float = 1e-5
    # Decoupled weight decay (AdamW, Loshchilov & Hutter 2019), as optax.adamw
    # and torch's AdamW apply it: ``weight_decay * param`` joins the Adam
    # direction before the learning rate scales it, so the decay is
    # multiplied by the learning rate (a schedule's current value; the
    # paper's is not). With a zero gradient one step multiplies every
    # parameter by ``1 - lr * weight_decay``: a time constant of
    # ``1 / (lr * weight_decay)`` steps. 0 keeps plain Adam.
    weight_decay: float = 0.0
    # The update rule. "adam" (Kingma & Ba 2015); "radam" (Liu et al. 2020,
    # optax.radam: Adam with its variance rectified while few gradients are
    # averaged); "centered_rmsprop" (Graves 2013, optax.rmsprop with
    # centered=True): the gradient over sqrt(E[g^2] - E[g]^2 + eps), both
    # averages decaying at beta_2, no bias correction, beta_1 unused.
    optimizer: str = "adam"


@struct.dataclass
class AlphaConfig:
    alpha_init: float
    learning_rate: float
