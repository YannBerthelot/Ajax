from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

import flashbax as fbx
import flax
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState
from gymnax import EnvParams
from jax.tree_util import Partial as partial

from ajax.types import EnvStateType, EnvType, InitializationFunction
from ajax.wrappers import NormalizationInfo


@struct.dataclass
class Transition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    next_obs: jnp.ndarray
    raw_obs: Optional[jnp.ndarray] = None
    log_prob: Optional[jnp.ndarray] = None
    inside_box: Optional[jnp.ndarray] = None
    # Pre-tanh sample for SquashedNormal; lets on-policy agents recompute
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
class EnvironmentConfig:
    env: EnvType
    env_params: EnvParams
    n_envs: int
    continuous: bool


@struct.dataclass
class RollingMeanState:
    buffer: jnp.ndarray  # shape (window_size,n_envs)
    index: jnp.ndarray  # shape (1,n_envs)
    count: jnp.ndarray  # shape (1,n_envs)
    sum: jnp.ndarray  # shape (1,n_envs)


@struct.dataclass
class RollinEpisodicMeanRewardState(RollingMeanState):
    last_return: jnp.ndarray
    cumulative_reward: jnp.ndarray


T = TypeVar("T")


def linear_schedule(train_time_fraction: float) -> float:
    return 1 - train_time_fraction


def exponential_schedule(train_time_fraction: float, p: float = 5) -> float:
    return jnp.exp(-p * train_time_fraction)


def polynomial_schedule(train_time_fraction: float, p: int = 2) -> float:
    return jnp.abs(train_time_fraction - 1) ** p


class LoadedStateMixin:
    train_time_fraction: float = flax.struct.field(pytree_node=False)

    def __getattr__(self, name):
        return getattr(self._state, name)

    @property
    def linear_schedule(self) -> float:
        return linear_schedule(self.train_time_fraction)

    @property
    def exponential_schedule(self) -> float:
        return exponential_schedule(self.train_time_fraction)

    @property
    def polynomial_schedule(self) -> float:
        return polynomial_schedule(self.train_time_fraction)


def make_loaded_state_class(base_cls: Type[T]) -> Type[T]:
    @partial(struct.dataclass, kw_only=True)
    class LoadedEnvState(base_cls, LoadedStateMixin):  # type: ignore[valid-type,misc]
        train_time_fraction: float = flax.struct.field(pytree_node=False)

        @property
        def linear_schedule(self) -> float:
            return linear_schedule(self.train_time_fraction)

        @property
        def exponential_schedule(self) -> float:
            return exponential_schedule(self.train_time_fraction)

        @property
        def polynomial_schedule(self) -> float:
            return polynomial_schedule(self.train_time_fraction)

    return LoadedEnvState


# class LoadedEnvState:
#     state: EnvStateType
#     train_time_fraction: float = flax.struct.field(pytree_node=False)
#     # def __init__(self, state: EnvStateType, train_time_fraction: float):
#     #     self._state = state
#     #     self.train_time_fraction = train_time_fraction

#     def __getattr__(self, name):
#         return getattr(self._state, name)

#     @property
#     def linear_schedule(self) -> float:
#         return 1 - self.train_time_fraction

#     @property
#     def exponential_schedule(self) -> float:
#         return jnp.exp(-self.train_time_fraction)


def load_state(state, train_time_fraction: float):
    LoadedCls = make_loaded_state_class(type(state))
    state_dict = dict(state.__dict__)
    state_dict.pop("train_time_fraction", None)
    return LoadedCls(
        **state_dict,
        train_time_fraction=train_time_fraction,
    )


@struct.dataclass
class CollectorState:
    """The variables necessary to interact with the environment and collect the transitions"""

    rng: jax.Array
    _env_state: EnvStateType
    last_obs: jnp.ndarray
    last_terminated: jnp.ndarray
    last_truncated: jnp.ndarray
    episodic_return_state: RollinEpisodicMeanRewardState
    episodic_mean_return: float = jnp.nan
    num_update: int = 0
    timestep: int = 0
    average_reward: float = 0.0
    buffer_state: Optional[fbx.flat_buffer.TrajectoryBufferState] = None
    rollout: Optional[Transition] = None
    cumulative_reward: Optional[jnp.ndarray] = None
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
    # Live gating telemetry (per-step batch means from the latest
    # collect_experience call). Plumbed into SACAux for tensorboard.
    # NaN until the first collection step. last_expert_frac is universal;
    # the other three only populate for LCB / Thompson gates.
    last_expert_frac: float = jnp.nan
    last_q_advantage: float = jnp.nan
    last_critic_sigma_actor: float = jnp.nan
    last_critic_sigma_expert: float = jnp.nan
    # Empirical max of the LCB gate's expert-arm probability over the
    # last collection batch. Diagnostic for the Coverage Lemma's gap-
    # bound hypothesis: a uniform p_max < 1 over training implies the
    # bounded-gap precondition holds on the visited support.
    last_p_expert_max: float = jnp.nan
    # Per-env step counter within the current episode. Incremented on
    # every env step, reset to 0 on done. Used by jsrl_curriculum to
    # decide whether the expert acts (step_in_episode < H_t) or the
    # learner does. Shape: (n_envs,). Zero-initialized.
    step_in_episode: Optional[jnp.ndarray] = None

    @property
    def train_time_fraction(self) -> Optional[float]:
        if self.max_timesteps is None:
            return None
        return self.timestep / self.max_timesteps

    @property
    def env_state(
        self,
    ) -> LoadedStateMixin:
        if self.train_time_fraction is None:
            return self._env_state
        return load_state(self._env_state, train_time_fraction=self.train_time_fraction)

    @env_state.setter
    def env_state(self, value) -> None:
        self._env_state = value
        # LoadedEnvState(
        #     value, train_time_fraction=self.train_time_fraction
        # )


@partial(struct.dataclass, kw_only=True)
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

    def soft_update(self, tau):
        new_target_params = optax.incremental_update(
            self.params,
            self.target_params,
            tau,
        )
        return self.replace(target_params=new_target_params)

    @classmethod
    def create(cls, *, hidden_state=None, apply_fn: Callable, **kwargs):
        # Ensure apply_fn is passed to the parent TrainState
        instance = super().create(apply_fn=apply_fn, **kwargs)
        # Determine if the state is recurrent
        recurrent = hidden_state is not None
        # Return a new instance with hidden_state and recurrent attributes
        return instance.replace(hidden_state=hidden_state, recurrent=recurrent)

    def apply(self, params, *args, **kwargs):
        """Call the apply_fn with the given parameters and arguments."""
        return self.apply_fn(params, *args, **kwargs)

    def __eq__(self, value):
        return (
            self.params.values == value.params.values
            and self.opt_state == value.opt_state
        )

    def params_equal(self, value):
        """Check if the parameters are equal."""
        return self.params.values == value.params.values


def normalize_observation(obs: jax.Array, norm_info: NormalizationInfo) -> jax.Array:
    """Normalize the observation using the normalization info."""
    if norm_info is None or norm_info.var is None:
        return obs
    return (obs - norm_info.mean) / jnp.sqrt(norm_info.var + 1e-8)


def unnormalize_observation(obs: jax.Array, norm_info: NormalizationInfo) -> jax.Array:
    """Unnormalize the observation using the normalization info."""
    if norm_info is None or norm_info.var is None:
        return obs
    return obs * jnp.sqrt(norm_info.var + 1e-8) + norm_info.mean


def simplex(a, b, dyna_factor):
    # Use a fused multiply-add to minimize rounding errors:
    return jnp.add(a, dyna_factor * (b - a))


def get_double_train_state(second_state_type: str, dyna_factor: float = 0.5):
    assert second_state_type in [
        "avg",
        "SAC",
    ], f"Invalid second_state_type: {second_state_type}. Expected 'avg' or 'SAC'."

    @struct.dataclass
    class DoubleTrainState(LoadedTrainState):
        second_state: Optional[LoadedTrainState] = None
        norm_info: Optional[NormalizationInfo] = None
        hidden_state: Optional[Any] = None

        @classmethod
        def from_LoadedTrainState(
            cls,
            lts: LoadedTrainState,
            second_state: LoadedTrainState,
            norm_info: Optional[NormalizationInfo] = None,
        ):
            return cls(
                step=lts.step,
                apply_fn=lts.apply_fn,
                params=lts.params,
                tx=lts.tx,
                opt_state=lts.opt_state,
                target_params=lts.target_params,
                hidden_state=lts.hidden_state,
                recurrent=lts.recurrent,
                second_state=second_state,
                norm_info=norm_info,
            )

        def apply(self, params, obs, *args, **kwargs):
            """Call the apply_fn with the given parameters and arguments."""
            if (
                second_state_type == "avg"
            ):  # This means first state is SAC, so we need to normalize the raw obs
                if self.norm_info is None:
                    processed_obs = obs
                    print(
                        "Warning: norm_info or env is None, not normalizing"
                        " observations."
                    )
                else:
                    processed_obs = normalize_observation(
                        obs,
                        norm_info=jax.tree.map(
                            lambda x: x[0].reshape(1, -1), self.norm_info.obs
                        ),
                    )
            elif (
                second_state_type == "SAC"
            ):  # This means first state is AVG, so we need to unnormalize the obs
                if self.norm_info is None:
                    processed_obs = obs
                    print(
                        "Warning: norm_info or env is None, not unnormalizing"
                        " observations."
                    )
                else:
                    processed_obs = unnormalize_observation(
                        obs,
                        norm_info=jax.tree.map(
                            lambda x: x[0].reshape(1, -1), self.norm_info.obs
                        ),  # as the dimensions are only repeats, keep only the first one to prevent messy broadcasting
                    )
            else:
                raise ValueError(
                    f"Invalid second_state_type: {second_state_type}. Expected"
                    " 0:'avg' or 1: 'SAC'."
                )
            assert (
                obs.shape == processed_obs.shape
            ), f"{obs.shape} != {processed_obs.shape}"

            raw_output = self.apply_fn(params, obs, *args, **kwargs)
            second_output = self.second_state.apply_fn(
                self.second_state.target_params,
                obs,
                *args,
                **kwargs,
            )
            if isinstance(second_output, jnp.ndarray):
                assert jnp.all(
                    jnp.isfinite(second_output)
                ), "second_output has NaN or Inf!"

            _dyna_factor = dyna_factor(self.step).astype(jnp.float32)

            if not isinstance(raw_output, jnp.ndarray):
                # assume raw_output is a SquashedNormal distribution TODO : Make this for any distrax distributon?
                complete_output = raw_output.mix_distributions(
                    jax.lax.stop_gradient(second_output),
                    dyna_factor=jax.lax.stop_gradient(_dyna_factor),
                )

            else:
                complete_output = simplex(
                    raw_output,
                    jax.last.stop_gradient(second_output),
                    jax.lax_stop_gradient(_dyna_factor),
                )

            return complete_output

    return DoubleTrainState


@struct.dataclass
class BaseAgentState:
    rng: jax.Array
    actor_state: LoadedTrainState
    critic_state: LoadedTrainState
    collector_state: CollectorState
    eval_rng: jax.Array
    n_updates: int = 0
    n_logs: int = 0
    index: Optional[int] = None
    # Per-extension pytree state (one entry per Extension in the agent's
    # ExtensionStack; `()` -> no extensions / all stateless). See
    # `ajax.extensions.base`.
    ext_state: tuple = ()
    # Most recent rollout produced by the on-policy collector
    # (PPO / PQN / APO / AVG), exposed at the top level of agent_state
    # for cross-agent measurement extensions (EVarEst-style probes that
    # need an on-state-visitation batch but should not trigger a fresh
    # rollout per eval — analogous to the replay buffer for off-policy
    # agents).
    #
    # Trade-off:
    #   - ``expose_recent_rollout=False`` (default) → ``last_rollout``
    #     stays ``None``: zero new pytree leaves, identical JIT trace,
    #     zero memory cost.
    #   - ``expose_recent_rollout=True`` → carries the full
    #     ``(T, n_envs, ...)`` :class:`Transition` produced by the
    #     iteration's ``collect_experience`` scan. Adds the rollout's
    #     leaves to the carry pytree (extra memory ≈ one rollout's
    #     worth; extra JIT trace cost ≈ one ``replace`` per iteration).
    #     A static-shape placeholder is pre-allocated at ``make_train``
    #     time so the scan-carry pytree structure stays stable from
    #     iteration zero.
    #
    # Off-policy agents (SAC / DQN / REDQ / TD3 / ASAC / UDRL …)
    # already expose a replay buffer via ``collector_state.buffer_state``
    # and do not touch this field.
    last_rollout: Optional[Any] = None

    def replace(self, *args, **kwargs):  # To make mypy happy
        """Replace fields in the dataclass with new values."""
        return struct.replace(self, *args, **kwargs)


@struct.dataclass
class BaseAgentConfig:
    # Opt-in flag for on-policy agents (PPO, PQN, APO, AVG) to write the
    # most recent ``(T, n_envs, ...)`` rollout transition onto
    # ``BaseAgentState.last_rollout`` after each collection. Off by
    # default so the JIT trace and pytree shape are identical to the
    # legacy path. See :attr:`BaseAgentState.last_rollout` for the full
    # trade-off and rationale. Independent of any specific Extension's
    # hyperparameters — this is an algorithm-level resource-allocation
    # switch (whether to retain the rollout for downstream readers).
    expose_recent_rollout: bool = False


@struct.dataclass
class NetworkConfig:
    actor_architecture: Tuple[str]
    critic_architecture: Tuple[str]
    lstm_hidden_size: Optional[int] = None
    penultimate_normalization: bool = False
    actor_kernel_init: Optional[Union[str, InitializationFunction]] = None
    actor_bias_init: Optional[Union[str, InitializationFunction]] = None
    critic_kernel_init: Optional[Union[str, InitializationFunction]] = None
    critic_bias_init: Optional[Union[str, InitializationFunction]] = None
    encoder_kernel_init: Optional[Union[str, InitializationFunction]] = None
    encoder_bias_init: Optional[Union[str, InitializationFunction]] = None
    # Optional CNN encoder for image observations. When `cnn_image_shape`
    # is set, networks treat obs as a flat `(H*W*C + cnn_extra_obs_dim,)`
    # vector: the image portion is reshaped to NHWC and run through a conv
    # stack (see `CNNEncoder`), any trailing scalar dims are concatenated
    # after. None keeps the legacy MLP encoder.
    cnn_image_shape: Optional[Tuple[int, int, int]] = None
    cnn_extra_obs_dim: int = 0
    # Conv architecture for the CNN encoder (a `networks.CNNSpec`, kept as
    # a loose `tuple` annotation to avoid a networks<->state import cycle).
    # None -> CNNEncoder's default architecture.
    cnn_spec: Optional[tuple] = None
    # Actor head knobs (Actor in networks.py). Default = legacy Ajax.
    # Set per-env to match brax/playground convention for envs that
    # need it (e.g. mujoco_playground manip uses scalar state-indep
    # log_std at init=1.0, lecun_uniform mean head, no encoder-output
    # LayerNorm).
    log_std_state_independent: bool = False
    log_std_init: float = -1.0
    mean_kernel_init: Optional[Union[str, InitializationFunction]] = None
    disable_encoder_output_norm: bool = False
    squash: bool = False


@struct.dataclass
class OptimizerConfig:
    learning_rate: float | Callable[[int], float]
    max_grad_norm: Optional[float] = 0.5
    clipped: bool = True
    beta_1: float = 0.9
    beta_2: float = 0.999
    # Adam's epsilon for numerical stability. Default 1e-5 matches the
    # legacy Ajax behaviour; brax/torch/optax default is 1e-8 and is
    # the standard for PPO (Ajax PPO sets this to 1e-8 for the manip
    # tuned configs). Larger eps makes Adam less aggressive on small
    # gradients (e.g. log_std), slowing exploration adaptation.
    eps: float = 1e-5


@struct.dataclass
class AlphaConfig:
    alpha_init: float
    learning_rate: float


@struct.dataclass
class BufferConfig:
    buffer_size: int
    batch_size: int
    n_envs: int
