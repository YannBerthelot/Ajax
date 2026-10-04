"""TD-MPC2 static configuration, learner state and agent state.

:class:`TDMPC2Config` holds the static hyperparameters of the world model, of
one training update, of the MPPI planner (tdmpc2_spec §1-§3) and of the online
loop's replay (§4), with the paper-era defaults of
``nicklashansen/tdmpc2@5f6fade:tdmpc2/config.yaml``. It is the agent's
``<AGENT>Config`` of the agent anatomy (``CONTRIBUTING.md``): a
:class:`~ajax.state.BaseAgentConfig` whose own fields are all static
(``pytree_node=False``), so it is hashable and jitted functions take it as a
static argument. Not in it: the learning rates, which are schedulable
(``float | Callable``) and live in the optimizers
(:func:`ajax.agents.TDMPC2.core.create_update_state`), and the two values the
reference derives from the episode length (the discount and ``seed_steps``,
tdmpc2_spec 2.20 and 4.3), which the agent resolves once it knows its env
(:class:`ajax.agents.TDMPC2.TDMPC2.TDMPC2`).

:class:`TDMPC2UpdateState` is the part of the agent state one update reads and
writes. :func:`ajax.agents.TDMPC2.core.update` only accesses these four
attributes and ``replace``, so the agent state carries them as fields of its
own (:class:`TDMPC2State`, ``DESIGN.md`` §4.5) and is passed to ``update``
directly.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Optional

import jax
from flax import struct

from ajax.agents.TDMPC2.networks import resolve_model_size
from ajax.distributional import TwoHot
from ajax.environments.row_collector import RowCollectorState
from ajax.normalizers import RunningScale
from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState

if TYPE_CHECKING:  # buffer.py builds core.TDMPC2Batch; core imports this module
    from ajax.agents.TDMPC2.buffer import EpisodeBufferState


def _static(default: Any) -> Any:
    """A config field outside the pytree (a compile-time constant under jit)."""
    return struct.field(pytree_node=False, default=default)


@functools.partial(struct.dataclass, kw_only=True)
class TDMPC2Config(BaseAgentConfig):
    """Static TD-MPC2 hyperparameters (defaults: the 5M single-task model).

    Architecture (tdmpc2_spec 1.20; prefer :meth:`from_model_size`):
        latent_dim, enc_dim, mlp_dim, num_enc_layers, num_q: widths, encoder
            depth and number of Q members.
        simnorm_dim: SimNorm group size V (8).
        num_bins, vmax: two-hot bins ``linspace(-vmax, vmax, num_bins)`` in
            symlog space (101, 10; the reference's ``vmin = -vmax``, T24).
        dropout: dropout rate of each Q member's first layer (0.01).
        log_std_min, log_std_max: policy log-std range (-10, 2).

    Update (tdmpc2_spec §2):
        horizon: latent rollout length H (3).
        rho: temporal weight of the losses (0.5).
        consistency_coef, reward_coef, value_coef: world-model loss
            coefficients (20, 0.1, 0.1).
        entropy_coef: policy entropy coefficient beta (1e-4).
        grad_clip_norm: global-norm clip of each optimizer (20).
        tau: target-Q EMA rate and RunningScale rate (0.01).

    Planning (tdmpc2_spec §3; :mod:`ajax.agents.TDMPC2.planner`):
        iterations: MPPI iterations (6); :meth:`planning_iterations` adds the
            reference's +2 for action dimensions >= 20.
        num_samples: candidate action sequences per iteration, policy-prior
            trajectories included (512).
        num_elites: elites kept by top-k (64).
        num_pi_trajs: policy-prior trajectories among the candidates (24; 0
            is the reference's "planning without policy" ablation).
        min_std, max_std: clamp of the sampling std; ``max_std`` is also its
            value at the start of every decision (0.05, 2).
        temperature: multiplies the elite values in the MPPI score
            ``exp(temperature (V - max V))`` (0.5).

    Online loop (tdmpc2_spec §4; :mod:`ajax.agents.TDMPC2.train_TDMPC2`):
        batch_size: sub-trajectories of ``horizon + 1`` rows per update (256).
        buffer_size: replay capacity in env steps (1,000,000), clamped to the
            run length (:func:`ajax.agents.TDMPC2.buffer.replay_capacity`).
    """

    latent_dim: int = _static(512)
    enc_dim: int = _static(256)
    mlp_dim: int = _static(512)
    num_enc_layers: int = _static(2)
    num_q: int = _static(5)
    simnorm_dim: int = _static(8)
    num_bins: int = _static(101)
    vmax: float = _static(10.0)
    dropout: float = _static(0.01)
    log_std_min: float = _static(-10.0)
    log_std_max: float = _static(2.0)
    horizon: int = _static(3)
    rho: float = _static(0.5)
    consistency_coef: float = _static(20.0)
    reward_coef: float = _static(0.1)
    value_coef: float = _static(0.1)
    entropy_coef: float = _static(1e-4)
    grad_clip_norm: float = _static(20.0)
    tau: float = _static(0.01)
    iterations: int = _static(6)
    num_samples: int = _static(512)
    num_elites: int = _static(64)
    num_pi_trajs: int = _static(24)
    min_std: float = _static(0.05)
    max_std: float = _static(2.0)
    temperature: float = _static(0.5)
    batch_size: int = _static(256)
    buffer_size: int = _static(1_000_000)

    def __post_init__(self) -> None:
        if self.num_q < 2:
            raise ValueError(f"num_q must be >= 2 (random pairs), got {self.num_q}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if not 1 <= self.num_elites <= self.num_samples:
            raise ValueError(
                f"num_elites must be in [1, num_samples={self.num_samples}], got "
                f"{self.num_elites}"
            )
        if not 0 <= self.num_pi_trajs <= self.num_samples:
            raise ValueError(
                f"num_pi_trajs must be in [0, num_samples={self.num_samples}], got "
                f"{self.num_pi_trajs}"
            )
        if not 0 < self.min_std <= self.max_std:
            raise ValueError(
                f"need 0 < min_std <= max_std, got {self.min_std}, {self.max_std}"
            )
        if self.latent_dim % self.simnorm_dim:
            raise ValueError(
                f"latent_dim {self.latent_dim} must be divisible by simnorm_dim "
                f"{self.simnorm_dim}"
            )
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {self.buffer_size}")

    @classmethod
    def from_model_size(
        cls,
        model_size: int = 5,
        *,
        enc_dim: Optional[int] = None,
        mlp_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        num_enc_layers: Optional[int] = None,
        num_q: Optional[int] = None,
        **kwargs: Any,
    ) -> TDMPC2Config:
        """Config of a ``model_size`` preset; explicit widths override it.

        See :func:`ajax.agents.TDMPC2.networks.resolve_model_size`;
        ``kwargs`` are the other fields.
        """
        sizes = resolve_model_size(
            model_size,
            enc_dim=enc_dim,
            mlp_dim=mlp_dim,
            latent_dim=latent_dim,
            num_enc_layers=num_enc_layers,
            num_q=num_q,
        )
        return cls(**sizes, **kwargs)

    def planning_iterations(self, action_dim: int) -> int:
        """MPPI iterations for ``action_dim``: ``iterations + 2`` when it is >= 20.

        The reference's heuristic for large action spaces, applied once to its
        config in ``TDMPC2.__init__`` (``5f6fade:tdmpc2/tdmpc2.py:31``;
        tdmpc2_spec 3.1): 8 iterations for dog (A = 38) and humanoid (A = 21).
        """
        return self.iterations + 2 * int(action_dim >= 20)

    @property
    def two_hot(self) -> TwoHot:
        """The reward / value two-hot codec (``TwoHot.tdmpc2``)."""
        return TwoHot.tdmpc2(num_bins=self.num_bins, limit=self.vmax)


@struct.dataclass
class TDMPC2UpdateState:
    """What one TD-MPC2 update reads and writes.

    Attributes:
        world_model_state: encoder, dynamics, reward head and Q ensemble
            (``params = {"encoder", "dynamics", "reward", "q"}``) with their
            Adam state; ``target_params`` is the target Q ensemble (the ``q``
            subtree only, tdmpc2_spec 1.12).
        actor_state: the policy prior and its Adam state.
        q_scale: the RunningScale of the policy loss (tdmpc2_spec 2.13).
        pi_gradnorm_sq: squared norm of the previous update's post-clip policy
            gradient, which enters the paper-era world-model clip norm
            (deviations.md §2); 0 before the first update.
    """

    world_model_state: LoadedTrainState
    actor_state: LoadedTrainState
    q_scale: RunningScale
    pi_gradnorm_sq: jax.Array


@functools.partial(struct.dataclass, kw_only=True)
class TDMPC2State(BaseAgentState):
    """The TD-MPC2 agent state (``DESIGN.md`` §4.5).

    Keyword-only so the narrowed ``critic_state`` default can sit before the
    new required fields. Carries the four :class:`TDMPC2UpdateState`
    attributes, so :func:`ajax.agents.TDMPC2.core.update` takes it directly.

    Attributes:
        actor_state: the policy prior and its Adam state.
        critic_state: ``None``; the Q ensemble belongs to the world model.
        world_model_state: encoder, dynamics, reward head and Q ensemble with
            their Adam state; ``target_params`` is the target Q ensemble.
        q_scale: the RunningScale of the policy loss.
        pi_gradnorm_sq: squared norm of the previous update's post-clip
            policy gradient (paper-era world-model clip norm).
        collector_state: the row collector's state; ``policy_carry`` is the
            planner's per-env warm-start mean ``prev_mean [n_envs, H, A]``.
            ``collector_state.timestep`` counts env (agent) steps.
        buffer_state: the episode replay ring
            (:class:`~ajax.agents.TDMPC2.buffer.EpisodeBuffer`).
        update_metrics: the last update's logged quantities
            (:func:`ajax.agents.TDMPC2.core.update`), kept across ticks
            without updates (held ticks, seed phase) for logging, as the
            reference keeps its last ``train_metrics``.
        n_terminations: true terminations on scheduled episode ends
            (``is_terminal`` rows); with the collector's
            ``n_offschedule_dones`` it makes the agent refuse terminating
            tasks after ``train`` (deviation T10).
    """

    critic_state: Optional[LoadedTrainState] = None  # type: ignore[assignment]
    world_model_state: LoadedTrainState
    q_scale: RunningScale
    pi_gradnorm_sq: jax.Array
    collector_state: RowCollectorState
    buffer_state: EpisodeBufferState
    update_metrics: dict[str, jax.Array]
    n_terminations: jax.Array


@functools.partial(struct.dataclass, kw_only=True)
class TDMPC2MultiTaskState(BaseAgentState):
    """The offline multi-task TD-MPC2 state (``DESIGN.md`` §7; M8).

    :class:`TDMPC2State` without the env interaction: the dataset is not part
    of the state (it is shared by every seed, an argument of the jitted
    program), and nothing is collected. Carries the four
    :class:`TDMPC2UpdateState` attributes, so
    :func:`ajax.agents.TDMPC2.core.update` takes it directly.

    Attributes:
        actor_state: the policy prior and its Adam state.
        critic_state: ``None``; the Q ensemble belongs to the world model.
        collector_state: ``None``; offline training collects nothing.
        world_model_state: encoder, dynamics, reward head, Q ensemble and the
            task-embedding table (``params[core.TASK_EMB]``) with their Adam
            state; ``target_params`` is the target Q ensemble.
        q_scale: the RunningScale of the policy loss (one for all tasks).
        pi_gradnorm_sq: squared norm of the previous update's post-clip
            policy gradient (paper-era world-model clip norm).
        update_metrics: the last update's logged quantities.
        n_updates: updates done, the run's clock (``n_timesteps`` counts
            updates, as the reference's offline iterations).
    """

    critic_state: Optional[LoadedTrainState] = None  # type: ignore[assignment]
    collector_state: Optional[RowCollectorState] = None  # type: ignore[assignment]
    world_model_state: LoadedTrainState
    q_scale: RunningScale
    pi_gradnorm_sq: jax.Array
    update_metrics: dict[str, jax.Array]
