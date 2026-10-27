"""Static hyperparameters of DreamerV3 and the size presets.

DreamerV3 (Hafner et al., arXiv:2301.04104v2) scales one model dimension
``d`` (dreamerv3_spec 2.1; Table 3 p.20): ``units = hidden = d`` (the width
of every MLP and of the RSSM layers), ``deter = 8 d`` recurrent units and
``classes = d / 16`` classes per latent, with 32 latents and 8 GRU blocks at
every size. ``danijar/dreamerv3@2411f7d:dreamerv3/configs.yaml`` ships the
presets ``size12m`` ... ``size400m``; ``size1m`` (``d = 64``) exists only in
the later code (``e3f0224:dreamerv3/configs.yaml:120-153``) and is kept here
for small CPU runs. The paper prints 1024 recurrent units for 12M, a typo
for ``8 d = 2048`` (``docs/world_models/deviations.md`` section 1).

Every other hyperparameter -- of the world model, the actor-critic and the
optimizer -- is the same at all sizes (2411f7d ``configs.yaml``:
``rssm_loss``, ``loss_scales``, ``horizon``, ``actor``, ``critic``,
``imag_length``, ``return_lambda``, ``retnorm``, ``opt``, ...) and is a field
of :class:`DreamerV3Config` with that value as its default. The parity tests
pin these defaults against the reference's own configuration.

The agent declares them, with the hyperparameters of its training loop (the
train ratio, the batch geometry and the replay capacity), as its fields,
and this module holds its state (:class:`DreamerV3State`).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import jax
from flax import struct

from ajax.distributional import TwoHot
from ajax.environments.row_collector import RowCollectorState
from ajax.normalizers import ReturnNormalizer
from ajax.state import BaseAgentState, LoadedTrainState

if TYPE_CHECKING:
    from ajax.agents.DreamerV3.replay import ReplayState
    from ajax.log import MetricsAccumulator

#: A learning rate: a constant, or a schedule of the optimizer's update count.
LearningRate = Union[float, Callable[[jax.Array], jax.Array]]

#: Model dimension ``d`` of each preset (dreamerv3_spec 2.1).
MODEL_SIZES: dict[str, int] = {
    "1m": 64,
    "12m": 256,
    "25m": 384,
    "50m": 512,
    "100m": 768,
    "200m": 1024,
    "400m": 1536,
}


#: The widths a ``model_size`` preset sets (:meth:`DreamerV3Config.from_model_size`).
WIDTHS = ("units", "hidden", "deter", "classes")


@dataclass(frozen=True)
class DreamerV3Config:
    """Static DreamerV3 hyperparameters (frozen, hashable).

    The defaults are the ``12m`` preset of the paper-era code
    (``2411f7d:dreamerv3/configs.yaml``; dreamerv3_spec 2.1-2.16, 3.1-3.20,
    4.1-4.8). Build other sizes with :meth:`from_model_size`. Hashable, so it
    can be a ``jax.jit`` static argument or a field of a flax module.

    World model:
        units: width ``d`` of the encoder, decoder, reward, continue, actor
            and critic MLPs (``.*\\.units``).
        hidden: width of the RSSM's input embeddings, hidden layers and
            posterior / prior MLPs (``dyn.rssm.hidden``).
        deter: width of the deterministic state ``h`` (``dyn.rssm.deter``).
        stoch: number of categorical latents ``S`` (``dyn.rssm.stoch``).
        classes: classes per latent ``C`` (``dyn.rssm.classes``).
        blocks: number of GRU blocks ``g`` (``dyn.rssm.blocks``).
        enc_layers, dec_layers, rew_layers, con_layers: hidden layers of the
            vector encoder, vector decoder, reward head and continue head.
        bins: two-hot bins of the reward head and the critic
            (``rewhead.bins``, ``critic.bins``).
        unimix: uniform mixture of the latent categoricals
            (``dyn.rssm.unimix``; Table 4 "Latent unimix 1%").
        free_nats: free bits of the dynamics and representation losses
            (``rssm_loss.free``).
        rec_scale, rew_scale, con_scale, dyn_scale, rep_scale: loss scales
            (``loss_scales``: ``dec_mlp``, ``reward``, ``cont``, ``dyn``,
            ``rep``; Table 4).
        return_horizon: the reference's ``horizon``; the continue target is
            ``(1 - 1 / return_horizon) * (1 - is_terminal)`` (``contdisc``,
            dreamerv3_spec 2.13) and the replay critic's discount is
            ``1 - 1 / return_horizon`` (dreamerv3_spec 3.19). Renamed to
            avoid the clash with the imagination horizon and with APG /
            TD-MPC2 ``horizon``.

    Actor-critic (``29eb964:dreamerv3/configs.yaml:127-146``):
        actor_layers, critic_layers: hidden layers of the actor and the
            critic (``actor.layers``, ``critic.layers``).
        actor_unimix: uniform mixture of the discrete actor
            (``actor.unimix``, read by 2411f7d's ``onehot`` actor;
            deviations.md section 1).
        minstd, maxstd: range of the continuous actor's standard deviation
            ``(maxstd - minstd) sigmoid(x + 2) + minstd``.
        imag_horizon: imagined steps ``H`` per start (``imag_length``).
        lam: lambda of the imagination return (``return_lambda``).
        repval_lam: lambda of the replay return (``return_lambda_replay``).
        actent: entropy coefficient ``eta`` of the actor loss (``actent``).
        slowreg: weight of the slow-critic regulariser (``slowreg``).
        slow_rate: EMA rate of the slow critic after its first update
            (``slow_critic_fraction``; updated every step,
            ``slow_critic_update: 1``).
        retnorm_rate, retnorm_limit: EMA rate and scale floor of the return
            normaliser (``retnorm``).
        actor_scale, critic_scale, repval_scale: loss scales (``loss_scales``:
            ``actor``, ``critic``, ``replay_critic``; Table 4).

    Optimizer (``opt``; dreamerv3_spec 4.3-4.7; :func:`ajax.agents.DreamerV3.
    optim.laprop`):
        learning_rate: a float, or a schedule of the update count; the
            linear warmup multiplies it.
        agc, agc_pmin: adaptive gradient clipping threshold and parameter
            norm floor (``agc``, ``pmin``).
        beta1, beta2, eps: LaProp momentum and RMS decays and the epsilon
            added after the square root.
        warmup: updates of linear learning-rate warmup from 0 (0 disables).
        weight_decay: decoupled weight decay, multiplied by the (warmed-up)
            learning rate as ``OptimizerConfig.weight_decay`` is; Ajax's
            (0, the default, leaves the reference's chain unchanged).
    """

    units: int = 256
    hidden: int = 256
    deter: int = 2048
    stoch: int = 32
    classes: int = 16
    blocks: int = 8
    enc_layers: int = 3
    dec_layers: int = 3
    rew_layers: int = 1
    con_layers: int = 1
    bins: int = 255
    unimix: float = 0.01
    free_nats: float = 1.0
    rec_scale: float = 1.0
    rew_scale: float = 1.0
    con_scale: float = 1.0
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    return_horizon: float = 333.0
    actor_layers: int = 3
    critic_layers: int = 3
    actor_unimix: float = 0.01
    minstd: float = 0.1
    maxstd: float = 1.0
    imag_horizon: int = 15
    lam: float = 0.95
    repval_lam: float = 0.95
    actent: float = 3e-4
    slowreg: float = 1.0
    slow_rate: float = 0.02
    retnorm_rate: float = 0.01
    retnorm_limit: float = 1.0
    actor_scale: float = 1.0
    critic_scale: float = 1.0
    repval_scale: float = 0.3
    learning_rate: LearningRate = 4e-5
    agc: float = 0.3
    agc_pmin: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-20
    warmup: int = 1000
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        sizes = ("units", "hidden", "deter", "stoch", "classes", "blocks", "bins")
        for name in (*sizes, "actor_layers", "critic_layers", "imag_horizon"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.deter % self.blocks:
            raise ValueError(
                f"deter ({self.deter}) must be divisible by blocks ({self.blocks}):"
                " the block GRU splits the state into equal blocks"
            )
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError(f"unimix must be in [0, 1), got {self.unimix}")
        if self.return_horizon <= 1.0:
            raise ValueError(f"return_horizon must be > 1, got {self.return_horizon}")
        if not 0.0 <= self.actor_unimix < 1.0:
            raise ValueError(f"actor_unimix must be in [0, 1), got {self.actor_unimix}")
        if not 0.0 < self.minstd < self.maxstd:
            raise ValueError(
                f"need 0 < minstd < maxstd, got minstd={self.minstd},"
                f" maxstd={self.maxstd}"
            )
        for name in ("lam", "repval_lam", "slow_rate", "retnorm_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {getattr(self, name)}")
        for name in ("beta1", "beta2"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {getattr(self, name)}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be >= 0, got {self.warmup}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")

    @classmethod
    def from_model_size(
        cls,
        model_size: str = "12m",
        *,
        units: Optional[int] = None,
        hidden: Optional[int] = None,
        deter: Optional[int] = None,
        classes: Optional[int] = None,
        **kwargs: Any,
    ) -> DreamerV3Config:
        """The ``model_size`` preset (module docstring), with explicit widths
        taking precedence; ``kwargs`` set the remaining fields."""
        if model_size not in MODEL_SIZES:
            raise ValueError(
                f"Unknown model_size {model_size!r}; choose from {sorted(MODEL_SIZES)}"
            )
        d = MODEL_SIZES[model_size]
        return cls(
            units=d if units is None else units,
            hidden=d if hidden is None else hidden,
            deter=8 * d if deter is None else deter,
            classes=d // 16 if classes is None else classes,
            **kwargs,
        )

    @classmethod
    def from_arguments(cls, arguments: Mapping[str, Any]) -> DreamerV3Config:
        """:meth:`from_model_size` of the agent constructor's arguments:
        ``model_size`` and every field they name (the constructor names its
        hyperparameters as the fields); the other fields keep their
        defaults."""
        names = inspect.signature(cls).parameters  # the fields
        fields = {k: v for k, v in arguments.items() if k in names}
        return cls.from_model_size(arguments["model_size"], **fields)

    def resolved(self) -> dict[str, int]:
        """The widths the preset resolves, for the loggers' run config."""
        return {f"resolved_{k}": getattr(self, k) for k in WIDTHS}

    @property
    def feat_dim(self) -> int:
        """Width of the head input ``concat(deter, stoch_flat)`` (``8d + 2d``)."""
        return self.deter + self.stoch * self.classes

    @property
    def gamma(self) -> float:
        """The discount ``1 - 1 / return_horizon`` (0.997 for the default 333).

        The continue target folds it in, ``gamma (1 - is_terminal)``
        (``contdisc``, dreamerv3_spec 2.13), and the replay critic discounts
        with it (``29eb964:dreamerv3/agent.py:340``; dreamerv3_spec 3.19); the
        imagination return does not, the continue head's probability already
        carrying it (dreamerv3_spec 3.6).
        """
        return 1.0 - 1.0 / self.return_horizon

    @property
    def loss_scales(self) -> dict[str, float]:
        """The scale of each loss term (``loss_scales``, ``configs.yaml:98``)."""
        return {
            "rec": self.rec_scale,
            "rew": self.rew_scale,
            "con": self.con_scale,
            "dyn": self.dyn_scale,
            "rep": self.rep_scale,
            "actor": self.actor_scale,
            "critic": self.critic_scale,
            "repval": self.repval_scale,
        }

    @property
    def two_hot(self) -> TwoHot:
        """The reward and critic two-hot codec, ``symexp_twohot`` over
        ``bins`` bins (:meth:`~ajax.distributional.TwoHot.dreamerv3`)."""
        return TwoHot.dreamerv3(self.bins)


@struct.dataclass(kw_only=True)
class DreamerV3State(BaseAgentState):
    """The DreamerV3 agent's state (``docs/world_models/DESIGN.md`` section 6.6).

    Besides :class:`~ajax.state.BaseAgentState`'s fields -- ``actor_state``
    (the actor), ``critic_state`` (the critic; ``target_params`` is the slow
    critic, and its ``step`` counts the slow critic's updates, which
    :func:`~ajax.agents.DreamerV3.learner.update_slow_critic` reads),
    ``n_updates`` (training steps so far), ``rng``, ``eval_rng``,
    ``ext_state`` --:

    Attributes:
        collector_state: the row collector's state; its ``policy_carry`` is
            the acting posterior and previous action
            (:class:`~ajax.agents.DreamerV3.world_model.PolicyCarry`).
        world_model_state: the world model (``enc``, ``rssm``, ``dec``,
            ``rew``, ``con``) and its LaProp instance.
        retnorm: the return normaliser.
        replay_state: the replay rings and the online-queue counter
            (:class:`~ajax.agents.DreamerV3.replay.ReplayState`).
        train_metrics: the training metrics since the last log.

    The schedule needs no state of its own: it is a function of the
    absolute tick, the scan input.
    """

    collector_state: RowCollectorState
    world_model_state: LoadedTrainState
    retnorm: ReturnNormalizer
    replay_state: ReplayState
    train_metrics: MetricsAccumulator
