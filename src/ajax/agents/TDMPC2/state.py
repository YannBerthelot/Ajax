"""TD-MPC2 static configuration and learner state.

:class:`TDMPC2Config` holds the static hyperparameters of the world model and
of one training update (tdmpc2_spec §1-§2, paper-era defaults of
``nicklashansen/tdmpc2@5f6fade:tdmpc2/config.yaml``). It is the agent's
``<AGENT>Config`` of the agent anatomy (``CONTRIBUTING.md``): a
:class:`~ajax.state.BaseAgentConfig` whose own fields are all static
(``pytree_node=False``), so it is hashable and jitted functions take it as a
static argument; the agent (M4) adds its planning and collection
hyperparameters to it. The learning rates are not in it: they are schedulable
(``float | Callable``) and live in the optimizers
(:func:`ajax.agents.TDMPC2.core.create_update_state`).

:class:`TDMPC2UpdateState` is the part of the agent state one update reads and
writes. :func:`ajax.agents.TDMPC2.core.update` only accesses these four
attributes and ``replace``, so the agent state of M4 can carry them as fields of
its own (``DESIGN.md`` §4.5) and be passed to ``update`` directly.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import jax
from flax import struct

from ajax.agents.TDMPC2.networks import resolve_model_size
from ajax.distributional import TwoHot
from ajax.normalizers import RunningScale
from ajax.state import BaseAgentConfig, LoadedTrainState


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

    def __post_init__(self) -> None:
        if self.num_q < 2:
            raise ValueError(f"num_q must be >= 2 (random pairs), got {self.num_q}")
        if self.latent_dim % self.simnorm_dim:
            raise ValueError(
                f"latent_dim {self.latent_dim} must be divisible by simnorm_dim "
                f"{self.simnorm_dim}"
            )
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

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
