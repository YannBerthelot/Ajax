"""Static world-model hyperparameters of DreamerV3 and the size presets.

DreamerV3 (Hafner et al., arXiv:2301.04104v2) scales one model dimension
``d`` (dreamerv3_spec 2.1; Table 3 p.20): ``units = hidden = d`` (the width
of every MLP and of the RSSM layers), ``deter = 8 d`` recurrent units and
``classes = d / 16`` classes per latent, with 32 latents and 8 GRU blocks at
every size. ``danijar/dreamerv3@2411f7d:dreamerv3/configs.yaml`` ships the
presets ``size12m`` ... ``size400m``; ``size1m`` (``d = 64``) exists only in
the later code (``e3f0224:dreamerv3/configs.yaml:120-153``) and is kept here
for small CPU runs. The paper prints 1024 recurrent units for 12M, a typo
for ``8 d = 2048`` (``docs/world_models/deviations.md`` section 1).

Every other world-model hyperparameter is the same at all sizes
(2411f7d ``configs.yaml``, ``rssm_loss``, ``loss_scales``, ``horizon``) and
is a field of :class:`DreamerV3Config` with that value as its default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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


@dataclass(frozen=True)
class DreamerV3Config:
    """Static DreamerV3 world-model hyperparameters (frozen, hashable).

    The defaults are the ``12m`` preset of the paper-era code
    (``2411f7d:dreamerv3/configs.yaml``; dreamerv3_spec 2.1-2.16). Build
    other sizes with :meth:`from_model_size`. Hashable, so it can be a
    ``jax.jit`` static argument or a field of a flax module.

    Attributes:
        units: width ``d`` of the encoder, decoder, reward and continue MLPs
            (``.*\\.units``).
        hidden: width of the RSSM's input embeddings, hidden layers and
            posterior / prior MLPs (``dyn.rssm.hidden``).
        deter: width of the deterministic state ``h`` (``dyn.rssm.deter``).
        stoch: number of categorical latents ``S`` (``dyn.rssm.stoch``).
        classes: classes per latent ``C`` (``dyn.rssm.classes``).
        blocks: number of GRU blocks ``g`` (``dyn.rssm.blocks``).
        enc_layers, dec_layers, rew_layers, con_layers: hidden layers of the
            vector encoder, vector decoder, reward head and continue head.
        bins: two-hot bins of the reward head (``rewhead.bins``).
        unimix: uniform mixture of the latent categoricals
            (``dyn.rssm.unimix``; Table 4 "Latent unimix 1%").
        free_nats: free bits of the dynamics and representation losses
            (``rssm_loss.free``).
        rec_scale, rew_scale, con_scale, dyn_scale, rep_scale: loss scales
            (``loss_scales``: ``dec_mlp``, ``reward``, ``cont``, ``dyn``,
            ``rep``; Table 4).
        return_horizon: the reference's ``horizon``; the continue target is
            ``(1 - 1 / return_horizon) * (1 - is_terminal)`` (``contdisc``,
            dreamerv3_spec 2.13). Renamed to avoid the clash with the
            imagination horizon and with APG / TD-MPC2 ``horizon``.
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

    def __post_init__(self) -> None:
        widths = ("units", "hidden", "deter", "stoch", "classes", "blocks", "bins")
        for name in widths:
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
        """The ``model_size`` preset, with explicit widths taking precedence.

        ``d = MODEL_SIZES[model_size]`` gives ``units = hidden = d``,
        ``deter = 8 d`` and ``classes = d // 16`` (dreamerv3_spec 2.1); any
        width passed explicitly overrides its preset value, and ``kwargs``
        set the remaining fields.
        """
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

    @property
    def feat_dim(self) -> int:
        """Width of the head input ``concat(deter, stoch_flat)`` (``8d + 2d``)."""
        return self.deter + self.stoch * self.classes

    @property
    def cont_target_scale(self) -> float:
        """``1 - 1 / return_horizon``: the discount folded into the continue
        target (0.997 for the default 333)."""
        return 1.0 - 1.0 / self.return_horizon
