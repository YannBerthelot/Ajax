"""TD-MPC2 world model and policy prior (tdmpc2_spec §1).

The five learned components of TD-MPC2 (Hansen et al., arXiv:2310.16828v2,
Sec. 3.1, App. H) at the paper-era commit ``nicklashansen/tdmpc2@5f6fade``
(``tdmpc2/common/world_model.py:16-31``, ``tdmpc2/common/layers.py``), built
from the shared TD-MPC2 trunk :meth:`ajax.networks.blocks.NormedMLP.tdmpc2`
(``Linear -> [Dropout] -> LayerNorm(1e-5) -> Mish``, kernels ``N(0, 0.02^2)``)
and :func:`ajax.networks.blocks.linear`:

=============  ==============================  ================================
component      input (reference order)         layers
=============  ==============================  ================================
encoder h      ``[obs, e]``                    ``max(num_enc_layers - 1, 1)``
                                               hidden ``enc_dim``, then
                                               ``Linear -> LN -> SimNorm`` to
                                               ``latent_dim`` (spec 1.4)
dynamics d     ``[z, e, a]``                   2 hidden ``mlp_dim``, then
                                               ``Linear -> LN -> SimNorm`` to
                                               ``latent_dim`` (spec 1.5)
reward R       ``[z, e, a]``                   2 hidden, plain ``Linear`` to
                                               ``num_bins``, zero weight
                                               (spec 1.6)
Q ensemble     ``[z, e, a]``                   ``num_q`` independent members:
                                               2 hidden with dropout after the
                                               first Linear, plain ``Linear``
                                               to ``num_bins``, zero weight
                                               (spec 1.10)
policy prior   ``[z, e]``                      2 hidden, plain ``Linear`` to
                                               ``2 A`` (mean, raw log-std),
                                               ``N(0, 0.02^2)``, not zeroed
                                               (spec 1.8)
=============  ==============================  ================================

``e`` is an optional task embedding (multi-task, M8); single-task passes
``None`` and the inputs are exactly ``obs``, ``[z, a]`` and ``z``. There is no
embedding table here. Biases are zero and LayerNorm scales one at
initialisation (``common/init.py:4-17``); target Q is a copy of the online Q
taken by the caller after initialisation (``world_model.py:31``).

The encoder, dynamics, reward head and Q ensemble form one flax module,
:class:`WorldModel`, whose parameter tree ``{"encoder", "dynamics", "reward",
"q"}`` is what the world-model optimizer updates; the policy prior,
:class:`PolicyPrior`, is a separate module with its own optimizer
(``tdmpc2.py:21-28``). The Q members are stacked on a leading ``num_q`` axis by
``nn.vmap`` with split ``params`` and ``dropout`` RNG streams, so each member
initialises independently (``init.py:12-16`` draws the stacked tensors at
once) and gets its own dropout mask (``layers.py:16``,
``torch.vmap(randomness='different')``).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ajax.networks.blocks import NormedMLP, linear

if TYPE_CHECKING:
    from ajax.agents.TDMPC2.state import TDMPC2Config

# TD-MPC2's kernel initializer (``init.py:7``; see NormedMLP.tdmpc2 for why
# torch's absolute-bound trunc_normal_(std=0.02) is an untruncated normal).
KERNEL_INIT = "normal(0.02)"

# ``common/__init__.py:1-24`` (tdmpc2_spec 1.20, paper Table 9). ``num_q`` is
# the config default 5 where the reference table omits it.
MODEL_SIZE: dict[int, dict[str, int]] = {
    1: {
        "enc_dim": 256,
        "mlp_dim": 384,
        "latent_dim": 128,
        "num_enc_layers": 2,
        "num_q": 2,
    },
    5: {
        "enc_dim": 256,
        "mlp_dim": 512,
        "latent_dim": 512,
        "num_enc_layers": 2,
        "num_q": 5,
    },
    19: {
        "enc_dim": 1024,
        "mlp_dim": 1024,
        "latent_dim": 768,
        "num_enc_layers": 3,
        "num_q": 5,
    },
    48: {
        "enc_dim": 1792,
        "mlp_dim": 1792,
        "latent_dim": 768,
        "num_enc_layers": 4,
        "num_q": 5,
    },
    317: {
        "enc_dim": 4096,
        "mlp_dim": 4096,
        "latent_dim": 1376,
        "num_enc_layers": 5,
        "num_q": 8,
    },
}


def simnorm(x: jax.Array, simnorm_dim: int = 8) -> jax.Array:
    """Simplicial normalisation: softmax over consecutive groups of ``simnorm_dim``.

    ``layers.py:65-82`` (tdmpc2_spec 1.3): ``x [..., D]`` is viewed as
    ``[..., D / V, V]``, softmaxed over the last axis and viewed back. No
    temperature (paper Eq. 5 and App. H contradict each other; the code has
    none).
    """
    shape = x.shape
    if shape[-1] % simnorm_dim:
        raise ValueError(
            f"SimNorm needs a width divisible by {simnorm_dim}, got {shape[-1]}"
        )
    groups = x.reshape(*shape[:-1], shape[-1] // simnorm_dim, simnorm_dim)
    return jax.nn.softmax(groups, axis=-1).reshape(shape)


def concat_inputs(
    x: jax.Array,
    task_emb: Optional[jax.Array] = None,
    action: Optional[jax.Array] = None,
) -> jax.Array:
    """``[x, e, a]`` with the absent parts left out (``world_model.py:78-120``).

    The task embedding ``e [..., task_dim]`` is broadcast over the leading
    axes of ``x``, as ``task_emb`` repeats it over the horizon and batch
    (``world_model.py:87-90``).
    """
    parts = [x]
    if task_emb is not None:
        parts.append(jnp.broadcast_to(task_emb, (*x.shape[:-1], task_emb.shape[-1])))
    if action is not None:
        parts.append(action)
    return jnp.concatenate(parts, axis=-1)


def simnorm_head(latent_dim: int, simnorm_dim: int) -> NormedMLP:
    """``NormedLinear(act=SimNorm)``: ``Linear -> LayerNorm(1e-5) -> SimNorm``.

    The last layer of the encoder and of the dynamics (``layers.py:85-100,
    121``; tdmpc2_spec 1.3-1.5): the TD-MPC2 trunk layer of
    :meth:`ajax.networks.blocks.NormedMLP.tdmpc2` with SimNorm in place of
    Mish. Construct it inside an ``nn.compact`` method; it is named ``head``.
    """
    return NormedMLP(
        layers=1,
        units=latent_dim,
        act=functools.partial(simnorm, simnorm_dim=simnorm_dim),
        norm="layer",
        norm_eps=1e-5,
        kernel_init=KERNEL_INIT,
        name="head",
    )


class Encoder(nn.Module):
    """State encoder ``h([obs, e])`` to the latent (``layers.py:142-153``)."""

    enc_dim: int
    num_enc_layers: int
    latent_dim: int
    simnorm_dim: int

    @nn.compact
    def __call__(
        self, obs: jax.Array, task_emb: Optional[jax.Array] = None
    ) -> jax.Array:
        x = concat_inputs(obs, task_emb)
        hidden = max(self.num_enc_layers - 1, 1)
        x = NormedMLP.tdmpc2(hidden, self.enc_dim, name="trunk")(x)
        return simnorm_head(self.latent_dim, self.simnorm_dim)(x)


class Dynamics(nn.Module):
    """Latent dynamics ``d([z, e, a])`` (``world_model.py:25, 104-111``)."""

    mlp_dim: int
    latent_dim: int
    simnorm_dim: int

    @nn.compact
    def __call__(
        self, z: jax.Array, action: jax.Array, task_emb: Optional[jax.Array] = None
    ) -> jax.Array:
        x = concat_inputs(z, task_emb, action)
        x = NormedMLP.tdmpc2(2, self.mlp_dim, name="trunk")(x)
        return simnorm_head(self.latent_dim, self.simnorm_dim)(x)


class RewardHead(nn.Module):
    """Reward logits ``R([z, e, a])``; zero final weight (``world_model.py:26, 30``)."""

    mlp_dim: int
    num_bins: int

    @nn.compact
    def __call__(
        self, z: jax.Array, action: jax.Array, task_emb: Optional[jax.Array] = None
    ) -> jax.Array:
        x = concat_inputs(z, task_emb, action)
        x = NormedMLP.tdmpc2(2, self.mlp_dim, name="trunk")(x)
        return linear(self.num_bins, "zeros", name="out")(x)


class QFunction(nn.Module):
    """One Q member: 2 hidden layers (dropout in the first), zero final weight.

    ``world_model.py:28, 30``, ``layers.py:110-122`` (tdmpc2_spec 1.10).
    """

    mlp_dim: int
    num_bins: int
    dropout: float

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: bool) -> jax.Array:
        trunk = NormedMLP.tdmpc2(2, self.mlp_dim, dropout=self.dropout, name="trunk")
        x = trunk(x, deterministic=deterministic)
        return linear(self.num_bins, "zeros", name="out")(x)


class QEnsemble(nn.Module):
    """``num_q`` Q members stacked on a leading axis; logits ``[num_q, ..., num_bins]``.

    ``nn.vmap`` over members with the input broadcast, split ``params`` (each
    member initialised from its own key) and split ``dropout`` (an independent
    mask per member and call), as the reference's
    ``torch.vmap(..., randomness='different')`` (``layers.py:7-24``).
    ``deterministic=False`` applies dropout and needs a ``'dropout'`` RNG.
    At 5f6fade the dropout is active in every pass, eval mode included (see
    :mod:`ajax.agents.TDMPC2.core`, which passes ``deterministic=False`` with
    a dropout key in the TD target and both losses).
    Captured intermediates (``capture_intermediates``) are stacked per member
    like the parameters.
    """

    num_q: int
    mlp_dim: int
    num_bins: int
    dropout: float

    @nn.compact
    def __call__(
        self,
        z: jax.Array,
        action: jax.Array,
        task_emb: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        x = concat_inputs(z, task_emb, action)
        members = nn.vmap(
            QFunction,
            variable_axes={"params": 0, "intermediates": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(None, None),
            out_axes=0,
            axis_size=self.num_q,
        )
        return members(self.mlp_dim, self.num_bins, self.dropout, name="members")(
            x, deterministic
        )


class WorldModel(nn.Module):
    """Encoder, dynamics, reward head and Q ensemble (the world-model optimizer's
    parameters, ``tdmpc2.py:21-27``). Apply one component with
    ``method="encode" | "next" | "reward_logits" | "q_logits"``; ``__call__``
    exists to initialise all four.
    """

    latent_dim: int
    enc_dim: int
    mlp_dim: int
    num_enc_layers: int
    num_q: int
    num_bins: int
    dropout: float
    simnorm_dim: int

    def setup(self) -> None:
        self.encoder = Encoder(
            self.enc_dim, self.num_enc_layers, self.latent_dim, self.simnorm_dim
        )
        self.dynamics = Dynamics(self.mlp_dim, self.latent_dim, self.simnorm_dim)
        self.reward = RewardHead(self.mlp_dim, self.num_bins)
        self.q = QEnsemble(self.num_q, self.mlp_dim, self.num_bins, self.dropout)

    def encode(self, obs: jax.Array, task_emb: Optional[jax.Array] = None) -> jax.Array:
        """``z = h(obs, e)`` on the latent simplex product (``world_model.py:93-102``)."""
        return self.encoder(obs, task_emb)

    def next(
        self, z: jax.Array, action: jax.Array, task_emb: Optional[jax.Array] = None
    ) -> jax.Array:
        """``z' = d(z, a, e)`` (``world_model.py:104-111``)."""
        return self.dynamics(z, action, task_emb)

    def reward_logits(
        self, z: jax.Array, action: jax.Array, task_emb: Optional[jax.Array] = None
    ) -> jax.Array:
        """Two-hot reward logits ``[..., num_bins]`` (``world_model.py:113-120``)."""
        return self.reward(z, action, task_emb)

    def q_logits(
        self,
        z: jax.Array,
        action: jax.Array,
        task_emb: Optional[jax.Array] = None,
        deterministic: bool = True,
    ) -> jax.Array:
        """All members' logits ``[num_q, ..., num_bins]`` (``world_model.py:161-168``)."""
        return self.q(z, action, task_emb, deterministic)

    def __call__(
        self, obs: jax.Array, action: jax.Array, task_emb: Optional[jax.Array] = None
    ) -> jax.Array:
        z = self.encode(obs, task_emb)
        self.reward_logits(z, action, task_emb)
        self.q_logits(z, action, task_emb)
        return self.next(z, action, task_emb)


class PolicyPrior(nn.Module):
    """Policy prior network ``[z, e] -> (mean, raw log-std)``, each ``[..., A]``.

    ``world_model.py:27, 128-132``: the final Linear (``N(0, 0.02^2)``, not
    zeroed) outputs ``2 A`` values, the first ``A`` the mean. The log-std
    squashing, sampling and log-probability are in
    :func:`ajax.agents.TDMPC2.core.policy_sample`.
    """

    mlp_dim: int
    action_dim: int

    @nn.compact
    def __call__(
        self, z: jax.Array, task_emb: Optional[jax.Array] = None
    ) -> tuple[jax.Array, jax.Array]:
        x = concat_inputs(z, task_emb)
        x = NormedMLP.tdmpc2(2, self.mlp_dim, name="trunk")(x)
        out = linear(2 * self.action_dim, KERNEL_INIT, name="out")(x)
        mean, raw_log_std = jnp.split(out, 2, axis=-1)
        return mean, raw_log_std


def make_world_model(config: TDMPC2Config) -> WorldModel:
    """The :class:`WorldModel` of ``config``'s sizes."""
    return WorldModel(
        latent_dim=config.latent_dim,
        enc_dim=config.enc_dim,
        mlp_dim=config.mlp_dim,
        num_enc_layers=config.num_enc_layers,
        num_q=config.num_q,
        num_bins=config.num_bins,
        dropout=config.dropout,
        simnorm_dim=config.simnorm_dim,
    )


def make_policy_prior(config: TDMPC2Config, action_dim: int) -> PolicyPrior:
    """The :class:`PolicyPrior` of ``config``'s sizes for ``action_dim`` actions."""
    return PolicyPrior(mlp_dim=config.mlp_dim, action_dim=action_dim)
