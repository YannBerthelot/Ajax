"""DreamerV3 world-model networks: block-GRU RSSM, vector encoder, decoder, heads.

A transcription of the paper-era code ``danijar/dreamerv3@2411f7d`` with the
upstream bug fix ``29eb964`` (the fidelity target, ``docs/world_models/
deviations.md`` section 1; ``DESIGN.md`` section 6.2), for vector
observations and in float32 (deviation D4). The specification is
``docs/world_models/dreamerv3_spec.md`` sections 1-2 (Algorithms A-C);
parity with the reference code itself is tested in
``tests/agents/DreamerV3/test_dreamerv3_parity.py`` on fixtures that the
reference produced
(``docs/world_models/parity/dreamerv3_world_model_fixtures.py``).

Reference file:line citations are to ``29eb964``. Its ``dreamerv3/nets.py``
differs from 2411f7d in the encoder's integer-observation one-hot
(``03bc307``, unused for float observations), a CNN flag rename
(``1a532a7``) and the removal of ``Initializer.FORCE_STDDEV`` (inert at
2411f7d, whose ``winit_scale`` is 0.0). Line ``n`` of 2411f7d is line ``n``
of 29eb964 up to 237, ``n + 1`` for 238-247, ``n + 5`` for 248-818 (the
networks, heads and ``Linear`` / ``BlockLinear``), ``n + 4`` for 820-843 and
``n + 1`` from 848 on (``Initializer._fans``, ``get_act``).

Layout. Every hidden layer is ``Dense -> RMSNorm(eps 1e-4) -> SiLU`` with the
bias kept (:class:`ajax.networks.blocks.NormedMLP`, ``NormedMLP.dreamerv3``)
and every kernel is drawn from the fan-in truncated normal scaled by its
layer's ``outscale`` (``trunc_normal_fan_in``, deviation D23). The parameter
tree of :class:`WorldModel` is::

    enc       Dense_i, RMSNorm_i            3 hidden layers (tokens = layer 3)
    rssm      dynin0/1/2 (Dense_0, RMSNorm_0), dynhid0 (BlockLinear),
              dynhid0_norm (RMSNorm), dyngru (BlockLinear),
              obs (1 layer), obslogit, prior (2 layers), priorlogit
    dec       mlp (3 layers), out (outscale 0.1)
    rew       mlp (1 layer), out (255 logits, outscale 0)
    con       mlp (1 layer), out (1 logit, outscale 1)

Reference names: ``dyn`` is ``rssm`` here, ``dyn0`` / ``dyncore`` are
``dynhid0`` / ``dyngru`` (the later code's names, dreamerv3_spec Algorithm
A) and ``img0`` / ``img1`` / ``imglogit`` are ``prior`` / ``priorlogit``.
The block-GRU gate layout is the reference's ``[reset, cand, update]``
within each block and the head inputs are ``concat(deter, stoch)``
(2411f7d's decoder order; dreamerv3_spec 2.17), so the reference's weights
load without permutation. The one shape difference: 2411f7d's two-hot heads
emit ``bins + 1 = 256`` logits and drop the last (``nets.py:437-443``); the
reward head here emits the 255 that are used, as the later code does
(deviations.md section 1, "Two-hot output width"; layout only).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ajax.agents.DreamerV3.distributions import OneHot
from ajax.agents.DreamerV3.state import DreamerV3Config
from ajax.distributional import symlog
from ajax.networks.blocks import NormedMLP, linear
from ajax.networks.utils import trunc_normal_fan_in

#: DreamerV3's fan-in truncated normal (dreamerv3_spec 1.5, deviation D23).
KERNEL_INIT = "trunc_normal_fan_in"
#: 2411f7d vector-decoder output scale: each vector head was a ``Dist`` built
#: without ``outscale``, so its default 0.1 applied (``nets.py:326, :413``;
#: deviations.md section 1; later code: 1.0).
DECODER_OUTSCALE = 0.1
#: RSSM depths, fixed at every model size (``configs.yaml`` ``dyn.rssm``):
#: ``imglayers`` (prior), ``obslayers`` (posterior); ``dynlayers = 1`` is the
#: single ``dynhid0`` layer of :meth:`RSSM.core`.
PRIOR_LAYERS = 2
POSTERIOR_LAYERS = 1


class RSSMState(NamedTuple):
    """RSSM carry: ``deter [..., D]`` and the one-hot ``stoch [..., S, C]``."""

    deter: jax.Array
    stoch: jax.Array


class RSSMFeatures(NamedTuple):
    """One RSSM step's output: the new state and the raw (pre-unimix) logits.

    ``stoch`` is the straight-through sample of the posterior (observe) or the
    prior (imagine); ``logits`` are the logits it was sampled from, before the
    uniform mixture (:meth:`OneHot.from_logits` applies it).
    """

    deter: jax.Array
    stoch: jax.Array
    logits: jax.Array


def initial_state(config: DreamerV3Config, batch_shape: tuple[int, ...]) -> RSSMState:
    """The zero initial state (not learned; dreamerv3_spec 2.2, ``nets.py:39-45``)."""
    return RSSMState(
        deter=jnp.zeros((*batch_shape, config.deter), jnp.float32),
        stoch=jnp.zeros((*batch_shape, config.stoch, config.classes), jnp.float32),
    )


def features(deter: jax.Array, stoch: jax.Array) -> jax.Array:
    """Head input ``concat(deter, stoch_flat)``, ``[..., D + S C]``.

    The input of the decoder (2411f7d order, ``nets.py:282, :303``), reward
    and continue heads, actor and critic (``configs.yaml`` ``inputs: [deter,
    stoch]``; ``nets.py:791-818`` ``Input``).
    """
    return jnp.concatenate([deter, stoch.reshape(*stoch.shape[:-2], -1)], -1)


def encode_action(action: jax.Array, num_classes: Optional[int]) -> jax.Array:
    """The dynamics' action input: one-hot for discrete actions, else float.

    ``jaxutils.onehot_dict`` (``29eb964:dreamerv3/agent.py:233``, ``:135``):
    integer actions become ``one_hot(action, num_classes)``; continuous
    actions pass through unchanged (no symlog, no clipping -- the core bounds
    them, :meth:`RSSM.core`). ``num_classes=None`` marks a continuous space.
    """
    if num_classes is None:
        return jnp.asarray(action, jnp.float32)
    return jax.nn.one_hot(action, num_classes, dtype=jnp.float32)


class BlockLinear(nn.Module):
    """Block-diagonal dense layer (dreamerv3_spec 1.2; ``nets.py:643-706``).

    The input ``[..., I]`` is split into ``blocks`` equal chunks; output block
    ``k`` is ``x_k @ W_k`` with ``W [blocks, I / blocks, features / blocks]``,
    plus a zero-initialised bias ``[features]``. Equivalent to a dense layer
    whose kernel is ``block_diag(W_0, ..., W_{g-1})``. The kernel is
    initialised with fan-in = the **full** input width ``I`` (the reference's
    ``Initializer._fans``, ``nets.py:874-885`` with ``block_fans=False``,
    multiplies the per-block fan by ``g``; :func:`trunc_normal_fan_in` does
    the same for a rank-3 kernel), so ``Std[W] = 1 / sqrt(I)``, not
    ``1 / sqrt(I / g)``. The outscale is 1: the reference's RSSM builds
    ``dyn0`` and ``dyncore`` with BlockLinear's default (``nets.py:161-166``),
    and its only other BlockLinear is the image decoder's ``space0``, out of
    scope here.
    """

    features: int
    blocks: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        g, width = self.blocks, x.shape[-1]
        if width % g or self.features % g:
            raise ValueError(
                f"BlockLinear needs input ({width}) and output ({self.features})"
                f" widths divisible by blocks ({g})"
            )
        kernel = self.param(
            "kernel", trunc_normal_fan_in(), (g, width // g, self.features // g)
        )
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        lead = x.shape[:-1]
        y = jnp.einsum("...ki,kio->...ko", x.reshape(*lead, g, width // g), kernel)
        return y.reshape(*lead, self.features) + bias


class RSSM(nn.Module):
    """Block-GRU recurrent state-space model (dreamerv3_spec 2.1-2.8).

    The methods follow the spec's algorithms: :meth:`core` (the sequence
    model ``h_t = f(h_{t-1}, z_{t-1}, a_{t-1})``, Algorithm A),
    :meth:`observe_step` (filtering, Algorithm B), :meth:`imagine_step`
    (Algorithm C), then the two latent heads :meth:`posterior_logits`
    (``q(z_t | h_t, e_t)``) and :meth:`prior_logits` (``p(z_t | h_t)``).

    It is the world model's ``rssm`` submodule and is applied on its own
    parameter subtree: ``RSSM(config).apply({"params": params["rssm"]}, ...,
    method=RSSM.observe_step)`` (:func:`observe`).
    """

    config: DreamerV3Config

    def setup(self) -> None:
        c = self.config
        self.dynin0 = NormedMLP.dreamerv3(1, c.hidden)
        self.dynin1 = NormedMLP.dreamerv3(1, c.hidden)
        self.dynin2 = NormedMLP.dreamerv3(1, c.hidden)
        self.dynhid0 = BlockLinear(c.deter, c.blocks)
        self.dynhid0_norm = nn.RMSNorm(epsilon=1e-4)
        self.dyngru = BlockLinear(3 * c.deter, c.blocks)
        self.obs = NormedMLP.dreamerv3(POSTERIOR_LAYERS, c.hidden)
        self.obslogit = linear(c.stoch * c.classes, KERNEL_INIT)
        self.prior = NormedMLP.dreamerv3(PRIOR_LAYERS, c.hidden)
        self.priorlogit = linear(c.stoch * c.classes, KERNEL_INIT)

    def core(self, deter: jax.Array, stoch: jax.Array, action: jax.Array) -> jax.Array:
        """One block-GRU step (Algorithm A; ``nets.py:119-123, :151-173``).

        ``deter [..., D]``, ``stoch [..., S, C]``, ``action [..., A]`` (already
        encoded, :func:`encode_action`) give the next ``deter [..., D]``:

        * the action is divided by ``sg(max(1, |a|))`` (components beyond
          ``+-1`` become their sign, with gradient ``1 / |a|``);
        * ``dynin0/1/2`` embed ``deter``, the flattened ``stoch`` and the
          action, each ``Dense -> RMSNorm -> SiLU`` of width ``hidden``;
        * the three embeddings are concatenated and appended to every block's
          slice of ``deter`` -- the embedding of the full ``deter`` is the
          only mixing between blocks;
        * ``dynhid0``: BlockLinear to ``D``, RMSNorm over the **full** ``D``
          vector, SiLU; ``dyngru``: BlockLinear to ``3 D`` without norm or
          activation, read per block as ``[reset | cand | update]``;
        * ``cand = tanh(sigmoid(reset) * cand)`` (the reset gate scales the
          whole candidate pre-activation), ``u = sigmoid(update - 1)``,
          ``deter' = u * cand + (1 - u) * deter``.
        """
        c = self.config
        lead, g = deter.shape[:-1], c.blocks
        stoch = stoch.reshape(*lead, c.stoch * c.classes)
        action = jnp.asarray(action, jnp.float32)
        action = action / jax.lax.stop_gradient(jnp.maximum(1, jnp.abs(action)))
        x = jnp.concatenate(
            [self.dynin0(deter), self.dynin1(stoch), self.dynin2(action)], -1
        )
        x = jnp.broadcast_to(x[..., None, :], (*lead, g, x.shape[-1]))
        x = jnp.concatenate([deter.reshape(*lead, g, c.deter // g), x], -1)
        x = x.reshape(*lead, -1)
        x = nn.silu(self.dynhid0_norm(self.dynhid0(x)))
        gates = self.dyngru(x).reshape(*lead, g, 3, c.deter // g)
        reset, cand, update = (
            gates[..., i, :].reshape(*lead, c.deter) for i in range(3)
        )
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        return update * cand + (1 - update) * deter

    def observe_step(
        self,
        carry: RSSMState,
        token: jax.Array,
        action: jax.Array,
        is_first: jax.Array,
        noise: jax.Array,
    ) -> tuple[RSSMState, RSSMFeatures]:
        """One filtering step (Algorithm B; ``nets.py:53-76``).

        ``carry`` is ``(h_{t-1}, z_{t-1})``, ``token`` the encoder output
        ``e_t``, ``action`` the encoded previous action ``a_{t-1}``
        (:func:`encode_action`), ``is_first [...]`` bool and ``noise [..., S,
        C]`` the Gumbel noise of the posterior sample. Where ``is_first``,
        the previous ``deter``, ``stoch`` **and** action are zeroed before
        the core, so an episode starts from ``h_t = core(0, 0, 0)``
        (dreamerv3_spec 2.2). The action is zeroed after its encoding, so a
        discrete ``one_hot(0)`` is masked too: 29eb964 encodes the action
        before the step (``agent.py:233``) and masks the encoded action
        (``nets.py:63-64``), as the later code's mask-encode-mask does. Then
        ``z_t ~ q(h_t, e_t)``, sampled straight-through from the mixed
        distribution.
        """
        reset = is_first[..., None]
        deter = jnp.where(reset, 0, carry.deter)
        stoch = jnp.where(reset[..., None], 0, carry.stoch)
        action = jnp.where(reset, 0, action)
        deter = self.core(deter, stoch, action)
        logits = self.posterior_logits(deter, token)
        stoch = OneHot.from_logits(logits, self.config.unimix).sample(noise)
        return RSSMState(deter, stoch), RSSMFeatures(deter, stoch, logits)

    def imagine_step(
        self, carry: RSSMState, action: jax.Array, noise: jax.Array
    ) -> tuple[RSSMState, RSSMFeatures]:
        """One imagination step (Algorithm C; ``nets.py:78-95``).

        ``h_t = core(h_{t-1}, z_{t-1}, a_{t-1})`` without resets, then
        ``z_t ~ p(h_t)`` sampled straight-through from the **mixed** prior
        (unimix, dreamerv3_spec 2.8) with ``noise [..., S, C]``. The action
        comes from the caller (the actor, with its stop-gradient on the
        policy input, is the actor-critic's concern: ``agent.py:252-257``).
        """
        deter = self.core(carry.deter, carry.stoch, action)
        logits = self.prior_logits(deter)
        stoch = OneHot.from_logits(logits, self.config.unimix).sample(noise)
        return RSSMState(deter, stoch), RSSMFeatures(deter, stoch, logits)

    def posterior_logits(self, deter: jax.Array, token: jax.Array) -> jax.Array:
        """Raw posterior logits ``[..., S, C]`` from ``concat(h_t, e_t)``.

        ``obs0`` (``Dense -> RMSNorm -> SiLU``) then ``obslogit`` (outscale 1;
        ``nets.py:66-69, :207-211``; dreamerv3_spec 2.6).
        """
        x = self.obs(jnp.concatenate([deter, token], -1))
        return self.obslogit(x).reshape(
            *x.shape[:-1], self.config.stoch, self.config.classes
        )

    def prior_logits(self, deter: jax.Array) -> jax.Array:
        """Raw prior logits ``[..., S, C]`` from ``h_t`` alone.

        Two ``Dense -> RMSNorm -> SiLU`` layers then ``priorlogit`` (outscale
        1; ``nets.py:112-117``; dreamerv3_spec 2.5). The world-model loss
        computes them for all ``(b, t)`` at once from the posterior's ``h_t``
        after the observe scan (``nets.py:97-99``).
        """
        x = self.prior(deter)
        return self.priorlogit(x).reshape(
            *x.shape[:-1], self.config.stoch, self.config.classes
        )


class MLPHead(nn.Module):
    """``layers`` hidden layers then a linear output of ``features`` units.

    The reference ``MLP`` + ``Dist`` (``nets.py:372-441``): ``outscale``
    reaches only the output layer (dreamerv3_spec 1.4, 1.5).
    """

    layers: int
    units: int
    features: int
    outscale: float

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = NormedMLP.dreamerv3(self.layers, self.units, name="mlp")(x)
        return linear(self.features, KERNEL_INIT, outscale=self.outscale, name="out")(x)


class WorldModel(nn.Module):
    """DreamerV3 world model for a flat vector observation.

    Submodules ``enc`` (vector encoder), ``rssm`` (:class:`RSSM`), ``dec``
    (vector decoder), ``rew`` (two-hot reward head) and ``con`` (continue
    head); see the module docstring for the parameter tree. One decoder head
    covers the whole observation vector (deviation D5). ``__call__`` touches
    every parameter and exists for ``init`` (:func:`init_world_model`); the
    methods are applied with ``model.apply({'params': p}, ...,
    method=WorldModel.<name>)``, the RSSM's on ``p['rssm']`` (:class:`RSSM`).
    """

    config: DreamerV3Config
    obs_dim: int

    def setup(self) -> None:
        c = self.config
        self.enc = NormedMLP.dreamerv3(c.enc_layers, c.units)
        self.rssm = RSSM(c)
        self.dec = MLPHead(c.dec_layers, c.units, self.obs_dim, DECODER_OUTSCALE)
        # 255 logits, not 2411f7d's 256 with the last dropped (deviations.md
        # section 1, "Two-hot output width").
        self.rew = MLPHead(c.rew_layers, c.units, c.bins, 0.0)
        self.con = MLPHead(c.con_layers, c.units, 1, 1.0)

    def __call__(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, ...]:
        c = self.config
        batch_shape = obs.shape[:-1]
        _, post = self.rssm.observe_step(
            initial_state(c, batch_shape),
            self.encode(obs),
            action,
            jnp.zeros(batch_shape, bool),
            jnp.zeros((*batch_shape, c.stoch, c.classes), jnp.float32),
        )
        feat = features(post.deter, post.stoch)
        return (
            post.logits,
            self.rssm.prior_logits(post.deter),
            self.decode(feat),
            self.reward_logits(feat),
            self.cont_logit(feat),
        )

    def encode(self, obs: jax.Array) -> jax.Array:
        """Tokens ``e_t [..., units]`` of a vector observation ``[..., O]``.

        ``symlog`` then 3 x ``Dense -> RMSNorm -> SiLU``; the tokens are the
        output of the third layer, with no projection (dreamerv3_spec 2.9;
        ``nets.py:246-261``). 2411f7d symlogs integer observations too, after
        casting them to float (deviations.md section 1).
        """
        return self.enc(symlog(jnp.asarray(obs, jnp.float32)))

    def decode(self, feat: jax.Array) -> jax.Array:
        """Symlog-space reconstruction ``[..., O]`` of the observation."""
        return self.dec(feat)

    def reward_logits(self, feat: jax.Array) -> jax.Array:
        """Two-hot reward logits ``[..., bins]`` (zero at init; spec 2.12)."""
        return self.rew(feat)

    def cont_logit(self, feat: jax.Array) -> jax.Array:
        """Continue logit ``[...]`` (spec 2.13)."""
        return self.con(feat)[..., 0]


def init_world_model(
    key: jax.Array, config: DreamerV3Config, obs_dim: int, action_dim: int
) -> dict:
    """Initial parameters of ``WorldModel(config, obs_dim)``.

    ``action_dim`` is the width of the encoded action (the number of classes
    for a discrete space).
    """
    model = WorldModel(config, obs_dim)
    obs = jnp.zeros((1, obs_dim), jnp.float32)
    action = jnp.zeros((1, action_dim), jnp.float32)
    return model.init(key, obs, action)["params"]


def observe(
    rssm: RSSM,
    params: dict,
    carry: RSSMState,
    tokens: jax.Array,
    actions: jax.Array,
    is_first: jax.Array,
    noise: jax.Array,
) -> tuple[RSSMState, RSSMFeatures]:
    """Filter a batch of sequences: :meth:`RSSM.observe_step` over time.

    ``params`` are the RSSM's own (the world model's ``params['rssm']``).
    Inputs are batch-major: ``tokens [B, T, units]``, ``actions [B, T, A]``
    (``a_{t-1}`` for each ``x_t``), ``is_first [B, T]``, ``noise [B, T, S, C]``;
    ``carry`` is the state before the first step. Returns the final carry and
    the per-step features ``[B, T, ...]`` (``nets.py:59-62``: a scan over the
    time axis, BPTT through all ``T`` steps).
    """

    def step(state, inputs):
        return rssm.apply({"params": params}, state, *inputs, method=RSSM.observe_step)

    inputs = jax.tree.map(
        lambda x: jnp.swapaxes(x, 0, 1), (tokens, actions, is_first, noise)
    )
    carry, feats = jax.lax.scan(step, carry, inputs)
    return carry, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), feats)
