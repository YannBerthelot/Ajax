"""EVarEst measurement / instrumentation as composable :class:`Extension`s.

This module ports the measurement code from
``/home/yberthel/EVAREST/evarest/rl/`` to the Ajax Extension surface so
the same plasticity / decomposition / cliff diagnostics can be folded
into any Ajax agent (SAC, DQN, REDQ, TD3 off-policy; PPO, PQN, APO, AVG
on-policy) without touching the agent code. Each extension is a frozen
dataclass carrying its own hyperparameters; the user opts in via
``extensions=[ConditioningMetrics(), BiasVoreDecomposition(...), ...]``.

Coverage
--------
* :class:`ConditioningMetrics`     — eval_metrics phase, plasticity
  diagnostics (srank, dormant fraction, feat / weight L2 norms).
  Reads the agent's replay buffer (off-policy) OR
  ``agent_state.last_rollout`` (on-policy with ``expose_recent_rollout``),
  auto-detected.
* :class:`BiasVoreDecomposition`   — eval_metrics phase, EVarEst
  bias^2 / Var(residual) / MSE / |bias| decomposition of the critic
  objective. Mirrors the math in EVAREST's ``loss.py`` /
  ``mechanism.py``.
* :class:`CliffEta`                — eval_metrics phase, gauge-breaking
  cliff measurement (rho, Delta, S_task, eta = rho(1-rho)Delta^2/S_task)
  on the CliffCorridor diagnostic env. Mirrors EVAREST's
  ``cliff_measure.py``.
* :class:`DiagnosticSnapshots`     — post_update phase, periodic
  agent-state snapshot for offline post-mortem analysis (the chunked-
  training protocol from EVAREST's ``snapshots.py``).
* :class:`BiasVorePenalty`         — critic_loss phase, the EVarEst
  critic objective itself. ``penalty(alpha) = ((1-2alpha)/alpha) *
  Var(q_preds - target_q)`` added on top of the agent's vanilla MSE.

Each extension owns its own hyperparameters as frozen-dataclass fields
(no per-extension agent-side kwargs). Empty / mis-shaped state ⇒ the
extension emits an empty metrics dict so a misconfigured stack is
loud-failure rather than a wrong-numbers silent failure where possible.
The math is intentionally kept identical to the EVAREST sources so that
Phase 4c (EVAREST integration) becomes a thin wiring change.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ajax.extensions.base import Extension, ExtensionContext
from ajax.networks.networks import predict_value

# Public metric keys — duplicated here so Ajax callers don't need to
# import from EVAREST. The strings are identical so dashboards built on
# EVAREST logs render Ajax-side runs without change.
COND_METRIC_KEYS: Tuple[str, ...] = (
    "Cond/critic_srank",
    "Cond/critic_dormant_frac",
    "Cond/critic_feat_norm",
    "Cond/critic_weight_norm",
)
EVAREST_DECOMP_KEYS: Tuple[str, ...] = (
    "EVarEst/bias_sq",
    "EVarEst/var_resid",
    "EVarEst/mse",
    "EVarEst/abs_bias",
)
CLIFF_METRIC_KEYS: Tuple[str, ...] = (
    "Cliff/rho_hat",
    "Cliff/delta_hat",
    "Cliff/s_task_hat",
    "Cliff/eta_hat",
)


# ---------------------------------------------------------------------------
# Plasticity / conditioning primitives (EVAREST evarest/rl/conditioning.py)
# ---------------------------------------------------------------------------


def srank(feats: jax.Array, delta: float = 0.01) -> jax.Array:
    """Kumar et al. (2021) effective rank.

    Smallest k whose top-k singular values capture >= 1-delta of the
    total spectral mass. ``feats`` is the (batch, d) feature matrix.
    """
    s = jnp.linalg.svd(feats, compute_uv=False)
    cum = jnp.cumsum(s) / (jnp.sum(s) + 1e-12)
    return (jnp.sum(cum < (1.0 - delta)) + 1).astype(jnp.float32)


def dormant_fraction(feats: jax.Array, tau: float = 0.025) -> jax.Array:
    """Sokar et al. (2023) dormant-unit fraction.

    A unit is dormant when its mean-|activation| score, normalised by
    the layer mean, is <= tau. ``feats`` is the (batch, d) post-
    activation feature matrix.
    """
    score = jnp.mean(jnp.abs(feats), axis=0)
    norm = score / (jnp.mean(score) + 1e-9)
    return jnp.mean((norm <= tau).astype(jnp.float32))


def _global_l2(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


# ---------------------------------------------------------------------------
# State-batch sourcing helpers
# ---------------------------------------------------------------------------


def _has_buffer(agent_state: Any) -> bool:
    """True iff the agent carries a flashbax replay buffer."""
    cs = getattr(agent_state, "collector_state", None)
    if cs is None:
        return False
    return getattr(cs, "buffer_state", None) is not None


def _has_last_rollout(agent_state: Any) -> bool:
    """True iff Gap A exposed a ``last_rollout`` subsample."""
    return getattr(agent_state, "last_rollout", None) is not None


def _sample_state_batch(
    agent_state: Any,
    rng: jax.Array,
    buffer: Optional[Any],
    on_policy_take: int,
) -> Tuple[jax.Array, jax.Array]:
    """Return ``(obs, act)`` for a measurement batch.

    Off-policy: sample from the replay buffer via the public
    ``get_batch_from_buffer`` seam. On-policy: read the fixed-size
    subsample on ``agent_state.last_rollout`` and slice the first
    ``on_policy_take`` rows. Raises ``ValueError`` if neither source is
    populated — extensions need to fail loudly when their state-batch
    contract is broken.
    """
    if _has_buffer(agent_state) and buffer is not None:
        # Lazy import to keep the rest of the module buffer-agnostic
        # (CliffEta / DiagnosticSnapshots don't need flashbax at import
        # time so a missing optional dep doesn't break their import).
        from ajax.buffers.utils import get_batch_from_buffer

        batch = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, rng
        )
        return batch[0], batch[5]
    if _has_last_rollout(agent_state):
        sub = agent_state.last_rollout
        return sub.obs[:on_policy_take], sub.action[:on_policy_take]
    raise ValueError(
        "ConditioningMetrics / BiasVoreDecomposition need either a replay "
        "buffer (off-policy) or agent_state.last_rollout populated via "
        "BaseAgentConfig.expose_recent_rollout=True (on-policy)."
    )


# ---------------------------------------------------------------------------
# Conditioning / plasticity extension
# ---------------------------------------------------------------------------


def _encoder_features_ppo_sac(critic_state: Any, x: jax.Array) -> jax.Array:
    """Per-ensemble-member encoder features for SAC/PPO MultiCritic.

    Returns ``(num_critics, batch, d)``.
    """
    return critic_state.apply_fn(
        critic_state.params, x, method="apply_encoder_ensemble"
    )


def _encoder_features_qnet(actor_state: Any, x: jax.Array) -> jax.Array:
    """Penultimate features sown by DQN / PQN Q-networks. ``(1, batch, d)``."""
    _, mv = actor_state.apply_fn(actor_state.params, x, mutable=["intermediates"])
    feat = mv["intermediates"]["encoder_features"][0]
    return feat[None]


@dataclass(frozen=True)
class ConditioningMetrics(Extension):
    """Plasticity diagnostics on a state batch sampled from the agent's
    visited-state distribution.

    Auto-detects the critic location and input convention:
    * ``critic_loc='critic'`` + ``concat_action=True``  ⇒ SAC-style
      Q(obs,act) critic on ``critic_state``.
    * ``critic_loc='critic'`` + ``concat_action=False`` ⇒ PPO-style
      V(obs) critic on ``critic_state``.
    * ``critic_loc='actor'``  ⇒ DQN / PQN: Q-network on ``actor_state``,
      penultimate features read via the ``encoder_features`` sow.

    Auto-detection (``critic_loc='auto'``, ``concat_action='auto'``)
    picks the actor branch when the critic_state has no apply_fn
    intermediate sow point, the critic branch otherwise; for the
    critic branch it concats actions iff the on-policy ``last_rollout``
    action width is compatible with the critic's input dim.

    Hyperparameters live on this dataclass (no per-extension agent-side
    kwargs). Pass ``buffer=agent.buffer`` for off-policy agents whose
    buffer object isn't on the agent state itself; on-policy agents
    must set ``BaseAgentConfig.expose_recent_rollout=True``.
    """

    buffer: Any = None
    on_policy_batch: int = 256
    critic_loc: str = "auto"  # 'auto' | 'critic' | 'actor'
    concat_action: Any = "auto"  # bool | 'auto'
    name: str = "conditioning_metrics"

    def _features(self, agent_state: Any, obs: jax.Array, act: jax.Array) -> jax.Array:
        loc = self.critic_loc
        if loc == "auto":
            # SAC-style MultiCritic critic exposes apply_encoder_ensemble;
            # DQN/PQN Q-network is on actor_state and sows its features.
            cs = agent_state.critic_state
            has_ensemble = hasattr(cs, "apply_fn") and callable(
                getattr(cs, "apply_fn", None)
            )
            loc = "critic" if has_ensemble else "actor"
        if loc == "actor":
            return _encoder_features_qnet(agent_state.actor_state, obs)
        # critic branch: V(obs) (PPO) or Q(obs,act) (SAC). Decide
        # whether to concat the action via the explicit flag, falling
        # back to a buffer-vs-on-policy heuristic.
        concat = self.concat_action
        if concat == "auto":
            # Off-policy buffer agents (SAC) carry the (obs, act)
            # critic; on-policy rollout-only agents (PPO) carry V(obs).
            concat = bool(_has_buffer(agent_state) and self.buffer is not None)
        if concat:
            # Reshape action to match obs's batch convention; the
            # discrete-action subsample may have a trailing 1-dim or
            # not, so we flatten it back to (B, action_dim).
            if act.ndim == obs.ndim - 1:
                act = act[..., None]
            x = jnp.concatenate([obs, act.astype(obs.dtype)], axis=-1)
        else:
            x = obs
        return _encoder_features_ppo_sac(agent_state.critic_state, x)

    def _critic_params(self, agent_state: Any) -> Any:
        loc = self.critic_loc
        if loc == "auto":
            loc = "critic" if hasattr(agent_state.critic_state, "params") else "actor"
        return getattr(agent_state, f"{loc}_state").params

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> dict:
        del ext_state, ctx
        try:
            obs, act = _sample_state_batch(
                agent_state, rng, self.buffer, self.on_policy_batch
            )
        except ValueError:
            # If the state-batch source isn't ready (e.g. eval fires before
            # the first rollout populates last_rollout), emit an empty
            # dict so the eval log still completes.
            return {}
        feats = self._features(agent_state, obs, act)
        return dict(
            zip(
                COND_METRIC_KEYS,
                (
                    jax.vmap(srank)(feats).mean(),
                    jax.vmap(dormant_fraction)(feats).mean(),
                    jnp.linalg.norm(feats, axis=-1).mean(),
                    _global_l2(self._critic_params(agent_state)),
                ),
            )
        )


# ---------------------------------------------------------------------------
# Bias / VORE decomposition extension (eval-time)
# ---------------------------------------------------------------------------


def _bias_vore_decomposition_dict(residual: jax.Array) -> dict:
    """The four-term decomposition keyed by :data:`EVAREST_DECOMP_KEYS`."""
    bias = jnp.mean(residual)
    return dict(
        zip(
            EVAREST_DECOMP_KEYS,
            (
                bias**2,
                jnp.var(residual),
                jnp.mean(residual**2),
                jnp.abs(bias),
            ),
        )
    )


@dataclass(frozen=True)
class BiasVoreDecomposition(Extension):
    """EVarEst bias^2 / Var(residual) / MSE / |bias| decomposition.

    Ports the math from EVAREST ``loss.py`` /
    ``mechanism.make_evarest_decomposition_metrics``. The residual is
    reconstructed from the agent's *current* critic on a measurement
    batch sampled the same way :class:`ConditioningMetrics` samples
    (replay buffer for off-policy, ``last_rollout`` for on-policy). The
    SAC variant uses :func:`ajax.agents.SAC.core.compute_td_target` to
    rebuild the Bellman target; the PPO variant uses the supplied
    ``value_target_fn`` callable (kept generic so the extension stays
    agent-agnostic). When ``value_target_fn`` is None the extension
    assumes a SAC-style Q-critic.
    """

    buffer: Any = None
    on_policy_batch: int = 256
    gamma: float = 0.99
    reward_scale: float = 1.0
    # Optional override for non-SAC agents:
    # ``(agent_state, batch_dict, rng) -> (target, q_preds)``.
    value_target_fn: Optional[Callable] = None
    name: str = "bias_vore_decomposition"

    def _sac_residual(self, agent_state: Any, rng: jax.Array) -> jax.Array:
        from ajax.agents.SAC import core
        from ajax.buffers.utils import get_batch_from_buffer

        tkey, skey = jax.random.split(rng)
        batch = get_batch_from_buffer(
            self.buffer, agent_state.collector_state.buffer_state, skey
        )
        obs, terminated, _trunc, next_obs, rewards, act = batch[:6]
        alpha = jnp.exp(agent_state.alpha.params["log_alpha"])
        target_q = core.compute_td_target(
            actor_state=agent_state.actor_state,
            critic_state=agent_state.critic_state,
            next_observations=next_obs,
            dones=terminated,
            rewards=rewards,
            gamma=self.gamma,
            alpha=alpha,
            rng=tkey,
            recurrent=False,
            reward_scale=self.reward_scale,
        )
        q_preds = predict_value(
            critic_state=agent_state.critic_state,
            critic_params=agent_state.critic_state.params,
            x=jnp.concatenate([obs, act], axis=-1),
        )
        return q_preds - target_q

    def _on_policy_residual(self, agent_state: Any) -> jax.Array:
        """Generic on-policy residual: V(obs) - mean(reward) bootstrap.

        On-policy agents that opted into Gap A only expose ``(obs, act,
        reward, next_obs, done)`` -- enough for a single-step TD
        residual against the agent's *current* value head. The exact
        target (GAE for PPO, Q(lambda) for PQN, average-reward for APO/
        AVG) can be supplied via ``value_target_fn``; absent that, we
        compute ``r + gamma (1-d) V(s') - V(s)`` (one-step TD), which
        is the universal common subexpression of the four targets.
        """
        sub = agent_state.last_rollout
        cs = agent_state.critic_state
        v = predict_value(critic_state=cs, critic_params=cs.params, x=sub.obs).squeeze(
            0
        )
        v_next = predict_value(
            critic_state=cs, critic_params=cs.params, x=sub.next_obs
        ).squeeze(0)
        # Broadcast shapes so the subtraction is well-defined. Transition
        # stores terminated/truncated separately; episode-end ≡ either.
        reward = sub.reward.reshape(v.shape)
        done = jnp.logical_or(sub.terminated, sub.truncated).reshape(v.shape)
        target = reward + self.gamma * (1.0 - done) * v_next.astype(v.dtype)
        return v - target

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> dict:
        del ext_state, ctx
        if self.value_target_fn is not None:
            tkey, skey = jax.random.split(rng)
            try:
                obs, act = _sample_state_batch(
                    agent_state, skey, self.buffer, self.on_policy_batch
                )
            except ValueError:
                return {}
            batch_dict = {"observations": obs, "actions": act}
            target, q_preds = self.value_target_fn(agent_state, batch_dict, tkey)
            resid = q_preds - target
            return _bias_vore_decomposition_dict(resid)

        if _has_buffer(agent_state) and self.buffer is not None:
            resid = self._sac_residual(agent_state, rng)
            return _bias_vore_decomposition_dict(resid)
        if _has_last_rollout(agent_state):
            resid = self._on_policy_residual(agent_state)
            return _bias_vore_decomposition_dict(resid)
        return {}


# ---------------------------------------------------------------------------
# CliffCorridor gauge-breaking measurement extension
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CliffEta(Extension):
    """Gauge-breaking cliff measurement (eta = rho(1-rho)Delta^2 / S_task).

    Ports :func:`evarest.rl.cliff_measure.measure_cliff` verbatim. Rolls
    out the agent's policy on the CliffCorridor diagnostic env (or any
    gymnax env exposing ``info["terminated"]``, ``info["task_reward"]``
    and ``info["alive_reward"]``) and logs the four theory quantities
    every eval. ``n_steps`` and ``n_episodes`` must be Python ints (the
    scan length is static).
    """

    env: Any = None
    env_params: Any = None
    n_steps: int = 200
    n_episodes: int = 256
    gamma: float = 0.99
    stochastic: bool = True
    discrete: bool = False
    epsilon: float = 0.1
    name: str = "cliff_eta"

    def _act(self, actor_state: Any, obs: jax.Array, key: jax.Array) -> jax.Array:
        from ajax.environments.interaction import get_pi

        pi, _ = get_pi(actor_state, actor_state.params, obs, None, False)
        if self.discrete:
            q = pi.q_values
            greedy = jnp.argmax(q, axis=-1)
            krnd, keps = jax.random.split(key)
            rnd = jax.random.randint(krnd, greedy.shape, 0, q.shape[-1])
            explore = jax.random.uniform(keps, greedy.shape) < self.epsilon
            return jnp.where(explore, rnd, greedy)
        return pi.sample(seed=key) if self.stochastic else pi.mean()

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> dict:
        del ext_state, ctx
        if self.env is None:
            return {}
        env, params = self.env, self.env_params
        n_steps = int(self.n_steps)
        n_episodes = int(self.n_episodes)

        key, reset_key = jax.random.split(rng)
        obs0, st0 = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(reset_key, n_episodes), params
        )

        actor_state = agent_state.actor_state

        def body(carry, _):
            key, obs, st, done = carry
            key, ka, ks = jax.random.split(key, 3)
            action = self._act(actor_state, obs, ka)
            obs2, st2, _r, d, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(ks, n_episodes), st, action, params
            )
            running = 1.0 - done.astype(jnp.float32)
            out = (
                info["terminated"].astype(jnp.float32),
                info["task_reward"],
                info["alive_reward"],
                running,
            )
            return (key, obs2, st2, jnp.logical_or(done, d)), out

        init = (key, obs0, st0, jnp.zeros(n_episodes, dtype=bool))
        _, (term, task_r, alive_r, running) = jax.lax.scan(
            body, init, None, length=n_steps
        )

        rho = (term * running).max(axis=0).mean()
        disc = (self.gamma ** jnp.arange(n_steps))[:, None]
        delta = (disc * alive_r * running).sum(axis=0).mean()
        wsum = jnp.maximum(running.sum(), 1.0)
        mean_t = (running * task_r).sum() / wsum
        s_task = (running * (task_r - mean_t) ** 2).sum() / wsum
        eta = rho * (1.0 - rho) * delta**2 / (s_task + 1e-8)
        return dict(zip(CLIFF_METRIC_KEYS, (rho, delta, s_task, eta)))


# ---------------------------------------------------------------------------
# Diagnostic snapshots (post_update side-effect)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiagnosticSnapshots(Extension):
    """Periodic agent-state snapshot for offline post-mortem analysis.

    Ports :mod:`evarest.rl.snapshots` to the post_update phase. At every
    ``every_n_iters``-th call the extension pickles a chunk containing:
    * a uniform random subsample of the replay buffer (off-policy) OR
      the current ``last_rollout`` (on-policy),
    * the current actor parameters.
    Files are written under ``directory`` with names
    ``snapshot_<chunk_idx>.pkl``; ``chunk_idx`` is the per-instance
    counter stored in ``ext_state``. Snapshots are taken via
    ``jax.experimental.io_callback`` so the host-side pickle never
    blocks the JIT'd train loop.
    """

    directory: str = "./snapshots"
    every_n_iters: int = 100
    n_sub: int = 4096
    buffer: Any = None
    # Subsample-batch size for buffer sampling; matches EVAREST's
    # snapshots._SAMPLE_BATCH default.
    sample_batch: int = 256
    name: str = "diagnostic_snapshots"

    def init_state(self, agent_state: Any, rng: jax.Array) -> Any:
        del agent_state, rng
        os.makedirs(self.directory, exist_ok=True)
        return jnp.asarray(0, dtype=jnp.int32)

    def _take_transitions(self, agent_state: Any, rng: jax.Array) -> dict:
        if _has_buffer(agent_state) and self.buffer is not None:
            from ajax.buffers.utils import get_batch_from_buffer

            n_iter = (self.n_sub + self.sample_batch - 1) // self.sample_batch

            def one(_, k):
                b = get_batch_from_buffer(
                    self.buffer, agent_state.collector_state.buffer_state, k
                )
                return None, (b[0], b[5], b[4], b[3], b[1])

            _, parts = jax.lax.scan(one, None, jax.random.split(rng, n_iter))
            obs, act, rew, next_obs, term = jax.tree.map(
                lambda x: x.reshape((-1,) + x.shape[2:])[: self.n_sub], parts
            )
            return {
                "obs": obs,
                "act": act,
                "rew": rew,
                "next_obs": next_obs,
                "terminated": term,
            }
        if _has_last_rollout(agent_state):
            sub = agent_state.last_rollout
            return {
                "obs": sub.obs,
                "act": sub.action,
                "rew": sub.reward,
                "next_obs": sub.next_obs,
                "terminated": sub.terminated,
                "truncated": sub.truncated,
            }
        return {}

    def _host_save(
        self,
        chunk_idx: jax.Array,
        should_save: jax.Array,
        transitions: dict,
        params: Any,
    ) -> None:
        # Host-side gate: ``should_save`` is a 0-D bool array. We only
        # touch disk when True. Doing the gate host-side keeps the
        # io_callback unconditional, which avoids JAX's
        # "IO effect not supported in vmap-of-cond" limitation when
        # this extension runs under ``jax.vmap`` over seeds.
        if not bool(np.asarray(should_save)):
            return
        path = os.path.join(self.directory, f"snapshot_{int(chunk_idx)}.pkl")
        payload = {
            "chunk_idx": int(chunk_idx),
            "transitions": jax.tree.map(lambda x: np.asarray(x), transitions),
            "actor_params": jax.tree.map(lambda x: np.asarray(x), params),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def post_update(
        self, agent_state: Any, ext_state: Any, ctx: ExtensionContext
    ) -> Tuple[Any, Any]:
        # Snapshot every Nth iteration; ``ext_state`` is the per-instance
        # call counter so the cadence is robust to compile cache reuse.
        new_state = ext_state + 1
        should_save = (new_state % self.every_n_iters) == 0

        transitions = self._take_transitions(agent_state, ctx.rng)
        # Unconditional io_callback — host-side ``_host_save`` is the
        # gate, see its docstring.
        jax.experimental.io_callback(
            self._host_save,
            None,
            new_state,
            should_save,
            transitions,
            agent_state.actor_state.params,
        )
        return agent_state, new_state


# ---------------------------------------------------------------------------
# EVarEst critic objective (additive penalty on the critic loss)
# ---------------------------------------------------------------------------


def evarest_coeff(alpha: float) -> float:
    """The eq. (6) coefficient ``(1 - 2 alpha) / alpha``."""
    if alpha <= 0.0:
        raise ValueError(
            "alpha must be > 0 for the additive EVarEst form "
            "(alpha->0 is the AVEC limit; use a small alpha instead)."
        )
    return (1.0 - 2.0 * alpha) / alpha


@dataclass(frozen=True)
class BiasVorePenalty(Extension):
    """EVarEst critic objective as a ``critic_loss`` extension.

    Ports the additive form of EVAREST's
    :func:`make_evarest_loss` / :func:`make_evarest_ppo_loss` /
    :func:`make_evarest_td_loss`. The agent's vanilla critic loss
    already contains ``MSE = bias^2 + Var(residual)``; this extension
    adds the eq. (6) extra penalty term:

        penalty(alpha) = ((1 - 2 alpha) / alpha) * Var(q_preds - target_q)

    so the total objective becomes
    ``MSE + ((1-2 alpha)/alpha) * Var(residual)``. alpha=0.5 ⇒ coeff
    0 ⇒ identical to vanilla; alpha->0 is the AVEC limit (use a small
    alpha rather than zero). ``coeff_override`` lets callers fix the
    coefficient directly (e.g. to disable the extra penalty entirely
    without recomputing alpha). Reads ``observations`` and (for SAC)
    ``actions`` + ``targets`` from the ``batch`` dict the agent threads
    into the critic-loss fold. The extension is loss-agent-aware:
    SAC's critic-loss fold passes ``actions`` (Q-critic) while
    PPO/PQN/DQN pass only ``observations`` + ``targets`` (V-critic
    or already-gathered q_taken). Both paths reconstruct the same
    residual the agent's MSE term sees, so the bias/Var math is
    bit-for-bit consistent.
    """

    alpha: float = 0.5
    coeff_override: Optional[float] = None
    # PPO carries a 0.5 factor on its MSE; ``loss_scale`` mirrors that
    # so the MSE:Var ratio matches paper eq. (6) for PPO too.
    loss_scale: float = 1.0
    name: str = "bias_vore_penalty"

    def _coeff(self) -> float:
        if self.coeff_override is not None:
            return float(self.coeff_override)
        if self.alpha == 0.5:
            return 0.0
        return evarest_coeff(self.alpha)

    def critic_loss(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: dict,
        ctx: ExtensionContext,
    ) -> jax.Array:
        del ext_state, ctx
        coeff = self._coeff()
        if coeff == 0.0:
            return jnp.asarray(0.0)

        # Two batch-shape conventions the agents use:
        # - SAC critic loss fold passes 'critic_params', 'critic_state',
        #   'observations', 'actions', 'target_q' (or similar) so we
        #   can rebuild q_preds and residual. PPO fold passes
        #   'critic_params', 'critic_state', 'observations', 'targets'.
        # - DQN/PQN fold passes 'q_state', 'observations', 'actions',
        #   'targets'. Q-network case reconstructs q_taken via
        #   predict_q + gather.
        obs = batch.get("observations")
        if obs is None:
            return jnp.asarray(0.0)
        target = batch.get("target_q", batch.get("targets"))
        if target is None:
            return jnp.asarray(0.0)

        # Try the SAC/PPO MultiCritic path first.
        cs = batch.get("critic_state")
        params = batch.get("critic_params")
        actions = batch.get("actions")
        if cs is not None and params is not None:
            if actions is not None:
                # SAC Q-critic: q_preds shape (num_critics, B, 1).
                x = jnp.concatenate([obs, jax.lax.stop_gradient(actions)], axis=-1)
                q_preds = predict_value(critic_state=cs, critic_params=params, x=x)
            else:
                # PPO V-critic: (num_critics=1, B, 1) -> squeeze head.
                q_preds = predict_value(
                    critic_state=cs, critic_params=params, x=obs
                ).squeeze(0)
            resid = q_preds - target
            return self.loss_scale * coeff * jnp.var(resid)

        # DQN / PQN Q-network: q_state in batch.
        q_state = batch.get("q_state")
        if q_state is not None and actions is not None:
            # Reuse the agent's predict_q -> gather to keep numerics aligned
            # with the agent's q_taken.
            try:
                from ajax.agents.DQN.networks import predict_q
            except ImportError:
                return jnp.asarray(0.0)
            q_all = predict_q(q_state, q_state.params, obs)
            # Match q_loss_fn's gather convention: actions may carry a
            # trailing 1-dim or not.
            act_idx = actions if actions.ndim == q_all.ndim - 1 else actions.squeeze(-1)
            q_taken = jnp.take_along_axis(q_all, act_idx[..., None], axis=-1)
            resid = q_taken - target
            return self.loss_scale * coeff * jnp.var(resid)

        return jnp.asarray(0.0)


# Public re-exports for downstream importers.
__all__ = [
    "ConditioningMetrics",
    "BiasVoreDecomposition",
    "CliffEta",
    "DiagnosticSnapshots",
    "BiasVorePenalty",
    "COND_METRIC_KEYS",
    "EVAREST_DECOMP_KEYS",
    "CLIFF_METRIC_KEYS",
    "evarest_coeff",
    "srank",
    "dormant_fraction",
]
