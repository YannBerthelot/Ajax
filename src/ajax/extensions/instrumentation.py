"""EVarEst measurement / instrumentation as composable :class:`Extension`s.

This module ports the measurement code from
``/home/yberthel/EVAREST/evarest/rl/`` to the Ajax Extension surface so
the same plasticity / decomposition diagnostics can be folded
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

import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ajax.extensions.base import Extension, ExtensionContext
from ajax.networks.networks import action_value_input, predict_value

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


def _sample_transitions(
    agent_state: Any,
    rng: jax.Array,
    buffer: Optional[Any],
    on_policy_take: int,
) -> Tuple[jax.Array, ...]:
    """Return ``(obs, act, reward, next_obs, done)`` for a measurement
    batch, one ``(N, features)`` row per transition.

    Off-policy: sample from the replay buffer via the public
    ``get_batch_from_buffer`` seam; its ``next_obs`` is the observation a
    target bootstraps on, so only a termination ends the episode. On-policy:
    flatten the ``(T, n_envs)`` steps of ``agent_state.last_rollout`` into
    rows and keep the first ``on_policy_take``; srank and the dormant
    fraction need one feature row per sample. A rollout's episode ends at a
    termination or a truncation. Raises ``ValueError`` if neither source is
    populated — extensions need to fail loudly when their state-batch
    contract is broken.
    """
    if _has_buffer(agent_state) and buffer is not None:
        # Lazy import to keep the rest of the module buffer-agnostic
        # (DiagnosticSnapshots doesn't need flashbax at import
        # time so a missing optional dep doesn't break their import).
        from ajax.buffers.utils import get_batch_from_buffer

        obs, terminated, _, next_obs, reward, act = get_batch_from_buffer(
            buffer, agent_state.collector_state.buffer_state, rng
        )[:6]
        rows = (obs, act, reward, next_obs, terminated)
        take = None
    elif _has_last_rollout(agent_state):
        sub = agent_state.last_rollout
        done = jnp.logical_or(sub.terminated, sub.truncated)
        rows = (sub.obs, sub.action, sub.reward, sub.next_obs, done)
        take = on_policy_take
    else:
        raise ValueError(
            "ConditioningMetrics / BiasVoreDecomposition need either a replay "
            "buffer (off-policy) or agent_state.last_rollout populated via "
            "BaseAgentConfig.expose_recent_rollout=True (on-policy)."
        )
    n = math.prod(rows[0].shape[:-1])
    return tuple(x.reshape(n, -1)[:take] for x in rows)


# ---------------------------------------------------------------------------
# Finding the critic
# ---------------------------------------------------------------------------


def _network(train_state: Any) -> Any:
    """The flax module whose ``apply`` is ``train_state.apply_fn``."""
    return getattr(getattr(train_state, "apply_fn", None), "__self__", None)


def _is_ensemble_critic(train_state: Any) -> bool:
    """A :class:`~ajax.networks.networks.MultiCritic` exposes its members'
    encoder features; a Q-network does not."""
    return hasattr(_network(train_state), "apply_encoder_ensemble")


def _takes_action(
    critic_state: Any, obs: jax.Array, act: jax.Array, rng: jax.Array
) -> bool:
    """Whether an ensemble critic reads the observation and action (an
    action-value critic) or the observation alone (a state-value one).

    The answer is the input its parameters were built on: each candidate
    goes through the critic's ``init`` abstractly (shapes only, nothing is
    computed) and the one that rebuilds the critic's parameter shapes wins.
    Where the batch comes from does not say it: AVG learns an action-value
    critic from its rollout, without a replay buffer.
    """

    def shapes(tree: Any) -> list:
        return [leaf.shape for leaf in jax.tree.leaves(tree)]

    init = _network(critic_state).init
    with_action = action_value_input(obs, act.astype(obs.dtype))
    for takes_action, x in ((True, with_action), (False, obs)):
        if shapes(jax.eval_shape(init, rng, x)) == shapes(critic_state.params):
            return takes_action
    raise ValueError(
        "The critic's parameters fit neither the observation nor the"
        " observation and action of the measurement batch."
    )


def _critic_kind(
    agent_state: Any, obs: jax.Array, act: jax.Array, rng: jax.Array
) -> str:
    """The agent's critic, read from its networks: ``"q"`` or ``"v"`` for
    an ensemble critic on ``critic_state`` taking the observation and action
    or the observation alone, ``"qnet"`` for a Q-network on ``actor_state``
    (its output carries ``q_values``)."""
    critic_state = getattr(agent_state, "critic_state", None)
    if _is_ensemble_critic(critic_state):
        return "q" if _takes_action(critic_state, obs, act, rng) else "v"
    actor = agent_state.actor_state
    if hasattr(actor.apply_fn(actor.params, obs), "q_values"):
        return "qnet"
    raise ValueError(
        f"{type(agent_state).__name__} has neither an ensemble critic on"
        " critic_state nor a Q-network on actor_state to measure."
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
    if "encoder_features" not in mv.get("intermediates", {}):
        raise ValueError(
            "ConditioningMetrics found neither an ensemble critic on"
            " critic_state nor a Q-network sowing encoder_features on"
            " actor_state."
        )
    return mv["intermediates"]["encoder_features"][0][None]


@dataclass(frozen=True)
class ConditioningMetrics(Extension):
    """Plasticity diagnostics on a state batch sampled from the agent's
    visited-state distribution.

    Reads the critic's encoder features:
    * ``critic_loc='critic'`` + ``concat_action=True``  ⇒ Q(obs,act)
      ensemble critic on ``critic_state``.
    * ``critic_loc='critic'`` + ``concat_action=False`` ⇒ V(obs)
      ensemble critic on ``critic_state``.
    * ``critic_loc='actor'``  ⇒ Q-network on ``actor_state`` (DQN, PQN),
      penultimate features read via the ``encoder_features`` sow.

    ``'auto'`` (the default for both) reads them from the networks, never
    from the agent's name or the batch source: the critic branch when
    ``critic_state``'s network is an ensemble critic (it exposes
    ``apply_encoder_ensemble``), the actor branch otherwise; on the critic
    branch, the action is concatenated when the critic's parameters were
    built on the observation and action (:func:`_takes_action`).

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

    def _loc(self, agent_state: Any) -> str:
        if self.critic_loc != "auto":
            return self.critic_loc
        return "critic" if _is_ensemble_critic(agent_state.critic_state) else "actor"

    def _features(
        self, agent_state: Any, obs: jax.Array, act: jax.Array, rng: jax.Array
    ) -> jax.Array:
        if self._loc(agent_state) == "actor":
            return _encoder_features_qnet(agent_state.actor_state, obs)
        critic_state = agent_state.critic_state
        concat = self.concat_action
        if concat == "auto":
            concat = _takes_action(critic_state, obs, act, rng)
        x = action_value_input(obs, act.astype(obs.dtype)) if concat else obs
        return _encoder_features_ppo_sac(critic_state, x)

    def _critic_params(self, agent_state: Any) -> Any:
        return getattr(agent_state, f"{self._loc(agent_state)}_state").params

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> dict:
        del ext_state, ctx
        try:
            obs, act, *_ = _sample_transitions(
                agent_state, rng, self.buffer, self.on_policy_batch
            )
        except ValueError:
            # If the state-batch source isn't ready (e.g. eval fires before
            # the first rollout populates last_rollout), emit an empty
            # dict so the eval log still completes.
            return {}
        feats = self._features(agent_state, obs, act, rng)
        return dict(
            zip(
                COND_METRIC_KEYS,
                (
                    jax.vmap(srank)(feats).mean(),
                    jax.vmap(dormant_fraction)(feats).mean(),
                    jnp.linalg.norm(feats, axis=-1).mean(),
                    _global_l2(self._critic_params(agent_state)),
                ),
                strict=False,
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
            strict=False,
        )
    )


@dataclass(frozen=True)
class BiasVoreDecomposition(Extension):
    """EVarEst bias^2 / Var(residual) / MSE / |bias| decomposition.

    Ports the math from EVAREST ``loss.py`` /
    ``mechanism.make_evarest_decomposition_metrics``. The residual is
    reconstructed from the agent's *current* critic on a measurement
    batch sampled the same way :class:`ConditioningMetrics` samples
    (replay buffer for off-policy, ``last_rollout`` for on-policy).

    By default it is the critic's one-step TD residual under the current
    policy, ``prediction - (reward_scale * r + gamma (1 - d) next)``, with
    the critic read from the networks (:func:`_critic_kind`):

    * Q(s, a) ensemble: ``next = min_i Q_i(s', a') - alpha log pi(a'|s')``,
      ``a' ~ pi(s')``, ``alpha`` the agent's temperature if it has one,
      else 0;
    * V(s) ensemble: ``next = min_i V_i(s')``;
    * Q-network: ``next = max_a' Q(s', a')``.

    The bootstrap uses the critic's current parameters: whether and how an
    agent keeps a target network (SAC does, AVG and PQN do not) is its own
    business, as are GAE, Q(lambda) and average-reward targets. Pass
    ``value_target_fn`` to measure against an agent's exact target.
    """

    buffer: Any = None
    on_policy_batch: int = 256
    gamma: float = 0.99
    reward_scale: float = 1.0
    # Optional override for an agent's exact target:
    # ``(agent_state, batch_dict, rng) -> (target, q_preds)``.
    value_target_fn: Optional[Callable] = None
    name: str = "bias_vore_decomposition"

    def _td_residual(
        self, agent_state: Any, transitions: Tuple[jax.Array, ...], rng: jax.Array
    ) -> jax.Array:
        obs, act, reward, next_obs, done = transitions
        kind = _critic_kind(agent_state, obs, act, rng)
        if kind == "qnet":
            from ajax.agents.DQN.networks import predict_q

            q_state = agent_state.actor_state
            q_values = predict_q(q_state, q_state.params, obs)
            prediction = jnp.take_along_axis(q_values, act.astype(jnp.int32), axis=-1)
            next_value = predict_q(q_state, q_state.params, next_obs).max(
                -1, keepdims=True
            )
        else:
            critic_state = agent_state.critic_state

            def value(x: jax.Array) -> jax.Array:
                return predict_value(critic_state, critic_state.params, x)

            if kind == "q":
                from ajax.agents.SAC.core import sample_next_actions

                next_act, log_prob = sample_next_actions(
                    agent_state.actor_state, next_obs, rng
                )
                alpha = getattr(agent_state, "alpha", None)
                temperature = (
                    0.0 if alpha is None else jnp.exp(alpha.params["log_alpha"])
                )
                prediction = value(action_value_input(obs, act.astype(obs.dtype)))
                next_q = value(action_value_input(next_obs, next_act)).min(0)
                next_value = next_q - temperature * log_prob
            else:
                prediction = value(obs)
                next_value = value(next_obs).min(0)
        target = self.reward_scale * reward + self.gamma * (1.0 - done) * next_value
        return prediction - target

    def eval_metrics(
        self,
        agent_state: Any,
        ext_state: Any,
        rng: jax.Array,
        ctx: ExtensionContext,
    ) -> dict:
        del ext_state, ctx
        tkey, skey = jax.random.split(rng)
        try:
            transitions = _sample_transitions(
                agent_state, skey, self.buffer, self.on_policy_batch
            )
        except ValueError:
            return {}
        if self.value_target_fn is not None:
            batch_dict = {"observations": transitions[0], "actions": transitions[1]}
            target, q_preds = self.value_target_fn(agent_state, batch_dict, tkey)
            return _bias_vore_decomposition_dict(q_preds - target)
        resid = self._td_residual(agent_state, transitions, tkey)
        return _bias_vore_decomposition_dict(resid)


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
    without recomputing alpha). It rebuilds the residual the agent's MSE
    term sees from the ``critic_loss`` batch: ``observations``,
    ``targets`` and the critic (``critic_params`` and ``critic_state``,
    with ``actions`` for a Q-critic; DQN's and PQN's ``q_state`` and
    ``actions``), and raises on a batch without them rather than add 0.
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
        del agent_state, ext_state, ctx
        coeff = self._coeff()
        if coeff == 0.0:
            return jnp.asarray(0.0)

        missing = [key for key in ("observations", "targets") if key not in batch]
        if missing:
            raise ValueError(
                f"BiasVorePenalty needs {missing} in the critic_loss batch."
            )
        obs, target = batch["observations"], batch["targets"]

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

        raise ValueError(
            "BiasVorePenalty needs the critic in the critic_loss batch:"
            " critic_params and critic_state, or q_state and actions."
        )


# Public re-exports for downstream importers.
__all__ = [
    "ConditioningMetrics",
    "BiasVoreDecomposition",
    "DiagnosticSnapshots",
    "BiasVorePenalty",
    "COND_METRIC_KEYS",
    "EVAREST_DECOMP_KEYS",
    "evarest_coeff",
    "srank",
    "dormant_fraction",
]
