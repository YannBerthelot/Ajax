"""TD-MPC2 training update as pure functions (tdmpc2_spec §2, paper-era code).

Reproduces one ``TDMPC2.update()`` of ``nicklashansen/tdmpc2@5f6fade``
(``tdmpc2/tdmpc2.py:173-290``; agent algorithm identical to b67b21c, the commit
behind the paper's curves) with the version choices of
``docs/world_models/deviations.md`` §2. Single-task agent (M4) and the
multi-task trainer (M8) both import from here (lineage rule).

Every function is pure and jittable. Randomness is an explicit input: one
update consumes an :class:`UpdateNoise` drawn by :func:`draw_update_noise`
(policy noise ``eps``, the random pairs of Q heads and the dropout keys), so a
test can inject the draws recorded from the reference. Static hyperparameters
come from a :class:`~ajax.agents.TDMPC2.state.TDMPC2Config`.

Order of operations in :func:`update` (spec 2.19, 2.21), all from the state
at the start of the update unless stated:

1. TD target (:func:`td_target`): online encoder of the next observations,
   a stochastic policy sample there, the min of a random pair of *target* Q
   heads, ``y = r + gamma * Q``.
2. World-model loss and gradients (:func:`world_model_loss`).
3. Clip with the paper-era norm: the previous update's post-clip policy
   gradient counts in the world-model clip norm (:func:`clip_grad_norm`).
4. World-model Adam step (encoder learning rate scaled by 0.3).
5. Policy loss on the *pre-step* latents of step 2 with the *post-step*
   online Q (:func:`policy_loss`), RunningScale updated before dividing.
6. Policy gradients, torch-style clip, policy Adam step.
7. Target-Q EMA from the post-step online Q.

Q-ensemble dropout: at 5f6fade (and b67b21c) the ensemble's dropout is active
in *every* forward pass, including the TD target's target-Q pass. The
reference builds the ensemble with ``functorch.combine_state_for_ensemble``
(``common/layers.py:12-21``), whose functional module is held only by the
``torch.vmap`` closure, so ``model.train()`` / ``model.eval()`` never reach its
dropout layers (verified by running the reference; recorded by
``docs/world_models/parity/tdmpc2_update_fixtures.py``). The latest code's
tensordict ensemble is a registered submodule, so there the TD target is
dropout-free (tdmpc2_spec 2.3 and 2.16, which mark the PE behaviour; §0.5
item 10). Ajax follows the paper-era code: :func:`td_target` takes a dropout
key.

Multi-task conditioning (M8; the paper-era ``cfg.multitask`` branches of the
same functions, ``world_model.py:19-23, 78-148``, ``tdmpc2.py:26, 32-34,
215``). Every function takes an optional :class:`TaskContext` (``None``, the
default, is the single-task computation, unchanged):

* the task-embedding table ``[num_tasks, task_dim]`` is a world-model
  parameter, ``wm_params[TASK_EMB]`` (the reference's ``_task_emb`` is in the
  world-model Adam at the full learning rate and in its clip,
  ``tdmpc2.py:21-27, 271``), created by :func:`create_update_state`; target
  Q has none (``world_model.py:31``), so online and target Q read the same
  online table. The embedding ``e`` is looked up from the parameters inside
  each function, so the world-model loss trains it, and concatenated in the
  reference order (``[obs, e]``, ``[z, e, a]``, ``[z, e]``; tdmpc2_spec
  1.22);
* ``nn.Embedding(max_norm=1)``: every looked-up row of norm above 1 is
  rescaled in the stored table, outside autograd, at every look-up
  (:func:`renorm_task_embedding`). Within one update this binds at the
  update's first look-up (before the TD target) and at the first look-up
  after the world-model Adam step (before the policy loss); :func:`update`
  writes the renormed rows back at exactly these two points;
* the prefix action masks multiply the policy's mean, log-std and noise
  before sampling and the log-probability, which counts the valid dims
  (:func:`squashed_gaussian`, spec 1.9, 1.23);
* the discount is per task: ``gamma`` may be a per-sample array (spec 2.20);
* the policy loss reads the embedding from the stop-gradiented parameters,
  as 5f6fade's ``track_q_grad(False)`` freezes ``_task_emb`` with the Q
  heads (``world_model.py:58-68``; spec 2.18: the latest code leaks this
  gradient into the next world-model step).

:mod:`ajax.agents.TDMPC2.multitask` builds the contexts and the per-task
discounts from a task set; :mod:`ajax.agents.TDMPC2.planner` takes the same
context.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from ajax.agents.TDMPC2.networks import make_policy_prior, make_world_model
from ajax.agents.TDMPC2.state import TDMPC2Config, TDMPC2UpdateState
from ajax.distributional import TwoHot
from ajax.networks.utils import get_adam_tx
from ajax.normalizers import RunningScale
from ajax.state import LoadedTrainState
from ajax.types import FloatOrCallable

Params = Any
IntLike = Union[int, jax.Array]
ApplyFn = Callable[..., Any]

# torch.nn.utils.clip_grad_norm_ adds this to the norm before dividing.
_CLIP_EPS = 1e-6
# math.py:35-37, the tanh-squash correction log(relu(1 - a^2) + 1e-6).
_SQUASH_EPS = 1e-6
_HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))
# nn.Embedding(max_norm=1) (world_model.py:20) and the 1e-7 torch's
# embedding_renorm_ adds to the norm (tdmpc2_spec 1.22).
_EMB_MAX_NORM = 1.0
_EMB_RENORM_EPS = 1e-7
# The task-embedding table's key in the world-model parameters.
TASK_EMB = "task_emb"


def discount_from_episode_length(
    episode_length: int, denom: float = 5.0, lo: float = 0.95, hi: float = 0.995
) -> float:
    """TD-MPC2's discount heuristic ``clip((T/denom - 1) / (T/denom), lo, hi)``.

    ``tdmpc2.py:36-49`` (tdmpc2_spec 2.20; paper Table 8): T = 500 gives
    0.99, T = 100 gives 0.95, T = 1000 gives 0.995.
    """
    frac = episode_length / denom
    return float(min(max((frac - 1) / frac, lo), hi))


@struct.dataclass
class TDMPC2Batch:
    """One sampled batch, time-major (tdmpc2_spec 2.1).

    Attributes:
        obs: ``[H + 1, B, obs_dim]``, observations ``s_0 .. s_H``.
        action: ``[H, B, A]``, the buffer actions ``a_0 .. a_{H-1}``.
        reward: ``[H, B]``, ``r_t`` received after ``a_t``.
    """

    obs: jax.Array
    action: jax.Array
    reward: jax.Array


@struct.dataclass
class UpdateNoise:
    """The random draws of one update (``DESIGN.md`` §4.2 randomness seam).

    Attributes:
        td_eps: ``[H, B, A]`` standard normal policy noise of the TD target's
            action sample (``tdmpc2.py:214`` via ``world_model.py:134``).
        td_pair: ``int32[2]``, distinct target-Q heads of the TD target
            (``world_model.py:170``).
        td_dropout: dropout key of the TD target's target-Q pass (see the
            module docstring).
        value_dropout: dropout key of the value loss' online-Q pass.
        pi_eps: ``[H + 1, B, A]`` policy noise of the policy loss.
        pi_pair: ``int32[2]``, distinct online-Q heads of the policy loss.
        pi_dropout: dropout key of the policy loss' online-Q pass.
    """

    td_eps: jax.Array
    td_pair: jax.Array
    td_dropout: jax.Array
    value_dropout: jax.Array
    pi_eps: jax.Array
    pi_pair: jax.Array
    pi_dropout: jax.Array


def draw_q_pair(key: jax.Array, num_q: int) -> jax.Array:
    """Two distinct Q-head indices, uniform over ordered pairs.

    The distribution of ``np.random.choice(num_q, 2, replace=False)``
    (``world_model.py:170``); with ``num_q = 2`` both heads.
    """
    return jax.random.permutation(key, num_q)[:2].astype(jnp.int32)


def draw_update_noise(
    key: jax.Array, config: TDMPC2Config, batch_size: int, action_dim: int
) -> UpdateNoise:
    """Every random draw of one :func:`update`."""
    keys = jax.random.split(key, 7)
    horizon = config.horizon
    return UpdateNoise(
        td_eps=jax.random.normal(keys[0], (horizon, batch_size, action_dim)),
        td_pair=draw_q_pair(keys[1], config.num_q),
        td_dropout=keys[2],
        value_dropout=keys[3],
        pi_eps=jax.random.normal(keys[4], (horizon + 1, batch_size, action_dim)),
        pi_pair=draw_q_pair(keys[5], config.num_q),
        pi_dropout=keys[6],
    )


@struct.dataclass
class TaskContext:
    """The multi-task conditioning of a batch or of one decision (M8).

    Built by :meth:`ajax.agents.TDMPC2.multitask.TaskSet.context` from task
    ids (tdmpc2_spec 1.22, 1.23).

    Attributes:
        ids: int32 task ids, ``[B]`` per batch element (the reference's
            ``task[0]`` of each slice, ``buffer.py:81``) or ``[]`` for one
            planning decision; they index ``wm_params[TASK_EMB]``.
        mask: float32 prefix action masks of those tasks, ``[B, A]`` (or
            ``[A]``): 1 on each task's ``action_dims[i]`` leading dims, 0 on
            the padding (``world_model.py:21-23``).
    """

    ids: jax.Array
    mask: jax.Array


def task_embedding(wm_params: Params, task: Optional[TaskContext]) -> Any:
    """``e = W[ids]`` from the world-model parameters, or ``None`` without a
    task (single task).

    A plain gather: the gradient w.r.t. the looked-up row is the identity,
    as torch's after its in-place renorm (``world_model.py:86``). The renorm
    itself is :func:`renorm_task_embedding`, applied by the callers at the
    reference's look-up points.
    """
    if task is None:
        return None
    return wm_params[TASK_EMB][task.ids]


def renorm_task_embedding(wm_params: Params, ids: jax.Array) -> Params:
    """``nn.Embedding(max_norm=1)``'s look-up-time renorm, written back.

    ``torch.embedding_renorm_`` (tdmpc2_spec 1.22, ``world_model.py:20, 86``):
    every row of ``wm_params[TASK_EMB]`` indexed by ``ids`` whose norm
    exceeds 1 is rescaled by ``1 / (norm + 1e-7)``; other rows, looked-up
    rows inside the unit ball included, are unchanged. Returns the
    parameters with the renormed table. Apply it outside any differentiated
    function and differentiate w.r.t. its result: torch renorms in place
    under ``no_grad``, takes the gradient w.r.t. the renormed table and
    steps Adam from it (the Adam moments are not touched). Fixed-shape: one
    mask over the table's rows, so repeated ids cost nothing.
    """
    table = wm_params[TASK_EMB]
    looked_up = jnp.zeros(table.shape[0], bool).at[jnp.ravel(ids)].set(True)
    norm = jnp.linalg.norm(table, axis=-1, keepdims=True)
    renormed = table * (_EMB_MAX_NORM / (norm + _EMB_RENORM_EPS))
    rows = looked_up[:, None] & (norm > _EMB_MAX_NORM)
    return {**wm_params, TASK_EMB: jnp.where(rows, renormed, table)}


class PolicySample(NamedTuple):
    """A reparameterised policy sample (``world_model.py:122-148``).

    Attributes:
        action: ``tanh(mean + eps * exp(log_std))``, ``[..., A]``.
        log_pi: the paper-era log-probability estimate, ``[...]``.
        mean: ``tanh(mean)``, the deterministic action, ``[..., A]``.
        log_std: the log standard deviation, ``[..., A]``.
    """

    action: jax.Array
    log_pi: jax.Array
    mean: jax.Array
    log_std: jax.Array


def squashed_gaussian(
    mean: jax.Array,
    raw_log_std: jax.Array,
    eps: jax.Array,
    config: TDMPC2Config,
    mask: Optional[jax.Array] = None,
) -> PolicySample:
    """The paper-era TD-MPC2 policy sample and log-probability.

    ``world_model.py:132-148`` with ``math.py:12-45`` (tdmpc2_spec 1.8, 1.9,
    2.12; deviations.md §2, "Policy entropy bonus"):

    * ``log_std = min + 0.5 (max - min) (tanh(raw) + 1)``;
    * multi-task (``mask``, the prefix action mask broadcast against
      ``[..., A]``): ``mean``, ``log_std`` and ``eps`` are multiplied by the
      mask, and ``n`` is the number of valid dims (``world_model.py:136-140``,
      spec 1.23); single task ``n = A``;
    * ``u = mean + eps exp(log_std)``, ``action = tanh(u)``;
    * ``log_pi = n (sum_d(-eps^2 / 2 - log_std) - ln(2 pi) / 2)
      - sum_d log(relu(1 - action^2) + 1e-6)``: the Gaussian part scaled by
      ``n``, the tanh correction unscaled and differentiated. Both sums run
      over all ``A`` dims; a masked dim adds 0 to the first and
      ``log(1 + 1e-6)`` to the second, and its action is exactly 0. The
      ``1e-6`` keeps ``log_pi`` and its gradient finite when ``tanh``
      saturates to exactly +-1 in float32.
    """
    log_std = config.log_std_min + 0.5 * (config.log_std_max - config.log_std_min) * (
        jnp.tanh(raw_log_std) + 1.0
    )
    n: Any = eps.shape[-1]
    if mask is not None:
        mean, log_std, eps = mean * mask, log_std * mask, eps * mask
        n = jnp.sum(mask, axis=-1)
    residual = jnp.sum(-0.5 * jnp.square(eps) - log_std, axis=-1)
    log_pi = (residual - _HALF_LOG_2PI) * n
    action = jnp.tanh(mean + eps * jnp.exp(log_std))
    log_pi = log_pi - jnp.sum(
        jnp.log(jax.nn.relu(1.0 - jnp.square(action)) + _SQUASH_EPS), axis=-1
    )
    return PolicySample(
        action=action, log_pi=log_pi, mean=jnp.tanh(mean), log_std=log_std
    )


def policy_sample(
    pi_apply: ApplyFn,
    pi_params: Params,
    z: jax.Array,
    eps: jax.Array,
    config: TDMPC2Config,
    task_emb: Optional[jax.Array] = None,
    mask: Optional[jax.Array] = None,
) -> PolicySample:
    """Sample the policy prior at latents ``z [..., L]`` with noise ``eps [..., A]``.

    Multi-task: the task embedding ``task_emb`` enters as ``[z, e]`` and the
    action ``mask`` as in :func:`squashed_gaussian` (``world_model.py:122-148``).
    """
    mean, raw_log_std = pi_apply({"params": pi_params}, z, task_emb)
    return squashed_gaussian(mean, raw_log_std, eps, config, mask)


def q_logits(
    wm_apply: ApplyFn,
    wm_params: Params,
    z: jax.Array,
    action: jax.Array,
    *,
    q_params: Optional[Params] = None,
    dropout_key: Optional[jax.Array] = None,
    task_emb: Optional[jax.Array] = None,
) -> jax.Array:
    """All Q members' logits ``[num_q, ..., num_bins]``.

    ``q_params`` replaces the online ensemble (the target Q for TD targets);
    ``dropout_key`` enables the members' dropout, ``None`` disables it;
    ``task_emb`` is the multi-task embedding (``[z, e, a]``).
    """
    params = wm_params if q_params is None else {**wm_params, "q": q_params}
    rngs = None if dropout_key is None else {"dropout": dropout_key}
    return wm_apply(
        {"params": params},
        z,
        action,
        task_emb,
        deterministic=dropout_key is None,
        method="q_logits",
        rngs=rngs,
    )


def q_pair_logits(
    config: TDMPC2Config,
    wm_params: Params,
    z: jax.Array,
    action: jax.Array,
    pair: jax.Array,
    *,
    dropout_key: Optional[jax.Array] = None,
    task_emb: Optional[jax.Array] = None,
) -> jax.Array:
    """Logits ``[2, ..., num_bins]`` of the online Q members ``pair`` only.

    For a pass that reduces a random pair (``world_model.py:170``): the
    reference evaluates all ``num_q`` members and keeps two; this gathers the
    pair's stacked parameters and runs them as a two-member ensemble, so the
    other ``num_q - 2`` members cost nothing. Without dropout the logits equal
    ``q_logits(...)[pair]``. With ``dropout_key`` each of the two members
    draws its own mask, independent across members and calls, as in the full
    ensemble: the distribution is the reference's, only the mapping from the
    key to the masks differs from :func:`q_logits` (torch's masks are not
    replayable anyway).
    """
    members = jax.tree_util.tree_map(lambda p: p[pair], wm_params["q"])
    pair_apply = make_world_model(config.replace(num_q=2)).apply
    return q_logits(
        pair_apply,
        wm_params,
        z,
        action,
        q_params=members,
        dropout_key=dropout_key,
        task_emb=task_emb,
    )


def reduce_q_pair(
    logits: jax.Array, pair: jax.Array, kind: Literal["min", "avg"], two_hot: TwoHot
) -> jax.Array:
    """``min`` or average of the decoded Q values of heads ``pair`` (``[...]``).

    ``world_model.py:170-172``: one pair per call, shared by the whole batch
    and horizon; the reduction acts on decoded scalars, ``(Q1 + Q2) / 2``.
    """
    q1 = two_hot.decode(logits[pair[0]])
    q2 = two_hot.decode(logits[pair[1]])
    if kind == "min":
        return jnp.minimum(q1, q2)
    if kind == "avg":
        return (q1 + q2) / 2
    raise ValueError(f"kind must be 'min' or 'avg', got {kind!r}")


def _rho_weights(rho: float, n: int) -> jax.Array:
    """``rho^t`` for ``t < n``, in float64 then float32 as the reference's
    Python ``rho**t`` (``tdmpc2.py:246, 257, 259``) and ``torch.pow`` (:192)."""
    return jnp.asarray(np.power(rho, np.arange(n, dtype=np.float64)), jnp.float32)


def td_target(
    wm_apply: ApplyFn,
    pi_apply: ApplyFn,
    wm_params: Params,
    target_q_params: Params,
    pi_params: Params,
    next_obs: jax.Array,
    reward: jax.Array,
    gamma: Union[float, jax.Array],
    eps: jax.Array,
    pair: jax.Array,
    dropout_key: Optional[jax.Array],
    config: TDMPC2Config,
    task: Optional[TaskContext] = None,
) -> tuple[jax.Array, jax.Array]:
    """TD targets ``y [H, B]`` and next latents ``h(s') [H, B, L]``, no gradient.

    ``tdmpc2.py:201-216, 231-233`` (tdmpc2_spec 2.3): ``next_z = h(obs[1:])``
    with the *online* encoder; ``a' = pi(next_z)`` a stochastic sample;
    ``y = r + gamma * min(Qbar_i, Qbar_j)(next_z, a')`` with a random pair of
    target heads; no termination factor. The paper-era target-Q pass has
    dropout on (module docstring): pass ``dropout_key``; ``None`` gives the
    dropout-free target of the latest code. ``next_z`` is also the consistency
    target. ``gamma`` is a Python float, a scalar array or, multi-task, the
    per-sample discounts ``[B]`` (``discount[task]``, ``tdmpc2.py:215``);
    ``task`` conditions every network on the (online) embedding and masks
    the policy sample.
    """
    emb = task_embedding(wm_params, task)
    mask = None if task is None else task.mask
    next_z = wm_apply({"params": wm_params}, next_obs, emb, method="encode")
    action = policy_sample(pi_apply, pi_params, next_z, eps, config, emb, mask).action
    logits = q_logits(
        wm_apply,
        wm_params,
        next_z,
        action,
        q_params=target_q_params,
        dropout_key=dropout_key,
        task_emb=emb,
    )
    q = reduce_q_pair(logits, pair, "min", config.two_hot)
    y = reward + gamma * q
    return jax.lax.stop_gradient(y), jax.lax.stop_gradient(next_z)


def world_model_loss(
    wm_params: Params,
    wm_apply: ApplyFn,
    batch: TDMPC2Batch,
    next_z: jax.Array,
    td_targets: jax.Array,
    dropout_key: Optional[jax.Array],
    config: TDMPC2Config,
    task: Optional[TaskContext] = None,
) -> tuple[jax.Array, tuple[dict[str, jax.Array], jax.Array]]:
    """World-model loss, its terms and the rollout latents ``zs [H + 1, B, L]``.

    ``tdmpc2.py:239-267`` (tdmpc2_spec 2.4-2.8): ``z_0 = h(s_0)``,
    ``z_{t+1} = d(z_t, a_t)`` with the buffer actions;

    * consistency ``(1/H) sum_t rho^t mean_{B,L}((z_{t+1} - sg(h(s_{t+1})))^2)``;
    * reward ``(1/H) sum_t rho^t mean_B CE(R(z_t, a_t), r_t)``;
    * value ``(1/(H Nq)) sum_t sum_k rho^t mean_B CE(Q_k(z_t, a_t), y_t)``
      over every online head, dropout on (``dropout_key``);
    * total ``20 consistency + 0.1 reward + 0.1 value`` (config coefficients).

    ``zs`` (``z_0`` encoded, ``z_1 .. z_H`` predicted) feeds the policy loss.
    Multi-task, ``task``'s embedding is looked up from ``wm_params`` here, so
    the loss trains the table (``tdmpc2.py:241-252``); the buffer actions are
    zero on the invalid dims and are not masked (spec 1.23).
    """
    horizon = config.horizon
    rho = _rho_weights(config.rho, horizon)
    emb = task_embedding(wm_params, task)
    z = wm_apply({"params": wm_params}, batch.obs[0], emb, method="encode")
    zs = [z]
    consistency = jnp.zeros((), jnp.float32)
    for t in range(horizon):
        z = wm_apply({"params": wm_params}, z, batch.action[t], emb, method="next")
        consistency = consistency + jnp.mean(jnp.square(z - next_z[t])) * rho[t]
        zs.append(z)
    latents = jnp.stack(zs)

    two_hot = config.two_hot
    q = q_logits(
        wm_apply,
        wm_params,
        latents[:-1],
        batch.action,
        dropout_key=dropout_key,
        task_emb=emb,
    )
    r = wm_apply(
        {"params": wm_params},
        latents[:-1],
        batch.action,
        emb,
        method="reward_logits",
    )
    reward_ce = jnp.mean(two_hot.loss(r, batch.reward), axis=-1)  # [H]
    value_ce = jnp.mean(two_hot.loss(q, td_targets[None]), axis=-1)  # [Nq, H]
    consistency_loss = consistency * (1.0 / horizon)
    reward_loss = jnp.sum(reward_ce * rho) * (1.0 / horizon)
    value_loss = jnp.sum(value_ce * rho[None]) * (1.0 / (horizon * config.num_q))
    total = (
        config.consistency_coef * consistency_loss
        + config.reward_coef * reward_loss
        + config.value_coef * value_loss
    )
    terms = {
        "consistency_loss": consistency_loss,
        "reward_loss": reward_loss,
        "value_loss": value_loss,
        "total_loss": total,
    }
    return total, (terms, latents)


def policy_loss(
    pi_params: Params,
    pi_apply: ApplyFn,
    wm_apply: ApplyFn,
    wm_params: Params,
    zs: jax.Array,
    q_scale: RunningScale,
    eps: jax.Array,
    pair: jax.Array,
    dropout_key: Optional[jax.Array],
    config: TDMPC2Config,
    task: Optional[TaskContext] = None,
) -> tuple[jax.Array, tuple[RunningScale, dict[str, jax.Array]]]:
    """Paper-era policy loss and the updated RunningScale.

    ``tdmpc2.py:173-199`` (tdmpc2_spec 2.11-2.13): on the stop-gradiented
    latents ``zs [H + 1, B, L]``, ``a ~ pi(zs)`` (reparameterised),
    ``Qp = avg`` of a random pair of online heads with their parameters
    stop-gradiented (``track_q_grad(False)``), dropout on;
    the RunningScale is updated with ``Qp[0]`` *before* dividing; the loss is
    ``mean_t rho^t mean_B(beta log_pi - Qp / S)``. The gradient reaches the
    policy only, through the action and ``log_pi``. In :func:`update`,
    ``wm_params`` are the post-step parameters and ``zs`` the pre-step
    latents (spec 2.19). Multi-task, the embedding is read from the
    stop-gradiented parameters too (``track_q_grad(False)`` freezes
    ``_task_emb``, ``world_model.py:58-68``; spec 2.18) and the policy
    sample is masked; one RunningScale serves every task (spec §2.A).
    """
    zs = jax.lax.stop_gradient(zs)
    frozen_wm = jax.lax.stop_gradient(wm_params)
    emb = task_embedding(frozen_wm, task)
    mask = None if task is None else task.mask
    sample = policy_sample(pi_apply, pi_params, zs, eps, config, emb, mask)
    logits = q_logits(
        wm_apply,
        frozen_wm,
        zs,
        sample.action,
        dropout_key=dropout_key,
        task_emb=emb,
    )
    q = reduce_q_pair(logits, pair, "avg", config.two_hot)  # [H + 1, B]
    q_scale = q_scale.update(q[0])
    q = q / q_scale.scale()
    rho = _rho_weights(config.rho, zs.shape[0])
    per_step = jnp.mean(config.entropy_coef * sample.log_pi - q, axis=-1)
    loss = jnp.mean(per_step * rho)
    aux = {
        "pi_loss": loss,
        "pi_entropy": -jnp.mean(sample.log_pi),
        "pi_log_std": jnp.mean(sample.log_std),
        "pi_scale": q_scale.value,
    }
    return loss, (q_scale, aux)


def clip_grad_norm(
    grads: Params, max_norm: float, extra_sq_norm: Union[float, jax.Array] = 0.0
) -> tuple[Params, jax.Array]:
    """``torch.nn.utils.clip_grad_norm_``: clipped gradients and the pre-clip norm.

    ``norm = sqrt(||grads||^2 + extra_sq_norm)``;
    ``grads * min(1, max_norm / (norm + 1e-6))``. ``extra_sq_norm`` adds the
    squared norm of gradients that are not returned: the paper-era
    world-model clip runs over ``model.parameters()``, which include the
    policy prior's stale post-clip gradients of the previous update
    (``tdmpc2.py:184, 236, 271``; deviations.md §2).
    """
    norm = jnp.sqrt(optax.tree_utils.tree_norm(grads, squared=True) + extra_sq_norm)
    coef = jnp.minimum(1.0, max_norm / (norm + _CLIP_EPS))
    return jax.tree_util.tree_map(lambda g: g * coef, grads), norm


def _scaled_lr(learning_rate: FloatOrCallable, scale: float) -> FloatOrCallable:
    if callable(learning_rate):
        schedule = learning_rate
        return lambda count: schedule(count) * scale
    return learning_rate * scale


def make_world_model_tx(
    learning_rate: FloatOrCallable = 3e-4, enc_lr_scale: float = 0.3
) -> optax.GradientTransformation:
    """World-model Adam with the encoder's learning rate scaled (``tdmpc2.py:21-27``).

    Two parameter groups (``optax.multi_transform``): ``encoder`` at
    ``learning_rate * enc_lr_scale``, everything else (dynamics, reward, Q
    and, multi-task, the task embedding: ``tdmpc2.py:26``) at
    ``learning_rate``; torch Adam defaults
    (betas 0.9 / 0.999, eps 1e-8; tdmpc2_spec 2.9). Clipping is not part of
    the transformation: :func:`update` clips once over all world-model
    gradients, with the paper-era norm, before the groups.
    """

    def labels(params: Params) -> dict[str, str]:
        return {k: "encoder" if k == "encoder" else "rest" for k in params}

    return optax.multi_transform(
        {
            "encoder": get_adam_tx(_scaled_lr(learning_rate, enc_lr_scale), eps=1e-8),
            "rest": get_adam_tx(learning_rate, eps=1e-8),
        },
        labels,
    )


def make_policy_tx(
    learning_rate: FloatOrCallable = 3e-4, eps: float = 1e-5
) -> optax.GradientTransformation:
    """Policy Adam(eps 1e-5) (``tdmpc2.py:28``); clipped in :func:`update`."""
    return get_adam_tx(learning_rate, eps=eps)


def create_update_state(
    key: jax.Array,
    config: TDMPC2Config,
    obs_dim: int,
    action_dim: int,
    *,
    learning_rate: FloatOrCallable = 3e-4,
    enc_lr_scale: float = 0.3,
    pi_eps: float = 1e-5,
    task_dim: int = 0,
    num_tasks: int = 0,
) -> TDMPC2UpdateState:
    """Initial networks, optimizers, target Q, RunningScale and clip carry.

    The target Q starts as a copy of the online Q after its zero-initialised
    final weights (``world_model.py:29-31``); the RunningScale at 1
    (``scale.py:9``); the stale policy-gradient norm at 0. ``task_dim > 0``
    initialises the input widths for a task embedding; with ``num_tasks > 0``
    as well, the world-model parameters hold the embedding table
    ``[num_tasks, task_dim]`` under :data:`TASK_EMB`, drawn ``U(-0.02,
    0.02)`` (``init.py:10-11``) from its own key, and the world-model
    optimizer trains it at the full learning rate (``tdmpc2.py:26``). Target
    Q holds no copy (``world_model.py:31``). Multi-task: ``obs_dim`` and
    ``action_dim`` are the padded maxima.
    """
    if num_tasks < 0 or task_dim < 0 or (num_tasks and not task_dim):
        raise ValueError(
            f"need num_tasks >= 0 and task_dim >= 0, and task_dim > 0 for a"
            f" task-embedding table, got num_tasks={num_tasks},"
            f" task_dim={task_dim}"
        )
    world_model = make_world_model(config)
    policy = make_policy_prior(config, action_dim)
    wm_key, pi_key = jax.random.split(key)
    emb = jnp.zeros((1, task_dim)) if task_dim else None
    wm_params = world_model.init(
        wm_key, jnp.zeros((1, obs_dim)), jnp.zeros((1, action_dim)), emb
    )["params"]
    pi_params = policy.init(pi_key, jnp.zeros((1, config.latent_dim)), emb)["params"]
    if num_tasks:
        # A key of its own, so the single-task keys above are unchanged.
        table = jax.random.uniform(
            jax.random.fold_in(key, 1),
            (num_tasks, task_dim),
            minval=-0.02,
            maxval=0.02,
        )
        wm_params = {**wm_params, TASK_EMB: table}
    world_model_state = LoadedTrainState.create(
        apply_fn=world_model.apply,
        params=wm_params,
        tx=make_world_model_tx(learning_rate, enc_lr_scale),
        target_params=wm_params["q"],
    )
    actor_state = LoadedTrainState.create(
        apply_fn=policy.apply,
        params=pi_params,
        tx=make_policy_tx(learning_rate, pi_eps),
    )
    return TDMPC2UpdateState(
        world_model_state=world_model_state,
        actor_state=actor_state,
        q_scale=RunningScale.create(rate=config.tau),
        pi_gradnorm_sq=jnp.zeros((), jnp.float32),
    )


class UpdateStateLike(Protocol):
    """Any state carrying the four :class:`TDMPC2UpdateState` attributes."""

    world_model_state: LoadedTrainState
    actor_state: LoadedTrainState
    q_scale: RunningScale
    pi_gradnorm_sq: jax.Array

    def replace(self, **updates: Any) -> Any: ...


S = TypeVar("S", bound=UpdateStateLike)


def update(
    state: S,
    batch: TDMPC2Batch,
    noise: UpdateNoise,
    *,
    config: TDMPC2Config,
    gamma: Union[float, jax.Array],
    task: Optional[TaskContext] = None,
) -> tuple[S, dict[str, jax.Array]]:
    """One paper-era TD-MPC2 update (``tdmpc2.py:218-290``); see the module docstring.

    Args:
        state: a :class:`TDMPC2UpdateState`, or any state with its four
            attributes and ``replace`` (the M4 agent state).
        batch: ``obs [H + 1, B, S]``, ``action [H, B, A]``, ``reward [H, B]``.
        noise: the update's random draws (:func:`draw_update_noise`).
        config: static hyperparameters (a jit static argument).
        gamma: the discount, a Python float or a scalar array; multi-task,
            the per-sample discounts ``[B]``.
        task: multi-task, the batch's :class:`TaskContext` (``None``: single
            task). The embedding rows the batch looks up are renormed and
            written back twice (``nn.Embedding(max_norm=1)``, module
            docstring): before the TD target, where the reference's first
            look-up is ``encode(obs[1:], task)`` (``tdmpc2.py:232``), and
            after the world-model Adam step, where it is the policy loss'
            ``pi(zs, task)`` (``tdmpc2.py:186``).

    Returns:
        The new state and the reference's logged quantities
        (``tdmpc2.py:282-290``: ``consistency_loss``, ``reward_loss``,
        ``value_loss``, ``pi_loss``, ``total_loss``, ``grad_norm`` (the
        world-model clip norm, stale policy gradients included), ``pi_scale``)
        plus ``pi_grad_norm`` (pre-clip), ``pi_entropy`` (``-mean log_pi``) and
        ``pi_log_std``.
    """
    wm_state = state.world_model_state
    pi_state = state.actor_state
    wm_apply, pi_apply = wm_state.apply_fn, pi_state.apply_fn
    if task is not None:  # pre-step renorm, written back
        wm_state = wm_state.replace(
            params=renorm_task_embedding(wm_state.params, task.ids)
        )

    td, next_z = td_target(
        wm_apply,
        pi_apply,
        wm_state.params,
        wm_state.target_params,
        pi_state.params,
        batch.obs[1:],
        batch.reward,
        gamma,
        noise.td_eps,
        noise.td_pair,
        noise.td_dropout,
        config,
        task,
    )

    (_, (wm_terms, zs)), wm_grads = jax.value_and_grad(world_model_loss, has_aux=True)(
        wm_state.params,
        wm_apply,
        batch,
        next_z,
        td,
        noise.value_dropout,
        config,
        task,
    )
    wm_grads, grad_norm = clip_grad_norm(
        wm_grads, config.grad_clip_norm, extra_sq_norm=state.pi_gradnorm_sq
    )
    wm_state = wm_state.apply_gradients(grads=wm_grads)
    if task is not None:  # post-step renorm, written back
        wm_state = wm_state.replace(
            params=renorm_task_embedding(wm_state.params, task.ids)
        )

    (_, (q_scale, pi_aux)), pi_grads = jax.value_and_grad(policy_loss, has_aux=True)(
        pi_state.params,
        pi_apply,
        wm_apply,
        wm_state.params,
        zs,
        state.q_scale,
        noise.pi_eps,
        noise.pi_pair,
        noise.pi_dropout,
        config,
        task,
    )
    pi_grads, pi_grad_norm = clip_grad_norm(pi_grads, config.grad_clip_norm)
    # Post-clip, carried to the next update's world-model clip.
    pi_gradnorm_sq = optax.tree_utils.tree_norm(pi_grads, squared=True)
    pi_state = pi_state.apply_gradients(grads=pi_grads)

    wm_state = wm_state.replace(
        target_params=optax.incremental_update(
            wm_state.params["q"], wm_state.target_params, config.tau
        )
    )
    new_state = state.replace(
        world_model_state=wm_state,
        actor_state=pi_state,
        q_scale=q_scale,
        pi_gradnorm_sq=pi_gradnorm_sq,
    )
    aux = {
        **wm_terms,
        **pi_aux,
        "grad_norm": grad_norm,
        "pi_grad_norm": pi_grad_norm,
    }
    return new_state, aux
