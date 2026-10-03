"""TD-MPC2 MPPI planner as pure functions (tdmpc2_spec §3, paper-era code).

Reproduces ``TDMPC2.act()`` with ``mpc = true`` of
``nicklashansen/tdmpc2@5f6fade`` (``tdmpc2/tdmpc2.py:70-171``: ``act``,
``_estimate_value`` and ``plan``; the agent algorithm is identical to b67b21c,
the commit behind the paper's curves) for one environment. Every function is
pure, jittable and vmappable over a leading environment axis (per-env
observation, ``t0``, warm-start mean and noise); all shapes are static.

One decision (:func:`plan`, spec 3.3-3.21; ``DESIGN.md`` §4.3):

1. ``z = h(obs)``; ``num_pi_trajs`` (24) policy-prior trajectories, each a
   *stochastic* prior sample at every step rolled through the deterministic
   dynamics (``H - 1`` dynamics steps, ``H`` samples; ``tdmpc2.py:119-126``).
   They fill candidate columns ``[:24]`` and are never resampled, but they are
   re-scored every iteration.
2. ``mean = 0``, except ``mean[:-1] = prev_mean[1:]`` when not ``t0`` (the
   last row stays 0); ``std = max_std``: only the mean is warm-started
   (``tdmpc2.py:129-133``).
3. ``config.planning_iterations(A)`` MPPI iterations (6, +2 when ``A >= 20``;
   ``lax.scan`` over the per-iteration draws, the horizon unrolled in Python):
   columns ``[24:]`` sampled as ``clip(mean + std * eps, -1, 1)``, clamped
   *before* evaluation; value :func:`estimate_value`, ``nan_to_num``; top-k
   elites; ``score = softmax(temperature * V_elite)`` written as the reference
   writes it; weighted mean and biased weighted std around the new mean,
   std clamped to ``[min_std, max_std]`` (:func:`mppi_step`,
   ``tdmpc2.py:139-159``).
4. An elite drawn from ``Categorical(score)`` of the last iteration
   (:func:`sample_elite`); the action is its first action, plus
   ``std[0] * eps`` unless ``eval_mode``, clipped to ``[-1, 1]``. The new warm
   start is the last iteration's mean (``tdmpc2.py:164-171``).

Paper-era specifics (deviations.md §2): no RunningScale, no target network and
no termination masking in planning; the Q ensemble's dropout is active in the
terminal Q pass (5f6fade's ``combine_state_for_ensemble`` ensemble ignores
``eval()``, spec §0.5 item 10), so every iteration takes a dropout key.

Randomness is an explicit input (``DESIGN.md`` §4.2): one decision consumes a
:class:`PlanNoise` drawn by :func:`draw_plan_noise`, so a test can inject the
draws recorded from the reference
(``docs/world_models/parity/tdmpc2_plan_fixtures.py``). The elite draw is one
uniform number turned into an index by the inverse CDF, the algorithm of the
reference's ``np.random.choice(p=score)``.

Single-task only. Multi-task (M8) adds the task embedding to every network
call, the action masks on the candidates and on mean / std (spec 3.9) and a
per-task discount.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Union

import jax
import jax.numpy as jnp
from flax import struct

from ajax.agents.TDMPC2.core import draw_q_pair, policy_sample, q_pair_logits
from ajax.agents.TDMPC2.networks import make_policy_prior, make_world_model
from ajax.agents.TDMPC2.state import TDMPC2Config

Params = Any

# tdmpc2.py:157-158: added to the score sum in the mean and std denominators.
_SCORE_EPS = 1e-9


@struct.dataclass
class PlanNoise:
    """The random draws of one decision (``DESIGN.md`` §4.2 randomness seam).

    ``H`` is the horizon, ``P = num_pi_trajs``, ``N = num_samples``,
    ``I = config.planning_iterations(A)``. In the reference's draw order:

    Attributes:
        pi_eps: ``[H, P, A]`` standard normal noise of the policy-prior
            trajectories' samples (``tdmpc2.py:124, 126`` via
            ``world_model.py:134``).
        candidate_eps: ``[I, H, N - P, A]`` standard normal noise of the
            Gaussian candidates of each iteration (``tdmpc2.py:142-144``).
        terminal_eps: ``[I, N, A]`` noise of each iteration's terminal policy
            sample in the value estimate (``tdmpc2.py:103``).
        q_pair: ``int32[I, 2]``, each iteration's distinct pair of online Q
            heads (``world_model.py:170``).
        q_dropout: ``[I, 2]`` dropout keys of each iteration's Q pass.
        elite_uniform: ``[]`` uniform in ``[0, 1)``, the elite draw
            (``tdmpc2.py:166``; see :func:`sample_elite`).
        action_eps: ``[A]`` standard normal exploration noise
            (``tdmpc2.py:170``); unused in ``eval_mode``.
    """

    pi_eps: jax.Array
    candidate_eps: jax.Array
    terminal_eps: jax.Array
    q_pair: jax.Array
    q_dropout: jax.Array
    elite_uniform: jax.Array
    action_eps: jax.Array


@struct.dataclass
class PlanInfo:
    """What one decision computed, for tests and diagnostics.

    ``E = num_elites``; per-iteration fields have a leading ``[I]`` axis and
    hold the iteration's statistics after its update.

    Attributes:
        init_mean: ``[H, A]`` warm-started mean before the first iteration.
        pi_actions: ``[H, P, A]`` policy-prior trajectories (columns ``[:P]``).
        value: ``[I, N]`` candidate values after ``nan_to_num``.
        elite_idx: ``int32[I, E]`` candidate indices of the elites, by
            decreasing value.
        elite_value: ``[I, E]`` their values.
        score: ``[I, E]`` normalised MPPI scores.
        mean: ``[I, H, A]`` mean after each iteration; ``mean[-1]`` is the new
            warm start.
        std: ``[I, H, A]`` clamped std after each iteration.
        elite_rank: ``int32[]`` rank (in the last iteration's elite order) of
            the elite whose first action is executed.
    """

    init_mean: jax.Array
    pi_actions: jax.Array
    value: jax.Array
    elite_idx: jax.Array
    elite_value: jax.Array
    score: jax.Array
    mean: jax.Array
    std: jax.Array
    elite_rank: jax.Array


def draw_plan_noise(key: jax.Array, config: TDMPC2Config, action_dim: int) -> PlanNoise:
    """Every random draw of one :func:`plan` decision for ``action_dim`` actions.

    Vmap it over keys for several environments. The Q pairs are uniform over
    ordered distinct pairs (:func:`ajax.agents.TDMPC2.core.draw_q_pair`, the
    distribution of the reference's ``np.random.choice(num_q, 2,
    replace=False)``).
    """
    horizon, n, p = config.horizon, config.num_samples, config.num_pi_trajs
    iterations = config.planning_iterations(action_dim)
    keys = jax.random.split(key, 7)
    return PlanNoise(
        pi_eps=jax.random.normal(keys[0], (horizon, p, action_dim)),
        candidate_eps=jax.random.normal(
            keys[1], (iterations, horizon, n - p, action_dim)
        ),
        terminal_eps=jax.random.normal(keys[2], (iterations, n, action_dim)),
        q_pair=jax.vmap(lambda k: draw_q_pair(k, config.num_q))(
            jax.random.split(keys[3], iterations)
        ),
        q_dropout=jax.random.split(keys[4], iterations),
        elite_uniform=jax.random.uniform(keys[5], ()),
        action_eps=jax.random.normal(keys[6], (action_dim,)),
    )


def _discount_powers(gamma: Union[float, jax.Array], horizon: int) -> list[Any]:
    """``gamma^t`` for ``t = 0 .. horizon`` by repeated multiplication.

    The reference multiplies a Python float (``discount *= self.discount``,
    ``tdmpc2.py:97-102``), i.e. in float64, rounded to float32 when it scales
    a tensor. The products are in ``gamma``'s own precision: a Python float
    reproduces the reference, a float32 array (traced under ``jit``)
    multiplies in float32.
    """
    powers: list[Any] = [1.0]
    for _ in range(horizon):
        powers.append(powers[-1] * gamma)
    return powers


def estimate_value(
    wm_params: Params,
    pi_params: Params,
    z: jax.Array,
    actions: jax.Array,
    terminal_eps: jax.Array,
    q_pair: jax.Array,
    q_dropout: jax.Array,
    *,
    config: TDMPC2Config,
    gamma: Union[float, jax.Array],
) -> jax.Array:
    """Values ``[N]`` of the action sequences ``actions [H, N, A]`` from ``z [N, L]``.

    ``_estimate_value`` (``tdmpc2.py:94-103``; tdmpc2_spec 3.10, 3.13, 3.14):
    ``G = sum_t gamma^t R(z_t, a_t)`` with ``z_{t+1} = d(z_t, a_t)``, each
    reward decoded from its two-hot logits; then ``a_H`` a *stochastic*
    policy sample at ``z_H`` (noise ``terminal_eps [N, A]``) and
    ``G + gamma^H (Q_i + Q_j) / 2`` over the online heads ``q_pair``, each
    decoded before averaging, the pair shared by all candidates. Only the two
    drawn heads are evaluated (:func:`~ajax.agents.TDMPC2.core.q_pair_logits`;
    the reference runs all ``num_q`` and keeps two). Q dropout is on (paper
    era; key ``q_dropout``, a no-op when ``config.dropout == 0``). No target
    network, no RunningScale, no termination masking (paper era).
    """
    action_dim = actions.shape[-1]
    wm_apply = make_world_model(config).apply
    pi_apply = make_policy_prior(config, action_dim).apply
    two_hot = config.two_hot
    discount = _discount_powers(gamma, config.horizon)
    value: Any = 0.0  # G = 0, then G += gamma^t r_t (tdmpc2.py:97-101)
    for t in range(config.horizon):
        reward = two_hot.decode(
            wm_apply({"params": wm_params}, z, actions[t], method="reward_logits")
        )
        z = wm_apply({"params": wm_params}, z, actions[t], method="next")
        value = value + discount[t] * reward
    terminal = policy_sample(pi_apply, pi_params, z, terminal_eps, config).action
    logits = q_pair_logits(
        config, wm_params, z, terminal, q_pair, dropout_key=q_dropout
    )
    q1, q2 = two_hot.decode(logits)  # world_model.py:171-172
    return value + discount[config.horizon] * ((q1 + q2) / 2)


def policy_trajectories(
    wm_params: Params,
    pi_params: Params,
    z: jax.Array,
    eps: jax.Array,
    *,
    config: TDMPC2Config,
) -> jax.Array:
    """The policy-prior candidates ``[H, P, A]`` from the latent ``z [L]``.

    ``tdmpc2.py:119-126`` (tdmpc2_spec 3.5, 3.6): ``P`` copies of ``z``; for
    ``t < H - 1`` a stochastic prior sample (noise ``eps[t] [P, A]``) and a
    dynamics step; then the last sample at ``z_{H-1}``.
    """
    horizon, p, action_dim = eps.shape
    if p == 0:  # the "planning without policy" ablation (spec 3.4)
        return jnp.zeros((horizon, 0, action_dim), jnp.float32)
    wm_apply = make_world_model(config).apply
    pi_apply = make_policy_prior(config, action_dim).apply
    z = jnp.broadcast_to(z, (p, z.shape[-1]))
    actions = []
    for t in range(horizon):
        action = policy_sample(pi_apply, pi_params, z, eps[t], config).action
        actions.append(action)
        if t < horizon - 1:
            z = wm_apply({"params": wm_params}, z, action, method="next")
    return jnp.stack(actions)


def warm_start_mean(prev_mean: jax.Array, t0: Union[bool, jax.Array]) -> jax.Array:
    """``[H, A]`` initial mean: zeros if ``t0``, else ``prev_mean`` shifted by one.

    ``tdmpc2.py:130-133`` (tdmpc2_spec 3.7): ``mean[:-1] = prev_mean[1:]``,
    the last row 0. At ``t0`` the previous mean is ignored (not cleared).
    """
    shifted = jnp.concatenate([prev_mean[1:], jnp.zeros_like(prev_mean[:1])], axis=0)
    return jnp.where(t0, jnp.zeros_like(prev_mean), shifted)


class MPPIStep(NamedTuple):
    """One MPPI iteration's statistics (:func:`mppi_step`).

    Attributes:
        value: ``[N]`` candidate values after ``nan_to_num``.
        elite_idx: ``int32[E]`` elite candidate indices, by decreasing value.
        elite_value: ``[E]`` their values.
        elite_actions: ``[H, E, A]`` their action sequences.
        score: ``[E]`` normalised scores.
        mean: ``[H, A]`` the new mean.
        std: ``[H, A]`` the new std, clamped.
    """

    value: jax.Array
    elite_idx: jax.Array
    elite_value: jax.Array
    elite_actions: jax.Array
    score: jax.Array
    mean: jax.Array
    std: jax.Array


def mppi_step(value: jax.Array, actions: jax.Array, config: TDMPC2Config) -> MPPIStep:
    """One MPPI update from candidate values ``[N]`` and actions ``[H, N, A]``.

    ``tdmpc2.py:149-159`` (tdmpc2_spec 3.15-3.17):

    * ``value = nan_to_num(value, 0)``: NaN to 0 and +-inf to the largest /
      smallest finite float32 (torch's ``nan_to_num_`` defaults, which
      ``jnp.nan_to_num`` shares);
    * the ``num_elites`` best candidates by ``lax.top_k``, by decreasing value.
      Ties: ``lax.top_k`` puts the lower index first. torch's CPU ``topk``
      does not: when tied values straddle the top-k boundary it selects a
      different *subset* of them (not only a different order). Ajax
      reproduces torch only for a fully tied vector, as at a zero-initialised
      model (spec 3.23), where both return the first indices in order
      (recorded by the plan parity fixture). Other ties arise only from
      degenerate values (NaN -> 0, +-inf, saturated decodes), where either
      choice is arbitrary;
    * ``score = exp(temperature (V - max V))``, normalised to sum 1;
    * ``mean = sum_k score_k a_k / (sum score + 1e-9)`` and the biased
      weighted ``std`` around the new mean with the same denominator,
      clamped to ``[min_std, max_std]``.
    """
    value = jnp.nan_to_num(value, nan=0.0)
    elite_value, elite_idx = jax.lax.top_k(value, config.num_elites)
    elite_actions = actions[:, elite_idx]  # [H, E, A]
    score = jnp.exp(config.temperature * (elite_value - jnp.max(elite_value)))
    score = score / jnp.sum(score)
    weight = score[None, :, None]
    denom = jnp.sum(score) + _SCORE_EPS
    mean = jnp.sum(weight * elite_actions, axis=1) / denom
    var = jnp.sum(weight * jnp.square(elite_actions - mean[:, None]), axis=1) / denom
    std = jnp.clip(jnp.sqrt(var), config.min_std, config.max_std)
    return MPPIStep(
        value=value,
        elite_idx=elite_idx,
        elite_value=elite_value,
        elite_actions=elite_actions,
        score=score,
        mean=mean,
        std=std,
    )


def sample_elite(score: jax.Array, uniform: jax.Array) -> jax.Array:
    """Index ``k ~ Categorical(score)`` from one ``uniform`` in ``[0, 1)``.

    The algorithm of the reference's ``np.random.choice(arange(E), p=score)``
    (``tdmpc2.py:166``; numpy ``mtrand.pyx``, ``RandomState.choice``): the
    cumulative sum of ``score`` normalised by its last entry, then the first
    index whose cumulative probability exceeds the uniform
    (``searchsorted(side='right')``). numpy computes the CDF in float64,
    Ajax in float32. A zero score is never drawn. The latest code's Gumbel
    max has the same distribution (tdmpc2_spec 3.19); the inverse CDF lets a
    test replay the reference's own uniform.
    """
    cdf = jnp.cumsum(score)
    cdf = cdf / cdf[-1]
    return jnp.searchsorted(cdf, uniform, side="right").astype(jnp.int32)


def check_plan_noise(noise: PlanNoise, config: TDMPC2Config, action_dim: int) -> None:
    """Raise ``ValueError`` unless ``noise`` has the shapes of one decision of
    ``config`` with ``action_dim`` actions (:func:`draw_plan_noise`).

    In particular it must hold ``config.planning_iterations(action_dim)``
    rows of per-iteration draws: :func:`plan` runs one MPPI iteration per row,
    so noise drawn for another action dimension would otherwise run the wrong
    number of iterations without error (the +2 rule, spec 3.1). The shapes
    are static, so the check costs nothing under ``jit`` and sees one
    environment's shapes under ``vmap``.
    """
    h, n, p = config.horizon, config.num_samples, config.num_pi_trajs
    iterations = config.planning_iterations(action_dim)
    expected = {
        "pi_eps": (h, p, action_dim),
        "candidate_eps": (iterations, h, n - p, action_dim),
        "terminal_eps": (iterations, n, action_dim),
        "q_pair": (iterations, 2),
        "q_dropout": (iterations,),  # leading axis of raw or typed keys
        "elite_uniform": (),
        "action_eps": (action_dim,),
    }
    shapes = {name: getattr(noise, name).shape for name in expected}
    shapes["q_dropout"] = shapes["q_dropout"][:1]
    if shapes != expected:
        raise ValueError(
            f"noise does not match the config with action_dim={action_dim} "
            f"({iterations} iterations): shapes {shapes}, expected {expected}"
        )


def plan_from_latent(
    wm_params: Params,
    pi_params: Params,
    z: jax.Array,
    prev_mean: jax.Array,
    t0: Union[bool, jax.Array],
    noise: PlanNoise,
    *,
    config: TDMPC2Config,
    gamma: Union[float, jax.Array],
    eval_mode: bool,
) -> tuple[jax.Array, jax.Array, PlanInfo]:
    """MPPI from the latent ``z [L]`` (the reference's ``plan(z)``,
    ``tdmpc2.py:105-171``); see :func:`plan`."""
    horizon, action_dim = prev_mean.shape
    if horizon != config.horizon:
        raise ValueError(
            f"prev_mean has horizon {horizon}, config.horizon is {config.horizon}"
        )
    check_plan_noise(noise, config, action_dim)
    pi_actions = policy_trajectories(
        wm_params, pi_params, z, noise.pi_eps, config=config
    )
    zs = jnp.broadcast_to(z, (config.num_samples, z.shape[-1]))
    init_mean = warm_start_mean(prev_mean, t0)
    init_std = jnp.full_like(prev_mean, config.max_std)

    def iteration(
        carry: tuple[jax.Array, jax.Array], draws: Any
    ) -> tuple[tuple[jax.Array, jax.Array], MPPIStep]:
        mean, std = carry
        candidate_eps, terminal_eps, q_pair, q_dropout = draws
        sampled = jnp.clip(mean[:, None] + std[:, None] * candidate_eps, -1.0, 1.0)
        actions = jnp.concatenate([pi_actions, sampled], axis=1)  # [H, N, A]
        value = estimate_value(
            wm_params,
            pi_params,
            zs,
            actions,
            terminal_eps,
            q_pair,
            q_dropout,
            config=config,
            gamma=gamma,
        )
        step = mppi_step(value, actions, config)
        return (step.mean, step.std), step

    draws = (noise.candidate_eps, noise.terminal_eps, noise.q_pair, noise.q_dropout)
    (mean, std), stats = jax.lax.scan(iteration, (init_mean, init_std), draws)

    # tdmpc2.py:164-171: the first action of an elite of the last iteration
    # drawn from its score, plus exploration noise outside eval_mode.
    rank = sample_elite(stats.score[-1], noise.elite_uniform)
    action = stats.elite_actions[-1, 0, rank]
    if not eval_mode:
        action = action + std[0] * noise.action_eps
    action = jnp.clip(action, -1.0, 1.0)
    info = PlanInfo(
        init_mean=init_mean,
        pi_actions=pi_actions,
        value=stats.value,
        elite_idx=stats.elite_idx,
        elite_value=stats.elite_value,
        score=stats.score,
        mean=stats.mean,
        std=stats.std,
        elite_rank=rank,
    )
    return action, mean, info


def plan(
    wm_params: Params,
    pi_params: Params,
    obs: jax.Array,
    prev_mean: jax.Array,
    t0: Union[bool, jax.Array],
    noise: PlanNoise,
    *,
    config: TDMPC2Config,
    gamma: Union[float, jax.Array],
    eval_mode: bool,
) -> tuple[jax.Array, jax.Array, PlanInfo]:
    """One paper-era TD-MPC2 MPPI decision for one environment.

    ``TDMPC2.act(obs, t0, eval_mode)`` with ``mpc = true``
    (``5f6fade:tdmpc2/tdmpc2.py:70-171``; tdmpc2_spec §3; module docstring).
    Vmap it over a leading environment axis of ``obs``, ``prev_mean``,
    ``t0`` and ``noise``.

    Args:
        wm_params: world-model parameters (``encoder``, ``dynamics``,
            ``reward``, ``q``: the *online* Q ensemble).
        pi_params: policy-prior parameters.
        obs: ``[obs_dim]`` observation.
        prev_mean: ``[H, A]`` the previous decision's returned mean.
        t0: first step of an episode (a bool or a bool array): the warm start
            is ignored.
        noise: the decision's draws (:func:`draw_plan_noise`).
        config: static hyperparameters; the number of iterations is
            ``config.planning_iterations(A)``, and ``noise`` must have been
            drawn for it (:func:`check_plan_noise`, else ``ValueError``).
        gamma: the discount, a Python float or a scalar array.
        eval_mode: static; True drops only the final exploration noise.
            Everything else stays stochastic (spec 3.20).

    Returns:
        ``(action [A], new_prev_mean [H, A], info)``: the executed action in
        ``[-1, 1]``, the last iteration's mean (stored before the exploration
        noise, in both modes; spec 3.21) and a :class:`PlanInfo`.
    """
    wm_apply = make_world_model(config).apply
    z = wm_apply({"params": wm_params}, obs[None], method="encode")[0]
    return plan_from_latent(
        wm_params,
        pi_params,
        z,
        prev_mean,
        t0,
        noise,
        config=config,
        gamma=gamma,
        eval_mode=eval_mode,
    )
