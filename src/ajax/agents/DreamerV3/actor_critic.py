"""DreamerV3 actor-critic learning in imagination, and the replay critic.

Algorithms F, G and H of ``docs/world_models/dreamerv3_spec.md`` (sections 3
and 4.1), as the paper-era code ``danijar/dreamerv3@2411f7d`` computes them
in ``Agent.loss`` (the fidelity target, ``docs/world_models/deviations.md``
section 1). The cited line numbers are those of ``29eb964``, the upstream
fix Ajax adopts, whose ``Agent.loss`` computes the same thing (``1a532a7``
only removed 2411f7d's inert ``scale_by_actent`` branch): line ``n`` of its
``agent.py`` is line ``n + 16`` of 2411f7d up to 318 and ``n + 21`` from 319
on.

* :func:`imagine` rolls the world model forward from every posterior state
  of the replay batch with actions sampled by the current actor (Algorithm C
  inside ``imgstep``, ``agent.py:251-282``).
* :func:`lambda_return` is the recursion both returns use (Algorithm H,
  ``agent.py:303-310`` and ``:339-345``).
* :func:`imagination_loss` gives the actor's REINFORCE loss and the critic's
  two-hot loss on the imagined trajectories, with the return normaliser
  updated then read (Algorithm F, ``agent.py:284-329``).
* :func:`replay_critic_loss` trains the critic on the replayed trajectories,
  bootstrapped from the imagination returns (Algorithm G, ``agent.py:331-352``).

The 2411f7d choices where the later code differs (deviations.md section 1):
the start state's reward and continuation are the replayed ``reward_t`` and
``1 - is_terminal_t``; the discrete actor's log-probability and entropy are
those of its 1 %-mixed distribution (:class:`~ajax.agents.DreamerV3.
distributions.OneHotPolicy`). With the reference's ``slowtar: False``,
``contdisc: True``, ``ac_grads: none``, ``replay_critic_grad: True``,
``replay_critic_bootstrap: imag`` and ``valnorm`` / ``advnorm`` off
(``configs.yaml:102-107``, ``:124``, ``:136-146``), the returns bootstrap
from the online critic, the imagination return has no explicit discount,
every actor and critic input in imagination is stop-gradiented, and the
replay critic trains the world model through its input features.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Union

import jax
import jax.numpy as jnp

from ajax.agents.DreamerV3.distributions import Policy
from ajax.agents.DreamerV3.networks import RSSM, RSSMState, features
from ajax.agents.DreamerV3.state import DreamerV3Config
from ajax.distributional import TwoHot
from ajax.normalizers import ReturnNormalizer

sg = jax.lax.stop_gradient


def lambda_return(
    reward: jax.Array,
    live: jax.Array,
    cont: Union[jax.Array, float],
    boot: jax.Array,
) -> jax.Array:
    """The lambda-return ``R_0 .. R_{L-2}`` of ``[N, L]`` sequences (Algorithm H).

    ``R_{L-1} = boot_{L-1}`` and, going backwards,

        R_t = reward_{t+1} + live_{t+1} ((1 - cont_{t+1}) boot_{t+1}
                                        + cont_{t+1} R_{t+1}),

    computed as the reference does, ``interm_t + live_{t+1} cont_{t+1}
    R_{t+1}`` with ``interm_t = reward_{t+1} + (1 - cont_{t+1}) live_{t+1}
    boot_{t+1}`` (``29eb964:dreamerv3/agent.py:303-310``, ``:339-345``). Step
    ``t`` reads the reward, continuation and bootstrap of the **next** state
    (dreamerv3_spec 3.7); ``reward_0`` is unused.

    Args:
        reward: ``[N, L]``.
        live: ``[N, L]`` the discounted continuation: ``gamma (1 -
            is_terminal)`` on replayed data, the continue head's probability
            on imagined states (its target folds ``gamma`` in, ``contdisc``).
        cont: ``[N, L]`` the lambda continuation ``lam (1 - is_last)`` of
            replayed data, or the scalar ``lam`` (imagination: no episode
            ends). A Python float keeps the reference's rounding of
            ``1 - lam``.
        boot: ``[N, L]`` the values bootstrapped from.

    Returns:
        ``[N, L - 1]`` float32.
    """
    live = live[:, 1:]
    if not isinstance(cont, (int, float)):
        cont = cont[:, 1:]
    interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
    factor = live * cont

    def step(ret: jax.Array, inputs: tuple[jax.Array, jax.Array]):
        interm_t, factor_t = inputs
        ret = interm_t + factor_t * ret
        return ret, ret

    _, rets = jax.lax.scan(step, boot[:, -1], (interm.T, factor.T), reverse=True)
    return rets.T


class Trajectory(NamedTuple):
    """Imagined states and actions, ``[N, H, ...]`` (steps 1..H of a rollout).

    Attributes:
        deter: ``[N, H, D]``.
        stoch: ``[N, H, S, C]`` the one-hot prior samples, stop-gradiented.
        action: ``[N, H, A]`` the action sampled at each state (the last one
            only completes the shape: no loss reads it).
    """

    deter: jax.Array
    stoch: jax.Array
    action: jax.Array


def imagine(
    rssm: RSSM,
    rssm_params: dict,
    policy: Callable[[jax.Array], Policy],
    start: RSSMState,
    start_action: jax.Array,
    prior_noise: jax.Array,
    action_noise: jax.Array,
) -> Trajectory:
    """Roll out ``H`` imagined steps from ``N`` start states (``imgstep``).

    From the stop-gradiented start ``(h_0, z_0)`` and action ``a_0``, each
    step is :meth:`RSSM.imagine_step` (``h_i = f(h_{i-1}, z_{i-1},
    a_{i-1})``, ``z_i`` sampled from the mixed prior with ``prior_noise``)
    followed by ``a_i ~ pi(concat(h_i, sg(z_i)))`` with ``action_noise``
    (``29eb964:dreamerv3/agent.py:252-257``, ``:276-278``): the stop-gradients
    sit where the reference puts them, on the start carry and on the sampled
    latent read by the policy (dreamerv3_spec 3.3-3.4). Discrete actions are
    the policy's straight-through one-hot samples, continuous ones its raw
    (unclipped) samples; the dynamics bound both (:meth:`RSSM.core`).

    Args:
        rssm: the world model's RSSM module; ``rssm_params`` its parameters.
        policy: the actor on its parameters, ``feat [N, F] -> Policy``.
        start: ``[N, ...]`` start states; ``start_action`` ``[N, A]``.
        prior_noise: ``[N, H, S, C]`` Gumbel noise of the prior samples.
        action_noise: ``[N, H, A]`` noise of the actions sampled at steps
            ``1..H``.
    """

    def step(carry, noise):
        state, action = carry
        prior_noise_t, action_noise_t = noise
        state, out = rssm.apply(
            {"params": rssm_params},
            state,
            action,
            prior_noise_t,
            method=RSSM.imagine_step,
        )
        stoch = sg(out.stoch)
        action = policy(features(out.deter, stoch)).sample(action_noise_t)
        return (state, action), Trajectory(out.deter, stoch, action)

    noise = (jnp.swapaxes(prior_noise, 0, 1), jnp.swapaxes(action_noise, 0, 1))
    _, traj = jax.lax.scan(step, sg((start, start_action)), noise)
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj)


class ImaginationLoss(NamedTuple):
    """Algorithm F's outputs on ``N`` trajectories of ``H + 1`` states.

    Attributes:
        actor: ``[N, H]`` the actor loss per state ``0..H-1``.
        critic: ``[N, H]`` the critic loss per state ``0..H-1``.
        value: ``[N, H + 1]`` the online critic's prediction.
        weight: ``[N, H + 1]`` the cumulative continuation.
        ret: ``[N, H]`` the lambda-returns.
        adv: ``[N, H]`` the normalised advantages.
        entropy: ``[N, H]`` the policy entropy.
        retnorm: the return normaliser after its update.
        scale: the return scale ``max(1, hi - lo)`` the advantages used.
    """

    actor: jax.Array
    critic: jax.Array
    value: jax.Array
    weight: jax.Array
    ret: jax.Array
    adv: jax.Array
    entropy: jax.Array
    retnorm: ReturnNormalizer
    scale: jax.Array


def imagination_loss(
    config: DreamerV3Config,
    policy: Policy,
    action: jax.Array,
    value_logits: jax.Array,
    slow_value: jax.Array,
    reward: jax.Array,
    cont: jax.Array,
    retnorm: ReturnNormalizer,
    update_retnorm: bool = True,
) -> ImaginationLoss:
    """Actor and critic losses on imagined trajectories (Algorithm F).

    The inputs are the heads evaluated on the stop-gradiented ``[N, H + 1]``
    states (``agent.py:284-299``): the policy, its sampled ``action``, the
    critic's logits, the slow critic's prediction and the reward and
    continuation of each state (index 0 from the replayed data). Then
    (``agent.py:300-329``; dreamerv3_spec 3.5-3.12, 3.17):

    * ``weight_t = prod_{i <= t} cont_i``, including the start's (3.6);
    * ``R = lambda_return(reward, live=cont, cont=lam, boot=value)`` from
      the **online** critic (``slowtar: False``, 3.7): the continuation
      probabilities are the discounted continuation ``live`` (the reference's
      ``disc``) and the scalar lambda its lambda continuation;
    * the normaliser folds the percentiles of ``R`` in, then gives ``S =
      max(1, hi - lo)`` (update then read, 3.8; ``update_retnorm=False``
      only reads it, as the reference's report pass);
    * ``adv = (R - value) / S``, no offset subtracted (3.10);
    * ``actor = sg(w) * -(log pi(sg(a)) * sg(adv) + actent * H[pi])``:
      REINFORCE for both action types (3.11);
    * ``critic = sg(w) * (CE(sg(R)) + slowreg * CE(sg(slow_value)))`` with
      the two-hot cross-entropy (3.17).

    Losses cover states ``0..H-1``; the last state only bootstraps.
    """
    twohot = TwoHot.dreamerv3(config.bins)
    value = twohot.decode(value_logits)
    weight = jnp.cumprod(cont, 1)
    ret = lambda_return(reward, live=cont, cont=config.lam, boot=value)
    if update_retnorm:
        retnorm = retnorm.update(ret)
    scale = retnorm.scale()
    adv = (ret - value[:, :-1]) / scale
    log_pi = policy.log_prob(sg(action))[:, :-1]
    entropy = policy.entropy()[:, :-1]
    actor = sg(weight[:, :-1]) * -(log_pi * sg(adv) + config.actent * entropy)
    target = jnp.concatenate([ret, 0 * ret[:, -1:]], 1)
    critic = (
        sg(weight)[:, :-1]
        * (
            twohot.loss(value_logits, sg(target))
            + config.slowreg * twohot.loss(value_logits, sg(slow_value))
        )[:, :-1]
    )
    return ImaginationLoss(
        actor=actor,
        critic=critic,
        value=value,
        weight=weight,
        ret=ret,
        adv=adv,
        entropy=entropy,
        retnorm=retnorm,
        scale=scale,
    )


def replay_critic_loss(
    config: DreamerV3Config,
    value_logits: jax.Array,
    slow_value: jax.Array,
    boot: jax.Array,
    reward: jax.Array,
    is_terminal: jax.Array,
    is_last: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """The replay critic loss ``[B, T - 1]`` and its returns (Algorithm G).

    On the replayed steps ``1..T`` of the batch (``agent.py:331-352``;
    dreamerv3_spec 3.19): the lambda-return of the replayed rewards with the
    fixed discount ``gamma = 1 - 1 / return_horizon``, ``live = gamma (1 -
    is_terminal)``, ``cont = repval_lam (1 - is_last)``, bootstrapped from
    ``boot [B, T]``, the imagination return ``R_0`` of the rollout that
    started at each replayed state; then the two-hot cross-entropy toward it
    and toward the slow critic's prediction, masked by ``1 - is_last`` (no
    continuation product). ``value_logits [B, T, bins]`` are the critic's on
    the replayed posterior features, which carry gradient into the world
    model (``replay_critic_grad: True``). The last step gets no loss.
    """
    twohot = TwoHot.dreamerv3(config.bins)
    live = (~is_terminal).astype(jnp.float32) * config.gamma
    cont = (~is_last).astype(jnp.float32) * config.repval_lam
    ret = lambda_return(reward, live=live, cont=cont, boot=boot)
    target = jnp.concatenate([ret, 0 * ret[:, -1:]], 1)
    mask = sg((~is_last).astype(jnp.float32))
    loss = (
        mask[:, :-1]
        * (
            twohot.loss(value_logits, sg(target))
            + config.slowreg * twohot.loss(value_logits, sg(slow_value))
        )[:, :-1]
    )
    return loss, ret
