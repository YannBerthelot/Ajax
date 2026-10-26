"""Offline-RL critic regularisers as composable :class:`Extension`s.

* :class:`ConservativeQ` — CQL(H)'s conservative critic term.
* :class:`CalQL`         — the same term, its policy-action values floored at
  the data's Monte-Carlo returns (Cal-QL).
* :class:`TargetBC`      — ReBRAC's penalty on the target's next actions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from ajax.environments.interaction import get_pi
from ajax.extensions.base import Extension, ExtensionContext
from ajax.networks.networks import predict_value


def policy_samples(
    actor_state: Any, observations: jax.Array, key: jax.Array, n: int
) -> tuple[jax.Array, jax.Array]:
    """``n`` actions of the actor at each observation, ``(n, B, A)``, and
    their log-densities, ``(n, B, 1)``, held constant."""
    pi, _ = get_pi(actor_state, actor_state.params, observations)
    actions, log_probs = jax.vmap(lambda k: pi.sample_and_log_prob(seed=k))(
        jax.random.split(key, n)
    )
    log_probs = log_probs.sum(-1, keepdims=True)
    return jax.lax.stop_gradient(actions), jax.lax.stop_gradient(log_probs)


def sampled_q(
    critic_state: Any, params: Any, observations: jax.Array, actions: jax.Array
) -> jax.Array:
    """Every critic at each observation and each of its ``(n, B, A)``
    actions: ``(critics, n, B, 1)``."""
    n, b, _ = actions.shape
    obs = jnp.broadcast_to(observations, (n, *observations.shape))
    x = jnp.concatenate([obs, actions], axis=-1).reshape(n * b, -1)
    q = predict_value(critic_state, params, x)
    return q.reshape(q.shape[0], n, b, 1)


@dataclass(frozen=True)
class ConservativeQ(Extension):
    """CQL(H)'s critic term (Kumar, Zhou, Tucker and Levine, NeurIPS 2020,
    eq. 4), in the importance-sampled form of the authors' code
    (aviralkumar2907/CQL@d67dbe9, d4rl/rlkit/torch/sac/cql.py:234-264,
    ``min_q_version`` 3)::

        alpha * mean( logsumexp_j [Q(s, a_j) - log density(a_j)] - Q(s, a) )

    over the critics and the batch, with ``a`` the row's action and the
    ``a_j`` ``n_actions`` uniform actions in ``[-1, 1]^A`` and ``n_actions``
    of the policy at ``s`` and at ``s'``, all scored at ``s``. It pushes the
    critic down on actions away from the data's and up on the data's. The
    reference sums each critic's Bellman loss and term; Ajax's critics share
    one mean squared error, so the term is the mean over them, at the same
    ratio. It needs a policy with a density: a deterministic one (TD3's) is
    a point mass, whose samples are copies of one action and whose density
    is undefined. Deviations from the reference's SAC: its target keeps SAC's
    entropy term (the reference's ``deterministic_backup`` drops it). Logs
    ``cql/term`` and ``cql/gap``, the logsumexp minus the data's value.
    """

    alpha: float = 5.0
    n_actions: int = 10
    name: str = "conservative_q"
    # The logsumexp weighs the policy's samples by their density: a point
    # mass has none (TD3's Deterministic reports 0), so it is refused.
    requires: ClassVar[frozenset[str]] = frozenset(
        {"actor", "continuous_actions", "policy_density", "q_critic"}
    )
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "critic_loss": frozenset(
            {
                "actions",
                "actor_state",
                "critic_params",
                "critic_state",
                "next_observations",
                "observations",
            }
        )
    }

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.n_actions < 1:
            raise ValueError(
                f"{type(self).__name__} takes alpha >= 0 and n_actions >= 1,"
                f" got {self.alpha} and {self.n_actions}."
            )

    def floor(self, q: jax.Array, batch: Any) -> tuple[jax.Array, dict]:
        """The policy actions' values as the term reads them (CalQL floors
        them); no metrics."""
        del batch
        return q, {}

    def critic_loss(
        self, agent_state: Any, ext_state: Any, batch: Any, context: ExtensionContext
    ) -> Any:
        del agent_state, ext_state
        obs, actions = batch["observations"], batch["actions"]
        critic, params = batch["critic_state"], batch["critic_params"]
        n, (b, dim) = self.n_actions, actions.shape
        k_uniform, k_now, k_next = jax.random.split(context.rng, 3)
        uniform = jax.random.uniform(k_uniform, (n, b, dim), minval=-1.0, maxval=1.0)
        now, log_now = policy_samples(batch["actor_state"], obs, k_now, n)
        nxt, log_nxt = policy_samples(
            batch["actor_state"], batch["next_observations"], k_next, n
        )
        q_now, metrics = self.floor(sampled_q(critic, params, obs, now), batch)
        q_nxt, _ = self.floor(sampled_q(critic, params, obs, nxt), batch)
        # The uniform actions' density is 2^-dim.
        q_uniform = sampled_q(critic, params, obs, uniform) + dim * jnp.log(2.0)
        samples = jnp.concatenate([q_uniform, q_nxt - log_nxt, q_now - log_now], 1)
        q_data = predict_value(critic, params, jnp.concatenate([obs, actions], -1))
        gap = jnp.mean(jax.scipy.special.logsumexp(samples, axis=1) - q_data)
        term = self.alpha * gap
        logged = {"cql/term": term, "cql/gap": gap}
        logged.update(metrics)
        return term, logged


@dataclass(frozen=True)
class CalQL(ConservativeQ):
    """Cal-QL (Nakamoto, Zhai, Singh, Mark, Ma, Finn, Kumar and Levine,
    NeurIPS 2023): :class:`ConservativeQ`'s term with the policy actions'
    values floored at the row's Monte-Carlo return to go before the
    logsumexp (nakamotoo/Cal-QL@ac6eafe, JaxCQL/conservative_sac.py:227-240),
    so the critic is pushed down no further than the data's own return. The
    return to go is the transition dataset's (``return_to_go``), at its
    ``gamma``, in the env's reward units: the agent's ``gamma`` and a
    ``reward_scale`` of 1 match it. Deviation: the reference computes a
    collected episode's returns when it ends; collected rows here keep
    ``-inf``, no floor. Also logs ``calql/bound_frac``, the share of policy
    actions at ``s`` whose value the floor raised.
    """

    name: str = "cal_ql"
    requires: ClassVar[frozenset[str]] = ConservativeQ.requires.union(
        {"discounted", "replay"}
    )
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "critic_loss": ConservativeQ.batch_keys["critic_loss"].union({"return_to_go"})
    }
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"return_to_go"})

    def floor(self, q: jax.Array, batch: Any) -> tuple[jax.Array, dict]:
        bound = jax.lax.stop_gradient(batch["return_to_go"])
        return jnp.maximum(q, bound), {"calql/bound_frac": jnp.mean(q < bound)}


@dataclass(frozen=True)
class TargetBC(Extension):
    """ReBRAC's critic penalty (Tarasov, Kurenkov, Nikulin and Kolesnikov,
    NeurIPS 2023; tinkoff-ai/ReBRAC@694385b, src/algorithms/rebrac.py:
    134-144): the bootstrap less ``beta`` times the squared distance from
    the next actions the target scored (TD3's smoothed ``next_actions``) to
    the dataset's next action ``a'`` (``next_action``)::

        y -= gamma (1 - d) beta sum_i (next_actions_i - a'_i)^2

    A row whose ``a'`` is unknown (an episode's last row, a collected one:
    NaN) keeps its target. Logs ``target_bc/penalty``, the mean penalty, and
    ``target_bc/known_frac``, the share of rows with an ``a'``.
    """

    beta: float = 1.0
    name: str = "target_bc"
    # The dataset's next action reaches the target through a replay, and the
    # penalty is on the next actions the target scores.
    requires: ClassVar[frozenset[str]] = frozenset(
        {
            "actor",
            "continuous_actions",
            "discounted",
            "next_action_target",
            "q_critic",
            "replay",
        }
    )
    batch_keys: ClassVar[Mapping[str, frozenset[str]]] = {
        "on_target": frozenset({"dones", "gamma", "next_action", "next_actions"})
    }
    buffer_fields: ClassVar[frozenset[str]] = frozenset({"next_action"})

    def on_target(
        self,
        agent_state: Any,
        ext_state: Any,
        batch: Any,
        target: jax.Array,
        context: ExtensionContext,
    ) -> Any:
        del agent_state, ext_state, context
        a_next = batch["next_action"]
        known = jnp.all(jnp.isfinite(a_next), axis=-1, keepdims=True)
        gap = batch["next_actions"] - jnp.where(known, a_next, 0.0)
        penalty = jnp.where(known, self.beta * jnp.sum(gap**2, -1, keepdims=True), 0.0)
        target = target - batch["gamma"] * (1.0 - batch["dones"]) * penalty
        return target, {
            "target_bc/penalty": jnp.mean(penalty),
            "target_bc/known_frac": jnp.mean(known),
        }


__all__ = ["CalQL", "ConservativeQ", "TargetBC"]
