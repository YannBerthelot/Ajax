"""PID policies: an actor that is a PID controller with learned gains.

The actor acts on the error terms ``(e, I, D)`` that
:class:`ajax.environments.pid_terms.PIDTermsWrapper` puts first in the
observation; its pre-squash action, per action dimension ``a``, is::

    u_a = sum_j Kp[a, j] e_j + Ki[a, j] I_j + Kd[a, j] D_j

Passing a :class:`PIDPolicy` as ``actor_architecture`` selects it in any
continuous-action agent (SAC, TD3, REDQ, ASAC, AVG, PPO, APO). Two kinds
of gains:

* ``gains="fixed"``: the gains are the actor's parameters, a linear map
  with no bias and no hidden layer, so the trained policy deploys as a
  plain PID with the gains :func:`pid_gains` reads off it (Lawrence et
  al. 2022, "Deep reinforcement learning with shallow controllers: an
  experimental application to PID tuning", cited from memory, unverified).
* ``gains="observation"``: an MLP predicts the gains from the observation
  (gain scheduling); with a constant network it is the fixed policy.

Each agent's own output rules apply to ``u`` as to any actor's mean: SAC,
REDQ, ASAC, AVG and APO squash it with ``tanh``, TD3 too (deterministic),
PPO samples around it unsquashed (unless ``squash=True``) and the env
clips. The deployed controller is that saturated PID. The stochastic
agents explore with a learned, state-independent log-std. Gains start at
zero (no control).
Observation normalisation (env- or agent-side) leaves the terms raw, as
the wrapper declares them, so the gains stay a PID's in the env's units;
the gain network of ``gains="observation"`` reads the rest normalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from ajax.agents.SAC.utils import SquashedNormal
from ajax.environments.pid_terms import split_pid_terms
from ajax.networks.utils import parse_architecture

GAIN_NAMES = ("kp", "ki", "kd")


@dataclass(frozen=True)
class PIDPolicy:
    """A PID actor, passed as an agent's ``actor_architecture``.

    n_errors: errors the :class:`PIDTermsWrapper` tracks (its terms are
        the first ``3 * n_errors`` observation entries).
    gains: ``"fixed"`` (learned constants) or ``"observation"`` (an MLP of
        ``architecture`` predicts them from the observation).
    architecture: the gain network's hidden layers (``"observation"``
        only; empty: gains affine in the observation).
    """

    n_errors: int = 1
    gains: Literal["fixed", "observation"] = "fixed"
    architecture: tuple = ()

    def __post_init__(self):
        if self.n_errors < 1 or self.gains not in ("fixed", "observation"):
            raise ValueError(
                "PIDPolicy needs n_errors >= 1 and gains 'fixed' or"
                f" 'observation', got {self.n_errors} and {self.gains!r}."
            )
        if self.gains == "fixed" and self.architecture:
            raise ValueError("Fixed PID gains have no network: drop architecture.")


class PIDLaw(nn.Module):
    """The PID output ``u`` for an observation, shape ``(..., action_dim)``."""

    policy: PIDPolicy
    action_dim: int

    def setup(self):
        shape = (self.action_dim, self.policy.n_errors)
        if self.policy.gains == "fixed":
            self.fixed = [
                self.param(n, nn.initializers.zeros, shape) for n in GAIN_NAMES
            ]
        else:
            self.features = parse_architecture(self.policy.architecture)
            self.head = nn.Dense(
                3 * self.action_dim * self.policy.n_errors,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
            )

    def gains(self, obs: jax.Array) -> jax.Array:
        """Kp, Ki, Kd stacked: ``(..., 3, action_dim, n_errors)``."""
        if self.policy.gains == "fixed":
            gains = jnp.stack(self.fixed)
            return jnp.broadcast_to(gains, obs.shape[:-1] + gains.shape)
        h = obs
        for layer in self.features:
            h = layer(h)
        shape = (3, self.action_dim, self.policy.n_errors)
        return self.head(h).reshape(*obs.shape[:-1], *shape)

    def __call__(self, obs: jax.Array) -> jax.Array:
        terms = split_pid_terms(obs, self.policy.n_errors)
        return jnp.einsum("...tak,...tk->...a", self.gains(obs), terms)


class PIDActor(nn.Module):
    """Stochastic PID actor: Normal (or tanh-squashed) around the PID output."""

    policy: PIDPolicy
    action_dim: int
    squash: bool = False
    log_std_init: float = -1.0

    def setup(self):
        self.pid = PIDLaw(self.policy, self.action_dim)
        self.log_std = self.param(
            "log_std", nn.initializers.constant(self.log_std_init), (self.action_dim,)
        )

    def gains(self, obs: jax.Array) -> jax.Array:
        return self.pid.gains(obs)

    def __call__(self, obs: jax.Array) -> distrax.Distribution:
        mean = self.pid(obs)
        std = jnp.exp(jnp.clip(jnp.broadcast_to(self.log_std, mean.shape), -20, 2))
        return SquashedNormal(mean, std) if self.squash else distrax.Normal(mean, std)


def check_pid_actor(continuous: bool, memory: object) -> None:
    """Reject what a PID actor cannot be: a discrete policy or a recurrent
    one (its memory is the wrapper's integral and difference)."""
    if not continuous:
        raise ValueError("A PIDPolicy needs a continuous action space.")
    if memory is not None:
        raise NotImplementedError(
            "A PIDPolicy reads its integral and difference from the"
            " observation (PIDTermsWrapper); it takes no memory."
        )


def pid_gains(actor_params) -> dict[str, jax.Array]:
    """``{"kp", "ki", "kd"}`` of a fixed-gain PID actor, each ``(...,
    action_dim, n_errors)`` (leading axes, e.g. seeds, kept).

    Observation-conditioned gains depend on the observation: read them
    with ``actor.apply(params, obs, method="gains")``.
    """
    law = actor_params["params"]["pid"]
    if "kp" not in law:
        raise ValueError(
            "These PID gains are predicted from the observation: read them"
            ' with actor.apply(params, obs, method="gains").'
        )
    return {name: law[name] for name in GAIN_NAMES}


__all__ = ["PIDActor", "PIDLaw", "PIDPolicy", "check_pid_actor", "pid_gains"]
