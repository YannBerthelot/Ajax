"""Where an agent spends its updates — per-sample loss weighting.

Standard deep-RL training spends gradient mass in proportion to how often
a state is *visited*: every sampled transition contributes equally to the
loss, so the update budget follows the state-visitation distribution and
nothing else. Sometimes that is not what you want — you may care more
about states where the policy's choice actually changes returns, about a
curriculum, or about correcting a sampling bias you introduced upstream.

This module exposes that budget as something an experiment can control,
via the ``*_loss_weights`` phases: an extension supplies a per-sample
weight vector and the agent reduces its per-sample loss as a weighted
mean (``sum(w·l)/sum(w)``) rather than a plain mean. Because the
reduction is self-normalised, weights re-*allocate* the update budget
without changing its total size — a weighting extension is not a
disguised learning-rate schedule, which is what makes an unweighted arm a
fair control.

:class:`InterestWeighting` is the general form. The name is Sutton et
al.'s: in emphatic TD, the *interest function* ``i(s) ≥ 0`` states how
much one cares about being accurate at ``s``, and the update at ``s`` is
scaled by it. Here the same object is a plain callable on observations,
usable with any agent whose losses honour the weight phases.

Actor vs critic
---------------
The two heads are separately switchable (``apply_to_actor`` /
``apply_to_critic``) because they are not symmetric, and the asymmetry
matters:

* The **critic** is infrastructure. Bootstrapped value estimates couple
  every state to its successors, so value information has to flow
  *through* states you may not care about. Starving them can break
  propagation globally, and down-weighting the critic is therefore a
  sharp instrument — reach for it deliberately, e.g. as the control arm
  of an experiment that asks whether it does break.
* The **actor** is the deliverable. A policy may be arbitrarily wrong
  wherever its choice does not change returns, at no cost, so re-spending
  actor capacity toward states that matter has no analogous coupling
  argument against it.

Both default to off; enable exactly the head the experiment is about.

Example
-------
.. code-block:: python

    from ajax import PPO
    from ajax.extensions.allocation import InterestWeighting

    agent = PPO(
        env_id="CartPole-v1",
        extensions=[
            InterestWeighting(weight_fn=my_interest, apply_to_actor=True),
        ],
    )

``weight_fn`` maps a batch of observations to non-negative weights and
must be jit-compatible (it is traced inside the loss). It receives the
observations exactly as the loss function sees them — already
preprocessed / normalised if the agent does that upstream.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ajax.extensions.base import Extension, ExtensionContext

__all__ = ["EntropyAllocation", "InterestWeighting"]


@dataclass(frozen=True)
class InterestWeighting(Extension):
    """Weight per-sample losses by a function of the observation.

    Args:
        weight_fn: ``observations -> weights``, non-negative and
            jit-compatible. Returns one weight per sample; a trailing
            singleton axis is fine (weights are broadcast against the
            per-sample loss). Only relative magnitudes matter — the
            agent's reduction is self-normalising.
        apply_to_actor: weight the actor / policy loss.
        apply_to_critic: weight the critic / value loss.
        floor: added to every weight before use. A strictly positive
            floor keeps zero-interest samples contributing *something*,
            which matters when the weighted head also carries
            propagation (the critic). ``0.0`` disables the floor.

    Raises:
        ValueError: if neither head is enabled (a silent no-op is far
            more likely to be a mistake than an intention), or if the
            floor is negative.
    """

    weight_fn: Callable[[jax.Array], jax.Array]
    apply_to_actor: bool = False
    apply_to_critic: bool = False
    floor: float = 0.0

    name: str = "interest_weighting"

    def __post_init__(self) -> None:
        if not (self.apply_to_actor or self.apply_to_critic):
            raise ValueError(
                "InterestWeighting does nothing unless apply_to_actor or "
                "apply_to_critic is True — enable the head you mean to weight."
            )
        if self.floor < 0.0:
            raise ValueError(f"floor must be non-negative, got {self.floor}")

    # -- internals -------------------------------------------------------
    def _weights(self, batch: Any) -> jax.Array | None:
        """Weights for this minibatch, or ``None`` if it carries no obs."""
        observations = batch.get("observations") if isinstance(batch, dict) else None
        if observations is None:
            return None
        weights = jnp.asarray(self.weight_fn(observations))
        return weights + self.floor

    # -- phases ----------------------------------------------------------
    def actor_loss_weights(
        self, agent_state: Any, ext_state: Any, batch: Any, ctx: ExtensionContext
    ) -> jax.Array | None:
        del agent_state, ext_state, ctx
        return self._weights(batch) if self.apply_to_actor else None

    def critic_loss_weights(
        self, agent_state: Any, ext_state: Any, batch: Any, ctx: ExtensionContext
    ) -> jax.Array | None:
        del agent_state, ext_state, ctx
        return self._weights(batch) if self.apply_to_critic else None


@dataclass(frozen=True)
class EntropyAllocation(Extension):
    """Decide *where* a policy is allowed to be decisive.

    An entropy bonus is normally state-blind: it pulls the policy toward
    uniform by the same amount everywhere, so an agent forced to give up
    decisiveness gives it up evenly — including at the states where being
    decisive is the whole point. This extension makes that pull
    state-dependent, redistributing a fixed total amount of entropy pressure
    across states.

    Args:
        weight_fn: ``observations -> weights``, non-negative and
            jit-compatible. **Higher weight means more pull toward uniform
            at that state**, i.e. less decisiveness — the opposite reading
            from :class:`InterestWeighting`, where higher means more
            emphasis. To concentrate decisiveness where some relevance
            signal ``r(s)`` is large, pass something decreasing in ``r``.
        floor: added to every weight, so no state's entropy term can be
            switched off entirely.

    Because the agent reduces the entropy term with a self-normalising
    weighted mean, only the *distribution* of pressure changes: the total is
    preserved, and uniform weights reproduce an unweighted run exactly. That
    is what makes an unweighted arm a fair control.

    Raises:
        ValueError: if the floor is negative.
    """

    weight_fn: Callable[[jax.Array], jax.Array]
    floor: float = 0.0

    name: str = "entropy_allocation"

    def __post_init__(self) -> None:
        if self.floor < 0.0:
            raise ValueError(f"floor must be non-negative, got {self.floor}")

    def entropy_weights(
        self, agent_state: Any, ext_state: Any, batch: Any, ctx: ExtensionContext
    ) -> jax.Array | None:
        del agent_state, ext_state, ctx
        observations = batch.get("observations") if isinstance(batch, dict) else None
        if observations is None:
            return None
        return jnp.asarray(self.weight_fn(observations)) + self.floor
