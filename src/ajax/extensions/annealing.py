"""MEW, Minimum Excess Work (Adamczyk, Rojas and Kulkarni, "Thermodynamics
of Reinforcement Learning Curricula", SciForDL 2026, arXiv 2603.12324;
JacobHA/MEW@8f180eb): the temperature of a maximum-entropy agent annealed
towards zero at a constant thermodynamic speed, in place of its own rule::

    zeta(beta) = beta sum_k Cov(r_t, r_t+k)          eq. 4, eq. 8 (friction)
    beta <- beta + speed / sqrt(zeta)                       Algorithm 1
    alpha = 1 / beta

The covariance is that of the agent's last ``history`` rewards per env, read
from its replay (or, on an agent without one, kept by MEW itself), so the
temperature falls slowly while the rewards fluctuate and fast once they
settle. As the authors' code (annealed_asac_temporal.py), :func:`friction`
uses the unbiased autocovariance of each env's centred rewards summed over
every lag, floored at 1e-3, and MEW steps once the agent has collected more
than ``history`` steps per env. The paper's Algorithm 1 writes the step as
speed / (beta sqrt(zeta) + eps) with zeta without beta; the code's form is
kept. MEW steps after each update (post_update), where the authors' code
steps before it: the agent's next update reads the new temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import jax
import jax.numpy as jnp
from flax import struct
from flax.core import FrozenDict

from ajax.buffers.utils import written_rows
from ajax.extensions.base import Extension, ExtensionContext
from ajax.types import Replaceable

#: The authors' floor on the friction (MIN_ZETA), against a step blowing up
#: on rewards that do not vary.
MIN_FRICTION = 1e-3
#: The authors' guard: their update needs more than 100 rewards.
MIN_HISTORY = 101


def integrated_autocovariance(rewards: jax.Array) -> jax.Array:
    """sum_k Cov(r_t, r_t+k) over every lag k of each env's rewards (n_envs,
    n), each lag's mean over its n - k pairs, averaged over the envs. As the
    authors', by FFT of the centred rewards zero-padded to 2n (no wrap)."""
    centred = rewards - jnp.mean(rewards, axis=1, keepdims=True)
    length = rewards.shape[1]
    spectrum = jnp.fft.rfft(centred, n=2 * length, axis=1)
    products = jnp.fft.irfft(spectrum * jnp.conj(spectrum), n=2 * length, axis=1)
    pairs = jnp.arange(length, 0, -1)
    return jnp.mean(jnp.sum(products[:, :length] / pairs, axis=1))


def friction(rewards: jax.Array, beta: jax.Array) -> jax.Array:
    """Eq. 4's friction for the inverse temperature beta, from eq. 8's
    reward autocovariance, floored at MIN_FRICTION."""
    return jnp.maximum(beta * integrated_autocovariance(rewards), MIN_FRICTION)


def anneal(alpha: jax.Array, rewards: jax.Array, speed: float) -> tuple:
    """Algorithm 1's step: the next temperature, and the friction it read."""
    beta = 1.0 / (alpha + 1e-8)  # the authors' guard against alpha = 0
    zeta = friction(rewards, beta)
    return 1.0 / (beta + speed / jnp.sqrt(zeta)), zeta


@struct.dataclass
class MEWState(Replaceable):
    """friction: the last step's (NaN before the first). rewards: the last
    rewards, (history, n_envs), on an agent without a replay, and stored:
    how many it has received; None on a replay agent."""

    friction: jax.Array
    rewards: Optional[jax.Array] = None
    stored: Optional[jax.Array] = None


@dataclass(frozen=True)
class MEW(Extension):
    """Anneal the temperature by Minimum Excess Work (the module's
    docstring), in place of the agent's own temperature step. Logs
    ``mew/friction``.

    Args:
        speed: The thermodynamic speed eta, the authors' 1e-6 (Figure 3
            finds 1e-7 to 1e-4 alike on Humanoid).
        history: The recent rewards per env the friction reads, the
            authors' 50,000 (HISTORY_STEPS; the paper's text says 5,000);
            more than 100, as the authors' update needs. A smaller replay
            gives all it holds.
    """

    speed: float = 1e-6
    history: int = 50_000

    requires: ClassVar[frozenset[str]] = frozenset({"entropy_temperature"})
    sets_temperature: ClassVar[bool] = True
    # An agent without a replay (AVG) keeps its last step for MEW to store.
    reads_last_rollout: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.history < MIN_HISTORY:
            raise ValueError(
                f"MEW needs a history above 100 rewards, got {self.history}."
            )
        if self.speed < 0:
            raise ValueError(f"MEW's speed cannot be negative, got {self.speed}.")

    def init_state(self, agent_state: Any, rng: jax.Array) -> MEWState:
        del rng
        nan = jnp.asarray(jnp.nan, jnp.float32)
        if agent_state.collector_state.buffer_state is not None:
            return MEWState(friction=nan)
        n_envs = agent_state.collector_state.last_terminated.shape[0]
        rewards = jnp.zeros((self.history, n_envs), jnp.float32)
        return MEWState(friction=nan, rewards=rewards, stored=jnp.asarray(0))

    def recent_rewards(self, agent_state: Any, state: MEWState) -> tuple:
        """The last rewards, (n_envs, n), how many each env has held, and
        the state, the agent's last step stored on one without a replay."""
        replay = agent_state.collector_state.buffer_state
        if replay is not None:
            rewards = replay.experience["reward"][..., 0]
            window = min(self.history, rewards.shape[1])
            rows = (replay.current_index - window + jnp.arange(window)) % rewards.shape[
                1
            ]
            return rewards[:, rows], written_rows(replay), state
        # init_state built them, the agent having no replay.
        assert state.rewards is not None and state.stored is not None
        step = agent_state.last_rollout.reward.reshape(-1, state.rewards.shape[1])
        stored = jnp.concatenate([state.rewards, step])[step.shape[0] :]
        count = state.stored + step.shape[0]
        state = state.replace(rewards=stored, stored=count)
        return stored.T, jnp.minimum(count, self.history), state

    def post_update(
        self, agent_state: Any, ext_state: MEWState, context: ExtensionContext
    ) -> tuple[Any, MEWState]:
        """One annealing step, once more than history steps per env are
        collected and the history is full."""
        del context
        rewards, held, ext_state = self.recent_rewards(agent_state, ext_state)
        steps_per_env = agent_state.collector_state.timestep // rewards.shape[0]
        ready = jnp.logical_and(steps_per_env > self.history, held >= rewards.shape[1])
        temperature = agent_state.alpha
        log_alpha = temperature.params["log_alpha"]
        alpha, zeta = anneal(jnp.exp(log_alpha), rewards, self.speed)
        # A step counts as the temperature's (REDQ reads its own until one).
        temperature = temperature.replace(
            params=FrozenDict(
                {"log_alpha": jnp.where(ready, jnp.log(alpha), log_alpha)}
            ),
            step=temperature.step + ready.astype(temperature.step.dtype),
        )
        zeta = jnp.where(ready, zeta, ext_state.friction)
        return agent_state.replace(alpha=temperature), ext_state.replace(friction=zeta)

    def eval_metrics(
        self, agent_state: Any, ext_state: MEWState, rng: jax.Array, context: Any
    ) -> dict:
        del agent_state, rng, context
        return {"mew/friction": ext_state.friction}
