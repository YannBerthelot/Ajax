"""The action transforms on a known answer: residual RL and gain mode.

One step from s in {-1, +1}, paying -(a - 0.8 s)^2: the best executed
action is 0.8 s, inside the box. The reading is the mean executed action
over the policy's samples, which the quadratic reward puts at 0.8 s whatever
their spread (a stochastic policy's squashed mean lies past it).

Residual RL (Johannink et al., 2019): the expert acts 0.4 s and the agent
executes clip(0.4 s + a_pi), so its policy must learn the other 0.4 s. A
critic scoring the policy's own action instead of the executed one pushes
the policy to the target itself, 0.8 s, and the executed action to the
bound.

Gain mode: a P expert acts g s, its gain g = 0.4 * 10^a set by the agent's
action a, so the agent must learn the gain 0.8 (a = log10 2) and execute
0.8 s. Untrained, the agent leaves the expert near its anchor gain: 0.4 s.

The cells are the agents whose answer here is the greedy one, calibrated on
2000-16000 steps (2026-10-24). Left out, each for an answer of its own:
- PPO and APO: an unsquashed Gaussian that keeps its spread within these
  budgets, whose best mean on a clipped box then lies past the bound (PPO
  without a transform does the same);
- IQL: advantage-weighted regression settles on a behaviour-weighted action;
- AVG: its fixed temperature biases the mean;
- TD3: a deterministic actor that steps past where the residual's clip
  zeroes the critic's gradient, and stays at the bound;
- CrossQ: its seeds spread (worst 0.23 at 16000 steps; 13/32 certified);
- REDQ: certified 32/32 at 16000 steps, but its 20 critic updates a step
  take 9 minutes, for the core's code TQC and ASAC probe already.
Their wiring is tests/agents/test_residual_policy.py's and test_gain_mode.py's.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from ajax.extensions.expert import ExpertGains, ResidualPolicy

from . import agents, envs, runs
from . import readouts as R
from .verdict import Case, Query, check, params

TARGET, BASE = 0.8, 0.4  # the best executed action, and the expert's, times s
SAMPLES = 256  # of the policy's actions per reading


def bandit() -> envs.Spec:
    def transition(st: envs.State, a: jax.Array, key: Any) -> tuple:
        a = a.reshape(-1)[0]
        return st.replace(s=st.s + 1), -((a - TARGET * st.c) ** 2), True

    def reset(key: jax.Array, params: Any) -> tuple:
        return 0, jnp.where(jax.random.bernoulli(key), 1.0, -1.0)

    obs = lambda st: st.c.reshape(1)  # noqa: E731
    return envs.Spec("transform_bandit", transition, obs, reset, obs_box=(-1.0, 1.0))


def residual_expert(obs: jax.Array) -> jax.Array:
    return BASE * obs[..., :1]


class GainExpert:
    """A P expert ``g s``, its gain learnable, anchored at ``BASE``."""

    learnable_fields, anchor_gains = ("kp",), jnp.array([BASE])

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return self.step_with_gains(state, obs, self.anchor_gains)

    def step_with_gains(self, state: jax.Array, obs: jax.Array, gains: Any) -> tuple:
        return jnp.clip(gains[..., :1] * obs[..., :1], -1.0, 1.0), state


#: Overrides of the probe preset for the agents it lacks or whose defaults
#: never warm up on one step.
EXTRA: dict[str, dict[str, Any]] = {
    # The policy updates from the warm-up's end, not SAC's 2000th step.
    "SAC": {"policy_update_start": 100, "alpha_update_start": 100},
}


def transform_readings(agent: str, gains: bool) -> Any:
    def build() -> Any:
        env, params = bandit().make()
        transform = ExpertGains() if gains else ResidualPolicy(scale=1.0)
        expert = GainExpert() if gains else residual_expert
        kw = EXTRA.get(agent, {}) | {"expert_policy": expert}
        return agents.make(agent, env, params, extensions=(transform,), **kw)

    def executed(a: jax.Array, s: float) -> jax.Array:
        if gains:
            return jnp.clip(BASE * 10**a * s, -1.0, 1.0)
        return jnp.clip(BASE * s + a, -1.0, 1.0)

    def read(run: runs.Run) -> dict:
        def at(n: R.Nets) -> dict:
            out = {}
            for s in (-1.0, 1.0):
                a = R.pi(n, s).sample(seed=jax.random.PRNGKey(0), sample_shape=SAMPLES)
                a = jnp.clip(a.reshape(-1), -1.0, 1.0)  # the agent's units
                out[f"a({s:g})"] = executed(a, s).mean()
            return out

        return R.per_seed(at, R.nets(run.state))

    return runs.readings(build, read)


def residual_case(agent: str, budget: int, tol: tuple = (), note: str = "") -> Case:
    queries = tuple(
        Query(
            f"a({s:g})",
            TARGET * s,
            {"the expert alone (untrained)": BASE * s, "the policy's action scored": s},
        )
        for s in (-1.0, 1.0)
    )
    name = f"residual-{agent}"
    return Case(name, queries, transform_readings(agent, False), budget, tol, note=note)


def gain_case(agent: str, budget: int, tol: tuple = (), note: str = "") -> Case:
    queries = tuple(
        Query(f"a({s:g})", TARGET * s, {"the anchor gain (untrained)": BASE * s})
        for s in (-1.0, 1.0)
    )
    name = f"gain-{agent}"
    return Case(name, queries, transform_readings(agent, True), budget, tol, note=note)


_LADDER = "ladder 2000-16000 on 1000-1031, 2000-2031"
CASES: dict[str, Case] = {
    c.id: c
    for c in (
        residual_case("SAC", 16000, (0.046, 0.057), _LADDER),
        residual_case("TQC", 16000, (0.048, 0.074), _LADDER),
        residual_case("ASAC", 16000, (0.089, 0.077), _LADDER),
        gain_case("ASAC", 8000, (0.094, 0.099), _LADDER),
        gain_case("SAC", 32000, (0.084, 0.086), _LADDER + ", then 32000"),
    )
}
ANSWER_DIGEST = "d57719eb620e"  # verdict.digest(CASES): every answer, pinned


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
