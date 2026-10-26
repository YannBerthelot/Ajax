"""The action stage on every off-policy agent: JSRL and EDGE substitute the
expert's action, the replay (AVG's critic, without one) learns from the
action executed.

Two steps on one observation, each paying clip(a) (a discrete action's
index), the second ending the episode: JSRL, its horizon one step that never
shrinks, hands the expert the first step and the policy the second, which
the replay tells apart by the termination alone. The expert takes -0.5
(index 0); the policy learns 1. The expert's first step bootstraps on the
policy's second, Q(-0.5) = -0.5 + gamma Q(1) = -0.5 + gamma, and the
policy's step pays its action, Q(a) = a. A critic scoring the policy's
action in place of the executed one fits half the rows at its own action to
the expert's reward: Q(1) = (-0.5 + gamma Q(1)) / 2 + 1/2, 0.36, so Q(a_pi)
- a_pi = -0.64 at a_pi = 1. EDGE's fixed gate hands the expert half the steps
of one-step episodes: half the rows hold a_E, and a critic scoring the
policy's action there reads Q(a) = a / 2 - 1/4. IBRL's expert takes 1, the
top of one-step episodes paying clip(a), above any action of the held
actor: once the target critics have learned Q(a) = a, collection executes
and flags the expert's action, and evaluation takes it too; an expert
taking -1, the bottom, is never taken.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from ajax.extensions.expert import JSRLCurriculum
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.target_mods import IBRL

from . import agents, envs, runs
from . import readouts as R
from .agents import GAMMA
from .verdict import Case, Query, check, params

CASES: dict[str, Case] = {}


def steps_on_one_observation(steps: int, actions: int = 0) -> envs.Spec:
    """One observation (1), steps per episode, each paying clip(a), or a
    discrete action's index."""

    def transition(st: envs.State, a: jax.Array, key: Any) -> tuple:
        if actions:
            paid = a.astype(jnp.float32)
        else:
            paid = jnp.clip(a.reshape(-1)[0], -1.0, 1.0)
        return st.replace(s=st.s + 1), paid, st.s + 1 >= steps

    def obs(st: envs.State) -> jax.Array:
        return jnp.ones(1, jnp.float32)

    spec = envs.Spec(f"steps_{steps}", transition, obs, actions=actions)
    return dataclasses.replace(spec, limit=steps + 1)


A_E = -0.5  # the expert's action (agent units: the box is [-1, 1])
FIRST_STEP = JSRLCurriculum(episode_length=1, decay_frac=1e6)
WARM = agents.PRESETS["probe"]["TD3"]["learning_starts"]  # the uniform rows


def expert(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), A_E, jnp.float32)


def first_index(obs: jax.Array) -> jax.Array:
    return jnp.zeros(obs.shape[:-1] + (1,), jnp.int32)


def handed_rows(run: runs.Run, start: int) -> dict[str, np.ndarray]:
    """Per seed, past row start, the share of the first and second steps'
    rows the expert took, as the replay flags them; and the share of the
    first steps' rows storing its action exactly (a pipeline storing its own
    action stores the substituted one)."""
    rows = R.replay_rows(run.state, ("action", "terminated", "is_expert"))
    second = rows["terminated"][:, 0, start:, 0] > 0.5
    took = rows["is_expert"][:, 0, start:, 0] > 0.5
    stored = rows["action"][:, 0, start:, 0] == A_E
    return {
        "expert share, first steps": np.mean(took, 1, where=~second),
        "expert share, second steps": np.mean(took, 1, where=second),
        "stored expert action, first steps": np.mean(stored, 1, where=~second),
    }


def scored(n: R.Nets) -> dict:
    a_pi = R.action(n, 1.0)
    return {
        "Q(a_E)": R.critic(n, 1.0, A_E),
        "Q(a_pi) - a_pi": R.critic(n, 1.0, a_pi) - a_pi,
    }


def readings(agent: str, extension: Any, steps: int, **kw: Any) -> Callable:
    """The agent on steps_on_one_observation, the expert handed steps by
    extension; readings past row start of a replay and its critic's."""
    discrete, start = kw.pop("discrete", False), kw.pop("start", 0)
    spec = steps_on_one_observation(steps, 2 if discrete else 0)

    def build() -> Any:
        policy = first_index if discrete else expert
        kw.update({"expert_policy": policy, "extensions": (extension,)})
        return agents.make(agent, *spec.make(), **kw)

    def read(run: runs.Run) -> dict:
        out = {}
        if run.state.collector_state.buffer_state is not None:
            out.update(handed_rows(run, start))
        if "q_critic" in run.agent.capabilities and not discrete:
            out.update(R.per_seed(scored, R.nets(run.state)))
        return out

    return runs.readings(build, read)


_HANDED = (
    Query("expert share, first steps", 1.0, {"JSRL not applied": 0.0}),
    Query(
        "expert share, second steps", 0.0, {"the expert acted past its horizon": 1.0}
    ),
)
_STORED = Query(
    "stored expert action, first steps",
    1.0,
    {"the pipeline's own action stored, not the executed one": 0.0},
)
_POLICY = "the critic scores the policy's action, not the executed one"
_UNBOOTSTRAPPED = {"the expert's step not bootstrapped": A_E}
#: TD3's bootstrap scores a' = clip(1 + clip(N(0, 0.2^2), -0.5, 0.5), -1, 1):
#: its mean, 1 - 0.2 (phi(0) - phi(2.5)) - 0.5 Phi(-2.5), 0.9206.
SMOOTHED = 1.0 - 0.2 * (norm.pdf(0.0) - norm.pdf(2.5)) - 0.5 * norm.cdf(-2.5)
DAGGER = {"actor_architecture": agents.NET, "batch_size": 64, "buffer_size": 10_000}
DAGGER.update({"beta_decay": 0.0, "iteration_steps": WARM})  # DAgger's own mix: off
CASES["q7-jsrl-TD3"] = Case(
    "q7-jsrl-TD3",
    (
        *_HANDED,
        Query("Q(a_E)", A_E + GAMMA * SMOOTHED, _UNBOOTSTRAPPED),
        Query("Q(a_pi) - a_pi", 0.0, {_POLICY: -0.64}),
    ),
    readings("TD3", FIRST_STEP, 2, start=WARM),
    2000,
    (0.02, 0.02, 0.056, 0.078),
    slow=True,
    note="ladder 1000-8000 on 1000-1031, 2000-2031: 2000 the first rung; cert 32/32",
)
CASES["q7-jsrl-AVG"] = Case(
    "q7-jsrl-AVG",
    (
        Query("Q(a_E)", A_E + GAMMA, _UNBOOTSTRAPPED),
        Query("Q(a_pi) - a_pi", 0.0, {_POLICY: -0.64}),
    ),
    readings("AVG", FIRST_STEP, 2, alpha_init=1e-4),
    8000,
    (0.142, 0.156),
    slow=True,
    ceiling=0.25,
    note="ladder 1000-16000: no rung within 0.1 (its incremental critic's"
    " spread); 8000 the first within 0.25, the critic's wrong answer 0.64 away;"
    " cert 31/32",
)
CASES["q7-jsrl-DAgger"] = Case(
    "q7-jsrl-DAgger",
    (*_HANDED, _STORED),
    readings("DAgger", FIRST_STEP, 2, start=WARM, preset=None, **DAGGER),
    200,
    (0.02, 0.02, 0.02),
    slow=True,
    note="exact: every seed at every rung (200, 400); cert 32/32",
)
CASES["q7-jsrl-DAgger-discrete"] = Case(
    "q7-jsrl-DAgger-discrete",
    _HANDED,
    readings("DAgger", FIRST_STEP, 2, start=WARM, discrete=True, preset=None, **DAGGER),
    200,
    (0.02, 0.02),
    slow=True,
    note="exact: every seed at every rung (200, 400); cert 32/32",
)
EDGE = EDGEExploration(gate="fixed", fixed_prob=0.5, decay_frac=0.0)


def edge_readings() -> Callable:
    """REDQ on one-step episodes, EDGE handing the expert half the steps: the
    share of the rows past the warm-up storing the expert's action, and the
    critic's readings."""
    env, params = steps_on_one_observation(1).make()

    def build() -> Any:
        return agents.make(
            "REDQ", env, params, extensions=(EDGE,), expert_policy=expert
        )

    def read(run: runs.Run) -> dict:
        rows = R.replay_rows(run.state, ("action",))["action"][:, 0, WARM:, 0]
        out = {"expert share": np.mean(rows == A_E, 1)}
        out.update(R.per_seed(scored, R.nets(run.state)))
        return out

    return runs.readings(build, read)


CASES["q7-edge-REDQ"] = Case(
    "q7-edge-REDQ",
    (
        Query("expert share", 0.5, {"the gate never hands over": 0.0, "always": 1.0}),
        Query("Q(a_pi) - a_pi", 0.0, {_POLICY: -0.7}),
    ),
    edge_readings(),
    1000,
    (0.082, 0.02),
    slow=True,
    note="ladder 1000-4000: 1000 the first rung, the share binomial over 900"
    " rows; cert 32/32",
)


@dataclasses.dataclass(frozen=True)
class Takes:
    action: float

    def __call__(self, obs: jax.Array) -> jax.Array:
        return jnp.full(obs.shape[:-1] + (1,), self.action, jnp.float32)


def ibrl_readings(agent: str, a_e: float, **kw: Any) -> Callable:
    """The agent on one-step episodes, IBRL's expert taking a_e, its actor
    held: per seed, the share of the replay's second half storing the
    expert's action and flagged as the expert's, and of the last
    evaluation's steps the expert took."""
    env, params = steps_on_one_observation(1).make()

    def build() -> Any:
        ibrl = IBRL(expert_policy=Takes(a_e))
        return agents.make(agent, env, params, extensions=(ibrl,), **kw)

    def read(run: runs.Run) -> dict:
        rows = R.replay_rows(run.state, ("action", "is_expert"))
        half = rows["action"].shape[2] // 2
        return {
            "rows storing a_E": np.mean(rows["action"][:, 0, half:, 0] == a_e, 1),
            "rows flagged": np.mean(rows["is_expert"][:, 0, half:, 0], 1),
            "evaluation's expert share": run.logged("Eval/expert share"),
        }

    return runs.readings(build, read, log_every=250)


_KEPT = {"IBRL keeps the agent's action": 0.0}
_UNFLAGGED = {"the expert's rows unflagged": 0.0}
_PLAIN = {"evaluation acts the plain policy": 0.0}
IBRL_ACTS = (
    Query("rows storing a_E", 1.0, _KEPT),
    Query("rows flagged", 1.0, _UNFLAGGED),
    Query("evaluation's expert share", 1.0, {**_PLAIN, **_UNFLAGGED}),
)
_ALWAYS = {"IBRL takes the expert's action whatever its value": 1.0}
IBRL_DECLINES = tuple(Query(q.name, 0.0, _ALWAYS) for q in IBRL_ACTS)
SAC_HELD = {"fixed_alpha": True, "alpha_init": 1e-4, "policy_update_start": 2**30}
TD3_HELD = {"actor_learning_rate": 0.0}
_NOTE = "ladder 250-2000 on 1000-1031, 2000-2031: 500 the first rung; cert 32/32"
CASES["q7-ibrl-acts-TD3"] = Case(
    "q7-ibrl-acts-TD3",
    IBRL_ACTS,
    ibrl_readings("TD3", 1.0, **TD3_HELD),
    500,
    (0.02, 0.02, 0.02),
    note=_NOTE,
)
CASES["q7-ibrl-acts-SAC"] = dataclasses.replace(
    CASES["q7-ibrl-acts-TD3"],
    id="q7-ibrl-acts-SAC",
    readings=ibrl_readings("SAC", 1.0, **SAC_HELD),
)
CASES["q7-ibrl-declines-TD3"] = dataclasses.replace(
    CASES["q7-ibrl-acts-TD3"],
    id="q7-ibrl-declines-TD3",
    queries=IBRL_DECLINES,
    readings=ibrl_readings("TD3", -1.0, **TD3_HELD),
)
ANSWER_DIGEST = "a7d7338019fd"


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
