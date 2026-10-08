"""Known-answer probes for Ajax agents.

A probe trains an agent on a tiny environment from ProbingEnvironments
whose exact answer is known, then reads the agent's values or actions at
chosen observations, one reading per seed. Each query states the right
answer and the wrong answers specific faults produce; its tolerance stays
below half the gap to the nearest one, so a failure says which fault the
reading looks like.

Every check trains 8 seeds in one compiled program. It passes if at least
7 seeds are within tolerance on every query, fails if 4 or fewer are, and
otherwise trains 8 fresh seeds and passes if at least 13 of the 16 are.
For a healthy agent whose seeds pass 99% of the time this fails about once
in 80,000 checks; a fault that halves the per-seed pass rate is caught 96%
of the time (binomial arithmetic, not a measurement).

Budgets and tolerances come from ``calibrate.py`` (seeds 1000-1031, kept
apart from the test seeds 0-15).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces
from probing_environments.gymnax_envs import (
    AdvantagePolicyLossPolicyUpdateEnv,
    PolicyAndValueEnv,
    RewardDiscountingEnv,
    ValueBackpropEnv,
    ValueLossOrOptimizerEnv,
)
from probing_environments.gymnax_envs import continuous_actions as continuous

from ajax import DQN, PPO, PQN, SAC

GAMMA = 0.62
"""Discount for every probe. At 0.62 the discounting probe's wrong answers
(gamma squared 0.38, 1 - gamma 0.38, no discount 1) sit at least 0.24 from
the right one; at the package's 0.5, gamma equals 1 - gamma."""

STAGE_1_SEEDS = tuple(range(8))
STAGE_2_SEEDS = tuple(range(8, 16))
CALIBRATION_SEEDS = tuple(range(1000, 1032))
POOLED_SEEDS = tuple(range(2000, 2032))
CERTIFICATION_SEEDS = tuple(range(3000, 3032))


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------


def _symmetric_actions(env_cls: type) -> type:
    """The continuous policy probes reward the sign or size of an action in
    [-1, 1] but declare Box(0, 1) and never clip, so an agent with an
    unsquashed Gaussian policy (PPO) earns more the further its mean leaves
    the range, and an agent that rescales to the declared bounds faces
    another task. Declare and enforce the range the rewards assume."""

    class Symmetric(env_cls):  # type: ignore[misc, valid-type]
        def action_space(self, params: Any = None) -> spaces.Box:
            return spaces.Box(-1.0, 1.0, (1,), dtype=jnp.float32)

        def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
            return super().step_env(key, state, jnp.clip(action, -1.0, 1.0), params)

    Symmetric.__name__ = env_cls.__name__
    return Symmetric


class SignedActionEnv(continuous.PolicyAndValueEnv):
    """Observation s in {-1, +1}, one step, reward clip(a, -1, 1) * s.

    The package's coupling env pays +-1 for the action's sign only, so once
    a Gaussian policy's samples all fall on one side nothing moves it: on
    32 seeds PPO ends on the wrong side for one observation on up to a
    quarter of them, depending on budget. A reward proportional to the
    action keeps a gradient everywhere."""

    def action_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (1,), dtype=jnp.float32)

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
        reward = jnp.clip(jnp.squeeze(action), -1.0, 1.0) * state.x
        state = type(state)(x=state.x, time=state.time + 1)
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            self.is_terminated(state, params),
            {"discount": self.discount(state, params)},
        )


def make_env(env_cls: type) -> tuple[Any, Any]:
    env = env_cls()
    # gymnax marks a step truncated once time reaches max_steps_in_episode;
    # the probes end every episode by termination, so keep that clause out.
    return env, env.default_params.replace(max_steps_in_episode=10_000)


# ---------------------------------------------------------------------------
# Agents and readouts
# ---------------------------------------------------------------------------

_NET = ("64", "relu", "64", "relu")

N_ENVS = {"PQN": 8}
"""Parallel environments per agent (default 1). PQN learns from parallel
environments instead of a replay buffer; with one environment the Q-value
of the action it stops exploring drifts, and on 64 seeds its coupling
check failed 1 to 4 seeds per budget, against none with 8 environments."""


def make_agent(agent_cls: type, env: Any, env_params: Any) -> Any:
    """The probing configuration of each agent (the package adaptor's)."""
    name = agent_cls.__name__
    common = {
        "env_id": env,
        "n_envs": N_ENVS.get(name, 1),
        "env_params": env_params,
        "gamma": GAMMA,
    }
    if name == "PQN":
        return agent_cls(
            **common,
            learning_rate=1e-3,
            architecture=_NET,
            n_steps=16,
            n_epochs=4,
            num_minibatches=1,
        )
    if name == "DQN":
        return agent_cls(
            **common,
            learning_rate=1e-3,
            architecture=_NET,
            learning_starts=100,
            buffer_size=10_000,
            batch_size=64,
            target_update_interval=100,
        )
    split = {
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "actor_architecture": _NET,
        "critic_architecture": _NET,
        "normalize_observations": False,
        "normalize_rewards": False,
    }
    if name == "PPO":
        return agent_cls(**common, **split, n_steps=32, batch_size=32, n_epochs=4)
    if name == "SAC":
        return agent_cls(
            **common, **split, learning_starts=100, buffer_size=10_000, batch_size=64
        )
    raise ValueError(f"No probing configuration for {name}")


def train(agent_cls: type, env_cls: type, seeds: Sequence[int], budget: int) -> Any:
    """Train every seed in one compiled program; the state keeps the seed
    axis first. ``budget=0`` returns the untrained networks."""
    env, env_params = make_env(env_cls)
    agent = make_agent(agent_cls, env, env_params)
    result = agent.train(seed=list(seeds), n_timesteps=budget, logging_config=None)
    return result[0] if isinstance(result, tuple) else result


def _obs(x: float) -> jax.Array:
    return jnp.array([[x]], dtype=jnp.float32)


def _networks(state: Any) -> dict[str, Any]:
    """The train states the readings use; only they carry the seed axis on
    every leaf."""
    nets = {"actor": state.actor_state}
    if getattr(state, "critic_state", None) is not None:
        nets["critic"] = state.critic_state
    return nets


def _policy(nets: Mapping[str, Any], x: float) -> Any:
    return nets["actor"].apply_fn(nets["actor"].params, _obs(x))


def _value(family: str) -> Callable[[Mapping[str, Any], float], jax.Array]:
    def read(nets: Mapping[str, Any], x: float) -> jax.Array:
        if family == "dqn":
            return _policy(nets, x).q_values.max()
        critic = nets["critic"]
        if family == "v":
            return critic.apply_fn(critic.params, _obs(x)).mean()
        pi = _policy(nets, x)
        # Q(s, mean action), averaged over the critic ensemble.
        q = critic.apply_fn(
            critic.params, jnp.concatenate([_obs(x), pi.mean()], axis=-1)
        )
        return q.mean()

    return read


def _q_value(action: int) -> Callable[[Mapping[str, Any], float], jax.Array]:
    def read(nets: Mapping[str, Any], x: float) -> jax.Array:
        return _policy(nets, x).q_values.reshape(-1)[action]

    return read


def _action_mean(nets: Mapping[str, Any], x: float) -> jax.Array:
    """The deterministic action as the env applies it (clipped to [-1, 1])."""
    return jnp.clip(_policy(nets, x).mean().reshape(()), -1.0, 1.0)


def _q_gap(right: int, wrong: int) -> Callable[[Mapping[str, Any], float], jax.Array]:
    def read(nets: Mapping[str, Any], x: float) -> jax.Array:
        q = _policy(nets, x).q_values.reshape(-1)
        return q[right] - q[wrong]

    return read


FAMILY = {"SAC": "q", "PPO": "v", "DQN": "dqn", "PQN": "dqn"}


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Query:
    """One reading of the trained agent and what it should be.

    A two-sided query passes when the reading is within its tolerance of the
    truth. A one-sided query (``margin=True``) asks only that the reading
    reach a margin on the truth's side of zero: a sign the reward rewards,
    or a gap between the right and the wrong action's values. Its truth is
    the reward-optimal reading, which max-entropy agents and Q-learners with
    decayed exploration need not reach, since nothing pays for the excess.
    """

    name: str
    reading: str  # "value", "q0", "q1", "gap01", "gap10" or "action"
    obs: float
    truth: float
    wrong: Mapping[str, float]
    margin: bool = False

    def read(self, family: str) -> Callable[[Mapping[str, Any], float], jax.Array]:
        if self.reading == "value":
            return _value(family)
        if self.reading == "action":
            return _action_mean
        if self.reading.startswith("gap"):
            return _q_gap(int(self.reading[3]), int(self.reading[4]))
        return _q_value(int(self.reading[1:]))

    @property
    def max_tolerance(self) -> float:
        """Half the gap to the nearest wrong answer."""
        return 0.5 * min(abs(self.truth - w) for w in self.wrong.values())

    def within(self, readings: np.ndarray, tolerance: float) -> np.ndarray:
        """Per seed: two-sided, |reading - truth| <= tolerance; one-sided,
        the reading reaches ``tolerance`` on the truth's side."""
        if self.margin:
            return readings * np.sign(self.truth) >= tolerance
        return np.abs(readings - self.truth) <= tolerance


@dataclasses.dataclass(frozen=True)
class Probe:
    name: str
    env: type
    queries: tuple[Query, ...]


_DONE_IGNORED = 1.0 / (1.0 - GAMMA)
_TERMINAL_DONE_IGNORED = 1.0 / (1.0 - GAMMA**2)


def _value_probes(discrete: bool) -> tuple[Probe, ...]:
    """The three value probes, in both action flavours."""
    pick = (lambda d, c: d) if discrete else (lambda d, c: c)
    return (
        Probe(
            "value",
            pick(ValueLossOrOptimizerEnv, continuous.ValueLossOrOptimizerEnv),
            (Query("V(0)", "value", 0.0, 1.0, {"done mask ignored": _DONE_IGNORED}),),
        ),
        Probe(
            "backprop",
            pick(ValueBackpropEnv, continuous.ValueBackpropEnv),
            (
                Query("V(0)", "value", 0.0, 0.0, {"observation ignored": 0.5}),
                Query("V(1)", "value", 1.0, 1.0, {"observation ignored": 0.5}),
            ),
        ),
        Probe(
            "discounting",
            pick(RewardDiscountingEnv, continuous.RewardDiscountingEnv),
            (
                Query(
                    "V(0)",
                    "value",
                    0.0,
                    GAMMA,
                    {
                        "gamma squared": GAMMA**2,
                        "1 - gamma": 1.0 - GAMMA,
                        "no discount": 1.0,
                        # The first transition of each episode masked as
                        # done (replay of the 0382f32 buffer bug).
                        "done flag one step late": 0.0,
                    },
                ),
                Query(
                    "V(1)",
                    "value",
                    1.0,
                    1.0,
                    {"done mask ignored": _TERMINAL_DONE_IGNORED},
                ),
            ),
        ),
    )


CONTINUOUS_PROBES = (
    *_value_probes(discrete=False),
    Probe(
        "advantage",
        _symmetric_actions(continuous.AdvantagePolicyLossPolicyUpdateEnv),
        # Reward = action; the reset observation is 1. Max-entropy agents
        # settle below the bound (SAC near 0.91), so ask for a margin.
        (
            Query(
                "a(1)",
                "action",
                1.0,
                1.0,
                {"no learning": 0.0, "policy gradient reversed": -1.0},
                margin=True,
            ),
        ),
    ),
    Probe(
        "coupling",
        SignedActionEnv,
        # The action must follow the observation's sign, and the critic must
        # value what the actor earns. Max-entropy agents settle below the
        # bound (SAC near 0.91), so both are margins.
        (
            Query(
                "a(+1)",
                "action",
                1.0,
                1.0,
                {"no learning": 0.0, "observation ignored or reversed": -1.0},
                margin=True,
            ),
            Query(
                "a(-1)",
                "action",
                -1.0,
                -1.0,
                {"no learning": 0.0, "observation ignored or reversed": 1.0},
                margin=True,
            ),
            Query(
                "V(+1)",
                "value",
                1.0,
                1.0,
                {"policy ignores the observation": 0.0, "reversed": -1.0},
                margin=True,
            ),
            Query(
                "V(-1)",
                "value",
                -1.0,
                1.0,
                {"policy ignores the observation": 0.0, "reversed": -1.0},
                margin=True,
            ),
        ),
    ),
)

DISCRETE_PROBES = (
    *_value_probes(discrete=True),
    Probe(
        "advantage",
        AdvantagePolicyLossPolicyUpdateEnv,
        # Reward = 1 - action; one observation, 0.
        (
            Query("Q(0, a=0)", "q0", 0.0, 1.0, {"actions swapped": 0.0}),
            Query(
                "Q(0, a=0) - Q(0, a=1)",
                "gap01",
                0.0,
                1.0,
                {"no learning": 0.0, "actions swapped": -1.0},
                margin=True,
            ),
        ),
    ),
    Probe(
        "coupling",
        PolicyAndValueEnv,
        # Reward 1 when the action equals the observation (0 or 1).
        (
            Query("Q(0, a=0)", "q0", 0.0, 1.0, {"observation ignored": 0.5}),
            Query("Q(1, a=1)", "q1", 1.0, 1.0, {"observation ignored": 0.5}),
            Query(
                "Q(0, a=0) - Q(0, a=1)",
                "gap01",
                0.0,
                1.0,
                {"observation ignored": 0.0, "reversed": -1.0},
                margin=True,
            ),
            Query(
                "Q(1, a=1) - Q(1, a=0)",
                "gap10",
                1.0,
                1.0,
                {"observation ignored": 0.0, "reversed": -1.0},
                margin=True,
            ),
        ),
    ),
)

AGENTS: Mapping[str, tuple[type, tuple[Probe, ...]]] = {
    "SAC": (SAC, CONTINUOUS_PROBES),
    "PPO": (PPO, CONTINUOUS_PROBES),
    "DQN": (DQN, DISCRETE_PROBES),
    "PQN": (PQN, DISCRETE_PROBES),
}

NOT_PROBED: Mapping[str, str] = {
    "DreamerV3": "probed in tests/agents/test_probing.py with its own adaptor",
    "TDMPC2": "probed in tests/agents/TDMPC2/test_tdmpc2_probes.py",
    "TDMPC2MultiTask": "no probe yet",
    "REDQ": "no probe yet (skipped as too slow in tests/agents/test_probing.py)",
    "ASAC": "average reward: needs differential-value answers; no probe yet",
    "APO": "average reward: needs differential-value answers; no probe yet",
    "AVG": "no probe yet (skipped as too slow in tests/agents/test_probing.py)",
    "APG": "needs probe environments that pass gradients through transitions",
}
"""Every other exported agent, with the reason it has no known-answer probe
here yet."""

# Budget (env steps) and per-query tolerance, from calibrate.py: the ladder
# on seeds 1000-1031, the tolerances re-derived on those plus 2000-2031 (and
# the ladder on both for the two probes whose first choice failed there),
# then checked on seeds 3000-3031 (see its docstring for the rule).
CALIBRATION: Mapping[tuple[str, str], tuple[int, Mapping[str, float]]] = {
    ("SAC", "value"): (1250, {"V(0)": 0.02}),
    ("SAC", "backprop"): (1250, {"V(0)": 0.037, "V(1)": 0.02}),
    ("SAC", "discounting"): (10000, {"V(0)": 0.093, "V(1)": 0.02}),
    ("SAC", "advantage"): (5000, {"a(1)": 0.333}),
    ("SAC", "coupling"): (
        5000,
        {"a(+1)": 0.333, "a(-1)": 0.341, "V(+1)": 0.329, "V(-1)": 0.342},
    ),
    ("PPO", "value"): (1250, {"V(0)": 0.02}),
    ("PPO", "backprop"): (1250, {"V(0)": 0.059, "V(1)": 0.05}),
    ("PPO", "discounting"): (10000, {"V(0)": 0.077, "V(1)": 0.067}),
    ("PPO", "advantage"): (1250, {"a(1)": 0.498}),
    ("PPO", "coupling"): (
        1250,
        {"a(+1)": 0.5, "a(-1)": 0.5, "V(+1)": 0.486, "V(-1)": 0.495},
    ),
    ("DQN", "value"): (1250, {"V(0)": 0.02}),
    ("DQN", "backprop"): (1250, {"V(0)": 0.02, "V(1)": 0.02}),
    ("DQN", "discounting"): (1250, {"V(0)": 0.02, "V(1)": 0.02}),
    ("DQN", "advantage"): (1250, {"Q(0, a=0)": 0.02, "Q(0, a=0) - Q(0, a=1)": 0.5}),
    ("DQN", "coupling"): (
        1250,
        {
            "Q(0, a=0)": 0.039,
            "Q(1, a=1)": 0.02,
            "Q(0, a=0) - Q(0, a=1)": 0.499,
            "Q(1, a=1) - Q(1, a=0)": 0.498,
        },
    ),
    ("PQN", "value"): (80000, {"V(0)": 0.02}),
    ("PQN", "backprop"): (5000, {"V(0)": 0.02, "V(1)": 0.02}),
    ("PQN", "discounting"): (5000, {"V(0)": 0.02, "V(1)": 0.02}),
    ("PQN", "advantage"): (80000, {"Q(0, a=0)": 0.02, "Q(0, a=0) - Q(0, a=1)": 0.5}),
    ("PQN", "coupling"): (
        5000,
        {
            "Q(0, a=0)": 0.02,
            "Q(1, a=1)": 0.02,
            "Q(0, a=0) - Q(0, a=1)": 0.499,
            "Q(1, a=1) - Q(1, a=0)": 0.497,
        },
    ),
}


def read(agent_name: str, probe: Probe, state: Any) -> dict[str, np.ndarray]:
    """Each query's reading, one entry per seed."""
    family = FAMILY[agent_name]
    nets = _networks(state)
    return {
        q.name: np.asarray(jax.vmap(lambda n, q=q: q.read(family)(n, q.obs))(nets))
        for q in probe.queries
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Verdict:
    passed: bool
    seeds_within: int
    seeds_run: int
    readings: dict[str, np.ndarray]
    report: str


def _within(
    probe: Probe, readings: Mapping[str, np.ndarray], tolerances: Mapping[str, float]
) -> np.ndarray:
    ok = np.ones(len(next(iter(readings.values()))), dtype=bool)
    for q in probe.queries:
        ok &= q.within(readings[q.name], tolerances[q.name])
    return ok


def _report(
    probe: Probe, readings: Mapping[str, np.ndarray], tolerances: Mapping[str, float]
) -> str:
    lines = []
    for q in probe.queries:
        r = readings[q.name]
        tol = tolerances[q.name]
        criterion = (
            f"{'>=' if q.truth > 0 else '<='} {np.sign(q.truth) * tol:.3f}"
            if q.margin
            else f"{q.truth:.3f} +- {tol:.3f}"
        )
        line = (
            f"  {q.name}: wants {criterion}; readings {np.array2string(r, precision=3)}"
        )
        median = float(np.median(r))
        if not q.within(np.array([median]), tol)[0]:
            nearest = min(q.wrong, key=lambda k: abs(q.wrong[k] - median))
            line += (
                f"; median {median:.3f} is nearest the wrong answer "
                f"'{nearest}' ({q.wrong[nearest]:.3f})"
            )
        lines.append(line)
    return "\n".join(lines)


def check(
    agent_name: str, probe: Probe, budget: int, tolerances: Mapping[str, float]
) -> Verdict:
    """Train, read and judge with the two-stage rule."""
    agent_cls, _ = AGENTS[agent_name]
    readings = read(
        agent_name, probe, train(agent_cls, probe.env, STAGE_1_SEEDS, budget)
    )
    within = int(_within(probe, readings, tolerances).sum())
    seeds_run = len(STAGE_1_SEEDS)
    if 4 < within < 7:
        more = read(
            agent_name, probe, train(agent_cls, probe.env, STAGE_2_SEEDS, budget)
        )
        within += int(_within(probe, more, tolerances).sum())
        seeds_run += len(STAGE_2_SEEDS)
        readings = {k: np.concatenate([readings[k], more[k]]) for k in readings}
        passed = within >= 13
    else:
        passed = within >= 7
    return Verdict(
        passed,
        within,
        seeds_run,
        readings,
        f"{agent_name} on the {probe.name} probe: {within} of {seeds_run} seeds "
        f"within tolerance (budget {budget} steps)\n"
        + _report(probe, readings, tolerances),
    )
