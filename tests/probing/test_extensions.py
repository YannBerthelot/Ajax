"""Extensions: every extension phase an agent claims to support, made an
executable spec by test-only extensions of known effect (P3), and the numbers
the real expert extensions produce (Q7). Learning cells are judged by the last
test of the module (``verdict``); exact cells (metrics, counters) assert on 2
seeds; construction checks (a phase no agent folds rejected) train nothing.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import probing_environments.gymnax_envs as discrete
import pytest
from probing_environments.gymnax_envs import continuous_actions as continuous

from ajax import PPO, PQN
from ajax.extensions.base import PHASES, Extension
from ajax.extensions.expert import OnlineBC
from ajax.extensions.pretrain import MCPretrain
from ajax.extensions.target_mods import IBRL, CriticBlend

from . import agents, envs, runs
from . import readouts as R
from .agents import GAMMA
from .verdict import Case, Query, check, params, xparam

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "eaa2f8d2a22b"  # verdict.digest(CASES): every answer, pinned


# --- P3: known-effect extensions ----------------------------------------------
# One extension per phase, under a name no agent whitelists. The phase
# batches are undocumented and differ by agent, so the extensions branch on
# the keys each agent passes; a documented schema should replace them.


@dataclasses.dataclass(frozen=True)
class TargetShift(Extension):
    name: str = "p3_target_shift"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target + 0.5


@dataclasses.dataclass(frozen=True)
class TerminalSafeShift(Extension):
    """+ gamma (1 - terminated). Replay agents pass only ``dones``, all pass
    (T, n_envs) flags beside (T, n_envs, 1) targets: no outer sum."""

    name: str = "p3_terminal_safe_shift"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        done = batch["terminated"] if "terminated" in batch else batch["dones"]
        if jnp.size(done) != target.size:
            raise RuntimeError(f"done flags {jnp.shape(done)}, target {target.shape}")
        done = jnp.asarray(done, target.dtype).reshape(target.shape)
        return target + batch["gamma"] * (1.0 - done)


@dataclasses.dataclass(frozen=True)
class CriticPull(Extension):
    """+ 100 mean(Q^2): a critic at w (Q - y)^2 settles at w y / (w + 100).
    DQN and PQN pass their Q-network's ``q_state`` at the differentiated
    params."""

    name: str = "p3_critic_pull"

    def critic_loss(self, agent_state, ext_state, batch, ctx):
        obs = batch["observations"]
        if "critic_params" in batch:
            if "actions" in batch:
                obs = jnp.concatenate([obs, batch["actions"]], axis=-1)
            q = batch["critic_state"].apply_fn(batch["critic_params"], obs)
        else:
            q_values = batch["q_state"].apply_fn(batch["q_state"].params, obs).q_values
            a = jnp.asarray(batch["actions"], jnp.int32)
            q = jnp.take_along_axis(q_values, a.reshape(q_values.shape[0], -1), -1)
        return 100.0 * jnp.mean(q**2)


A0 = -0.5


@dataclasses.dataclass(frozen=True)
class ActorPull(Extension):
    """+ 100 mean((a - A0)^2) on the deterministic action, from half the
    budget when ``gated``. PPO passes no observations (``probe_obs`` stands
    in), SAC only the pre-squash ``pi_loc``."""

    gated: bool = False
    probe_obs: tuple[float, ...] = (0.0,)
    name: str = "p3_actor_pull"

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        if "actor_params" in batch:
            obs = batch.get("observations")
            obs = jnp.asarray([self.probe_obs], jnp.float32) if obs is None else obs
            a = batch["actor_state"].apply_fn(batch["actor_params"], obs).mean()
        else:
            a = jnp.tanh(batch["pi_loc"])
        term = 100.0 * jnp.mean((a - A0) ** 2)
        if self.gated:
            term = jnp.where(ctx.step >= ctx.total_steps // 2, term, 0.0)
        return term


@dataclasses.dataclass(frozen=True)
class ActionOverride(Extension):
    name: str = "p3_action_override"

    def action(self, agent_state, ext_state, obs, rng, ctx):
        return jnp.zeros(obs.shape[:-1] + (1,))


@dataclasses.dataclass(frozen=True)
class EvalActionOverride(Extension):
    name: str = "p3_eval_action_override"

    def eval_action(self, agent_state, ext_state, obs, rng, ctx):
        return jnp.zeros(obs.shape[:-1] + (1,))


def _steps(state: Any) -> Any:
    """Actor plus critic optimiser steps (traced inside the hooks)."""
    steps = {k: v for k, v in R.optimizer_steps(state).items() if k != "alpha"}
    return sum(jnp.asarray(v, jnp.int32) for v in steps.values())


@dataclasses.dataclass(frozen=True)
class MarkThenCount(Extension):
    """``pretrain`` adds 1000, each ``post_update`` 1 and records the steps
    it sees: the final ones when it runs after the update."""

    name: str = "p3_pretrain_mark_then_count"

    def init_state(self, agent_state, rng):
        return {"count": jnp.int32(0), "steps_seen": jnp.int32(-1)}

    def pretrain(self, agent_state, ext_state, ctx):
        return agent_state, {**ext_state, "count": ext_state["count"] + 1000}

    def post_update(self, agent_state, ext_state, ctx):
        seen = _steps(agent_state)
        return agent_state, {"count": ext_state["count"] + 1, "steps_seen": seen}


@dataclasses.dataclass(frozen=True)
class KnownMetric(Extension):
    name: str = "p3_known_metric"

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {"P3/known metric": jnp.asarray(7.0, jnp.float32)}


def constant_expert(obs: jax.Array) -> jax.Array:
    return jnp.full(obs.shape[:-1] + (1,), 0.3, jnp.float32)


MC_KW: dict[str, Any] = {"n_mc_steps": 16, "n_mc_episodes": 32, "n_steps": 500}
MC = MCPretrain(constant_expert, use_online_light=False, **MC_KW)


class GradientValueEnv(continuous.ValueLossOrOptimizerEnv):
    supports_transition_gradients = True  # what APG requires
    transition_gradients_enabled = True


# (continuous, discrete) package probes: the 1-step value probe (reward 1);
# the chain (0 then 1: V(1) = 1, V(0) = gamma).
ENV = {
    "value": (continuous.ValueLossOrOptimizerEnv, discrete.ValueLossOrOptimizerEnv),
    "chain": (continuous.RewardDiscountingEnv, discrete.RewardDiscountingEnv),
}
CELLS = {  # cell: probe, extensions (continuous, discrete)
    "A": ("chain", ((TargetShift(), ActorPull()), (TargetShift(),))),
    "B": ("value", ((CriticPull(),), (CriticPull(),))),
    "E7": ("value", ((MC,), ())),
    "F": ("chain", ((TerminalSafeShift(),), (TerminalSafeShift(),))),
    "G": ("value", ((ActorPull(gated=True),), ())),
}


def _p3_read(agent: str, cell: str) -> Callable:
    """V(0), V(1) by the agent's own readout (SAC's chain V(0) without the
    entropy bonus gamma alpha H(pi(.|1)) of the step to 1), the action and
    phi*."""

    def one(n: R.Nets) -> dict:
        v = {"V(0)": R.value(agent, n, 0.0), "V(1)": R.value(agent, n, 1.0)}
        if agent == "SAC" and cell in "AF":
            v["V(0)"] -= GAMMA * R.alpha(n) * R.entropy(R.pi(n, 1.0), R.KEY)
        if agent in ("DQN", "PQN"):
            return v
        v["a(0)"] = R.action(n, 0.0)
        if cell == "E7":
            v["phi*(0, a_E)"] = R.critic(n, 0.0, 0.3, params=n.extra)
        return v

    def read(run: runs.Run) -> dict:
        extra = run.state.expert_critic_params if cell == "E7" else None
        return R.per_seed(one, R.nets(run.state, extra))

    return read


@functools.cache
def p3_readings(cell: str, agent: str) -> Callable:
    probe, exts = CELLS[cell]
    kind = int(agent in ("DQN", "PQN"))

    def build() -> Any:
        return agents.make(
            agent, *envs.package(ENV[probe][kind]), extensions=exts[kind]
        )

    return runs.readings(build, _p3_read(agent, cell))


def _default(cls: type, name: str) -> float:
    return inspect.signature(cls).parameters[name].default


LAMBDA = {"SAC": 0.0, "DQN": 0.0, "PPO": _default(PPO, "gae_lambda")}
LAMBDA["PQN"] = _default(PQN, "q_lambda")


def chain_v0(agent: str, v1: float, shift: float) -> float:
    """V(0) when V(1) = v1 and ``shift`` lands after the agent's estimator,
    whose lambda-return mixes V(1) with the unshifted Monte Carlo return."""
    return shift + GAMMA * ((1.0 - LAMBDA[agent]) * v1 + LAMBDA[agent])


def p3_queries(test: str, agent: str) -> tuple[Query, ...]:
    v0 = functools.partial(chain_v0, agent)
    ignored = {"shift ignored": v0(1.0, 0.0)}
    if test == "E1":
        wrong = {"shift applied twice": v0(2.0, 1.0)}
        if agent == "PPO":
            wrong["shift folded as a reward, before GAE"] = 0.5 + 1.5 * GAMMA
        v1 = Query("V(1)", 1.5, {"shift ignored": 1.0, "shift applied twice": 2.0})
        return v1, Query("V(0)", v0(1.5, 0.5), ignored | wrong)
    if test == "E9":  # PPO's V(0) misses a missing flag by 0.019: V(1) sees it
        wrong = {"shift applied twice": v0(1.0, 2 * GAMMA)}
        if agent != "PPO":
            wrong["done flag missing"] = v0(1.0 + GAMMA, GAMMA)
        v1 = Query("V(1)", 1.0, {"done flag missing": 1.0 + GAMMA})
        return v1, Query("V(0)", v0(1.0, GAMMA), ignored | wrong)
    w = 0.5 * _default(PPO, "vf_coef") if agent == "PPO" else 1.0
    loc = {"pull on the pre-squash loc": float(np.tanh(-0.5))} if agent == "SAC" else {}
    single = {
        "E2": Query("V(0)", w / (w + 100.0), {"term not folded or no gradient": 1.0}),
        "E3": Query("a(0)", -0.5, {"no pull (max-entropy or initial mean)": 0.0} | loc),
        "E3-gated": Query("a(0)", -0.5, {"gate never opens (ctx.step 0)": 0.0}),
        "E7": Query("phi*(0, a_E)", 1.0, {"MCPretrain unbound or untrained": 0.0}),
    }
    return (single[test],)


# "test-agent": cell, budget, tolerances calibrated on seeds 1000-1031 and
# 2000-2031, certified 32/32 on 3000-3031 unless noted (none: a defect).
P3_CAL = {
    "E1-SAC": ("A", 2500, ()),
    "E1-PPO": ("A", 5000, (0.1, 0.1)),  # no rung qualifies: the best, at the ceiling
    "E1-DQN": ("A", 1250, (0.02, 0.02)),
    "E1-PQN": ("A", 5000, (0.02, 0.02)),
    "E9-SAC": ("F", 2500, ()),
    "E9-PPO": ("F", 5000, (0.051, 0.068)),  # certified 31/32
    "E9-DQN": ("F", 1250, (0.02, 0.02)),
    "E9-PQN": ("F", 5000, (0.02, 0.02)),
    "E2-SAC": ("B", 1250, ()),
    "E2-PPO": ("B", 1250, (0.02,)),
    "E2-DQN": ("B", 1250, ()),
    "E2-PQN": ("B", 80_000, ()),
    "E3-SAC": ("A", 2500, (0.01894,)),  # half the gap to tanh(-0.5), worst 0.0058
    "E3-PPO": ("A", 5000, (0.045,)),
    "E3-gated-SAC": ("G", 5000, ()),
    "E3-gated-PPO": ("G", 1250, (0.02,)),
    "E7-SAC": ("E7", 200, (0.02,)),
}


for i, (cell, budget, tol) in P3_CAL.items():
    test, agent = i.rsplit("-", 1)
    query, reads = p3_queries(test, agent), p3_readings(cell, agent)
    CASES[f"p3-{i}"] = Case(f"p3-{i}", query, reads, budget, tol)


def _exact(names: Any) -> list:
    return [xparam(a) for a in names]


PHASE_EXTENSIONS = {
    "on_target": TargetShift,
    "critic_loss": CriticPull,
    "actor_loss": ActorPull,
    "action": ActionOverride,
    "eval_action": EvalActionOverride,
    "post_update": MarkThenCount,
    "eval_metrics": KnownMetric,
}


@pytest.mark.parametrize("phase", sorted(PHASE_EXTENSIONS))
def test_p3_extension_implements_its_phase(phase: str) -> None:
    """Exactly its phase (the counter also ``pretrain``), under a p3_ name: a
    misspelt hook would be a no-op that passes for the "ignored" defects."""
    ext = PHASE_EXTENSIONS[phase]()
    extra = {"pretrain"} if phase == "post_update" else set()
    assert ext.implemented_phases() == {phase} | extra and ext.name.startswith("p3_")


def _rejected(agent: str, ext: Extension) -> None:
    env, p = envs.package(ENV["value"][int(agent in ("DQN", "PQN"))])
    with pytest.raises(ValueError, match=ext.name):
        agents.make(agent, env, p, extensions=(ext,))


@pytest.mark.parametrize("agent", ("SAC", "PPO", "DQN", "PQN"))
def test_p3_e4_action_override_raises_at_construction(agent: str) -> None:
    """No agent folds ``action`` into its own policy (SAC only into its action
    pipeline's slots): an override is rejected, not silently ignored."""
    _rejected(agent, ActionOverride())


@pytest.mark.parametrize("agent", ("SAC", "PPO", "DQN", "PQN"))
def test_p3_e5_eval_action_override_raises_at_construction(agent: str) -> None:
    """No agent folds ``eval_action``: an override is rejected at construction."""
    _rejected(agent, EvalActionOverride())


# The counter, metric and E10 cells: every agent that trains on an env, on
# the bookkeeping preset (the world models count their own post_update folds
# in tests/agents/TDMPC2 and tests/agents/DreamerV3).
BOOKKEEPING = agents.PRESETS["bookkeeping"]
TRAINED = tuple(a for a in BOOKKEEPING if a not in ("DreamerV3", "TDMPC2"))
VALUE_C, VALUE_D = ENV["value"]
SMALL_ENV = {"DQN": VALUE_D, "PQN": VALUE_D, "UDRL": VALUE_D, "APG": GradientValueEnv}


def small(agent: str, n_envs: int, exts: tuple, cls: type | None = None) -> Any:
    env, p = envs.package(SMALL_ENV.get(agent, VALUE_C))
    return agents.make(
        agent, env, p, preset="bookkeeping", cls=cls, n_envs=n_envs, extensions=exts
    )


@pytest.mark.parametrize("agent", _exact(TRAINED))
def test_p3_eval_metrics_reach_the_log(agent: str) -> None:
    """A constant metric, 7.0, reaches every seed's evaluation log."""
    run = runs.train(small(agent, 1, (KnownMetric(),)), (0, 1), 1000, 250)
    r = run.logged("P3/known metric")
    assert np.all(np.abs(r - 7.0) <= 1e-4), r


@pytest.mark.parametrize("agent", _exact(("PPO", "DQN", "PQN")))
def test_p3_e7_unbound_mc_pretrain_raises(agent: str) -> None:
    """MCPretrain needs the agent's networks and env to build phi*: an agent
    that does not bind it must raise rather than run without phi*."""
    env, p = envs.package(VALUE_D if agent in ("DQN", "PQN") else VALUE_C)
    with pytest.raises(ValueError):
        runs.train(agents.make(agent, env, p, extensions=(MC,)), (0, 1), 128)


@functools.cache
def p3_counter(agent: str) -> list[tuple[np.ndarray, ...]]:
    """Per leg (1000 steps at 2 envs, then 1000 more resumed): the count,
    the steps the last post_update saw, the final steps."""
    run = runs.train(small(agent, 2, (MarkThenCount(),)), (0, 1), 1000)
    out = []
    for leg in range(2):
        run = runs.resume(run, 1000) if leg else run  # read before: it donates
        ext = run.state.ext_state[0]
        steps = (ext["count"], ext["steps_seen"], _steps(run.state))
        out.append(tuple(map(np.asarray, steps)))
    return out


@pytest.mark.parametrize("agent", _exact(TRAINED))
def test_p3_e8_post_update_counts_update_iterations(agent: str) -> None:
    """``pretrain`` marks 1000, each ``post_update`` adds 1: 1000 plus the
    update iterations (after learning starts), continued on resume."""
    cfg, want = BOOKKEEPING[agent], 1000
    per = cfg.get("n_steps", cfg.get("horizon", 1))  # env steps per iteration
    for leg, (count, _, _) in enumerate(p3_counter(agent)):
        its = runs.iterations(agent, 2, 1000, per, 1000 * leg)
        want += sum(t >= cfg.get("learning_starts", 0) for t in its)
        np.testing.assert_array_equal(count, want)


@pytest.mark.parametrize("agent", _exact(TRAINED))
def test_p3_e8_post_update_runs_after_the_update(agent: str) -> None:
    """As the hook after the update (extensions/base.py:204-209), the last
    ``post_update`` sees the final optimiser steps, on each leg."""
    for _, seen, final in p3_counter(agent):
        np.testing.assert_array_equal(seen, final)


@pytest.mark.parametrize("agent", _exact(BOOKKEEPING))
def test_p3_e10_undeclared_phase_raises_at_construction(agent: str) -> None:
    """An agent declaring every phase but ``action`` rejects an extension
    implementing it at construction, naming the phase."""
    cls = agents.agent_class(agent)
    phases = frozenset(PHASES) - {"action"}
    restricted = type(cls.__name__, (cls,), {"supported_extension_phases": phases})
    with pytest.raises(ValueError, match="action"):
        small(agent, 1, (ActionOverride(),), restricted)


@dataclasses.dataclass(frozen=True)
class AgentExpert:
    """An agent's ``expert_policy`` in both conventions: SAC's evaluation steps
    it with a state (evaluate.py:421), REDQ and APO call it plainly."""

    policy: Callable[[jax.Array], jax.Array]

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1), jnp.float32)

    def __call__(self, *args: jax.Array) -> Any:
        a = self.policy(args[-1])
        return (a, args[0]) if len(args) == 2 else a


STATELESS = "SAC's evaluation compares its policy with the expert by stepping the expert with a state (step_environment_expert, evaluate.py:406), unlike its own rollout and the collector, so a stateless expert raises TypeError at trace time; right: training runs"
EXPERTS = [xparam(constant_expert, STATELESS, TypeError, "stateless")]
EXPERTS += [pytest.param(AgentExpert(constant_expert), id="stateful")]
PREFILLED = ("action", "a_expert", "next_a_expert", "is_expert")


@pytest.mark.parametrize("n_envs", [1, 2])
@pytest.mark.parametrize("expert", EXPERTS)
def test_p3_e11_sac_expert_prefill_trains(expert: Any, n_envs: int) -> None:
    """SAC with a constant expert (0.3) and a 64-step expert prefill trains;
    each env's first 64 rows are the expert's, its actions at s and s'."""
    kw: dict = {"expert_policy": expert, "expert_buffer_n_steps": 64}
    run = runs.train(
        agents.make("SAC", *envs.package(VALUE_C), n_envs=n_envs, **kw), (0,), 500
    )
    for k, rows in R.replay_rows(run.state, PREFILLED).items():
        want = 1.0 if k == "is_expert" else 0.3
        np.testing.assert_allclose(rows[:, :, :64], want, atol=1e-6, err_msg=k)


# --- Q7: the real expert-critic extensions -----------------------------------
# On the chain with a constant expert a_E = 0.8: MCPretrain's phi* and value
# range (budget 0, reward_scale 2), CriticBlend's target, IBRL's gap, OnlineBC's
# value weighting; then the cloning pre-training (budget 0) on SignedActionEnv,
# expert 0.5 s, and on its twin observed at s + 2. The cloning cases come last:
# the pre-training calls jax.clear_caches() while tracing (cloning.py:476).


@dataclasses.dataclass(frozen=True)
class ConstantExpert:
    """a_E whatever the input's shape; frozen, so the jitted MC pre-training
    (a static argument) compiles once per value."""

    action: float

    def __call__(self, obs: jax.Array) -> jax.Array:
        return jnp.full(obs.shape[:-1] + (1,), self.action, jnp.float32)


@dataclasses.dataclass(frozen=True)
class ShiftedHalfExpert:
    shift: float  # a_E(s) = 0.5 (s - shift)

    def __call__(self, obs: jax.Array) -> jax.Array:
        return 0.5 * (obs - self.shift)


class ActionChainEnv(continuous.RewardDiscountingEnv):
    """The chain paying clip(a, -1, 1) at both steps: Q(1, a) = a."""

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
        state = type(state)(x=jnp.float32(state.time + 1), time=state.time + 1)
        reward = jnp.clip(jnp.squeeze(action), -1.0, 1.0)
        obs, done = self.get_obs(state), self.is_terminated(state, params)
        info = {"discount": self.discount(state, params)}
        return *jax.lax.stop_gradient((obs, state)), reward, done, info


class ShiftedSignedEnv(envs.SignedActionEnv):
    """Observed at s + 2: 1 and 3, mean 2 and std 1, standardised -1 and +1."""

    def get_obs(self, state: Any, params: Any = None, key: Any = None) -> Any:
        return jnp.array([state.x + 2.0])


A_E, RS = 0.8, 2.0  # the constant expert's action; MCPretrain's reward_scale
EXPERT, CHAIN = ConstantExpert(A_E), ENV["chain"][0]
ONE = 1e6  # critic_warmup_frac: the blend and clone weights stay at 1 (1e-6)


def mc(light: bool = False) -> MCPretrain:
    """512 MC rows, two regression batches (fewer than 256 rows divide by zero,
    modules/pretrain.py:178-213), 250 regression steps."""
    return MCPretrain(EXPERT, use_online_light=light, **MC_KW | {"n_steps": 250})


def q7_readings(agent: str, env_cls: type, read: Callable, exts=(), **kw) -> Any:
    """``read`` maps a seed's nets (``extra``: phi*) to readings; MCPretrain's
    value range joins them when it ran."""

    def build() -> Any:
        return agents.make(agent, *envs.package(env_cls), extensions=exts, **kw)

    def go(run: runs.Run) -> dict:
        s = run.state
        out = R.per_seed(read, R.nets(s, getattr(s, "expert_critic_params", None)))
        box = {k: getattr(s, f"expert_{k}", None) for k in ("v_min", "v_max")}
        return out | {k: np.reshape(v, -1) for k, v in box.items() if v is not None}

    return runs.readings(build, go)


def _phi(n: R.Nets) -> dict:
    return {f"phi*({x}, a_E)": R.critic(n, x, A_E, params=n.extra) for x in (0, 1)}


def _ibrl(n: R.Nets) -> dict:
    q = {f"Q(1, {a})": R.critic(n, 1.0, a) for a in (-0.5, 0.5)}
    return q | {"Q(0, 0)": R.critic(n, 0.0, 0.0)}


def _blend(n: R.Nets) -> dict:
    return {f"Q({x})": R.value("SAC", n, x) for x in (0, 1)}


def _bc(n: R.Nets) -> dict:
    return {"a(1)": R.action(n, 1.0), "a_E - a(0)": A_E - R.action(n, 0.0)}


_UNSCALED = "MC returns left unscaled (modules/pretrain.py:137)"
_G99 = {"gamma bound from the 0.99 default": 0.99 * RS}
_LAST = {"range of the last batch only, envs in lockstep": RS}
PHI = (
    Query("phi*(0, a_E)", GAMMA * RS, {_UNSCALED: GAMMA} | _G99),
    Query("phi*(1, a_E)", RS, {_UNSCALED: 1.0}),
)
V_RANGE = (
    Query("v_min", GAMMA * RS, {_UNSCALED: GAMMA} | _G99 | _LAST),
    Query("v_max", RS, {_UNSCALED: 1.0}),
)
# The light case was calibrated on SafeSAC, SAC with an extra critic head
# its losses never read; phi* and the value range are MCPretrain's own.
for light in (False, True):
    reads = q7_readings("SAC", CHAIN, _phi, (mc(light),), reward_scale=RS)
    note = "regression steps (250, 500): 250, worst 6e-6; cert 32/32"
    cid = f"q7-mc{'-light' * light}-SAC"
    CASES[cid] = Case(cid, PHI + V_RANGE, reads, 0, (0.02,) * 4, note=note)
CASES["q7-mc-lockstep-SAC"] = Case(
    "q7-mc-lockstep-SAC",
    V_RANGE,
    q7_readings("SAC", CHAIN, _phi, (mc(),), reward_scale=RS, n_envs=256),
    0,
)
_V_E, _S = "docstring, V(terminal) = 0", "blend of V_E(s') alone"
CASES["q7-blend-SAC"] = Case(
    "q7-blend-SAC",
    (
        Query("Q(0)", GAMMA, {f"{_S} (today)": 1.0, _V_E: 1.0}),
        Query(
            "Q(1)", 1.0, {f"{_S}, s' the reset observation (today)": GAMMA, _V_E: 0.0}
        ),
    ),
    q7_readings(
        "SAC",
        CHAIN,
        _blend,
        (mc(), CriticBlend(expert_policy=EXPERT, critic_warmup_frac=ONE)),
    ),
    1250,
    defect="CriticBlend's target is (1 - w) y + w V_E(s') with no reward, discount or done mask (target_mods.py:222-257), and a terminal row's s' is the next episode's reset observation (buffers/utils.py:140); right, consistent in time, (Q(0), Q(1)) = (0.62, 1.0), today (1.0, 0.62); the docstring's V(terminal) = 0 would give (1.0, 0.0) (r/gamma convention: owner's call)",
    slow=True,
)
_X02 = "gap not masked at the terminal step (X02, target_mods.py:116)"
IBRL_KW = {"fixed_alpha": True, "alpha_init": 1e-4, "policy_update_start": 2**30}
CASES["q7-ibrl-SAC"] = Case(
    "q7-ibrl-SAC",
    (
        *(Query(f"Q(1, {a})", a, {_X02: a + GAMMA}) for a in (-0.5, 0.5)),
        Query("Q(0, 0)", GAMMA, {"IBRL not applied": 0.0, "gap not discounted": 1.0}),
    ),
    q7_readings(
        "SAC",
        ActionChainEnv,
        _ibrl,
        (IBRL(expert_policy=ConstantExpert(1.0)),),
        **IBRL_KW,
    ),
    1250,
    (0.021, 0.02, 0.083),
    slow=True,
    note="a_E = 1, the best action; actor frozen at init, alpha 1e-4; cert 32/32",
)
_BC0 = {"BC weight 0 at observation 1 (v_min/v_max swapped, or no BC)": 0.0}
_BC1 = {"BC weight 1 at observation 0 (weight stuck at 1, or swapped)": 0.0}
CASES["q7-online_bc-SAC"] = Case(
    "q7-online_bc-SAC",
    (Query("a(1)", A_E, _BC0), Query("a_E - a(0)", A_E, _BC1, margin=True)),
    q7_readings(
        "SAC",
        CHAIN,
        _bc,
        (mc(), OnlineBC(expert_policy=EXPERT, bc_coef=100.0, critic_warmup_frac=ONE)),
        expert_policy=AgentExpert(EXPERT),  # the clone's target (train_SAC.py:597-603)
        expert_buffer_n_steps=0,
        policy_update_start=200,
    ),
    1250,
    (0.04, 0.36),
    slow=True,
    note="weight ((phi* - v_min) / (v_max - v_min))^2, 1 at obs 1, 0 at obs 0; cert 32/32",
)


def cloning(agent: str, shift: float) -> Any:
    """1000 expert steps, 10 epochs, budget 0: APO still takes one update
    (TrainLoop.on_policy), kept to one step; SAC's expert prefill off (E11)."""
    kw = {"SAC": {"expert_buffer_n_steps": 0}, "APO": {"n_epochs": 1}}.get(agent, {})
    kw |= {"pre_train_n_steps": 1000, "actor_cloning_epochs": 10}
    env_cls = ShiftedSignedEnv if shift else envs.SignedActionEnv

    def read(n: R.Nets) -> dict:
        return {f"a({x:g})": R.action(n, x) for x in (shift - 1.0, shift + 1.0)}

    expert = AgentExpert(ShiftedHalfExpert(shift))
    return q7_readings(agent, env_cls, read, expert_policy=expert, **kw)


_SKIP = {"cloning pre-training skipped": 0.0}
_RAW = {"trained on standardised, queried on raw observations": 0.5}
SIGNED = (Query("a(-1)", -0.5, _SKIP), Query("a(1)", 0.5, _SKIP))
SHIFTED = (Query("a(1)", -0.5, _RAW | _SKIP), Query("a(3)", 0.5, _SKIP))
STANDARDISED = "the cloning pre-training standardises its observations (cloning.py:301-304) but seeds the runtime statistics only when obs_norm_info is set (cloning.py:505; None unless normalize_obs_running, interaction.py:1251-1255), so get_pi queries the actor on raw observations (interaction.py:347-350); right a(1) = -0.5, today +0.5 (on {-1, 1} z is within 0.005 of s)"
# Epochs (10, 20): 10, the default, qualifies; cert 32/32. APO's update moves
# its action outwards (at 4 steps per iteration no rung qualified).
CLONED = {"SAC": (0.02, 0.02), "REDQ": (0.02, 0.02), "APO": (0.071, 0.081), "TD3": ()}
for name, shift, queries in (("cloning", 0.0, SIGNED), ("shifted", 2.0, SHIFTED)):
    for agent, tol in CLONED.items():
        CASES[f"q7-{name}-{agent}"] = Case(
            f"q7-{name}-{agent}",
            queries,
            cloning(agent, shift),
            0,
            () if shift else tol,
            STANDARDISED if shift else "",
            slow=True,
        )


# --- every judged case --------------------------------------------------------


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
