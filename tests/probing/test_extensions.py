"""Extensions: every extension phase an agent claims to support, made an
executable spec by test-only extensions of known effect (P3), and the numbers
the real expert extensions produce (Q7). Learning cells are judged by the last
test of the module (``verdict``); exact cells (overrides, metrics, counters,
construction checks) assert on 2 seeds.
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
from .verdict import Case, Query, check, params, xfail, xparam

CASES: dict[str, Case] = {}
ANSWER_DIGEST = "96e66ff4eba8"  # verdict.digest(CASES): every answer, pinned


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
    DQN and PQN pass ``q_state``, not the differentiated params."""

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
class RewardShift(Extension):
    name: str = "p3_reward_shift"

    def on_batch(self, batch, ext_state, ctx):
        if isinstance(batch, dict):
            return {**batch, "rewards": batch["rewards"] + 1.0}
        return batch.replace(reward=batch.reward + 1.0)


def _fixed(obs: jax.Array, value: float, shape: tuple[int, ...]) -> jax.Array:
    """``value`` for every env; ``shape`` () is a discrete action."""
    return jnp.full(obs.shape[:-1] + shape, value, jnp.float32 if shape else jnp.int32)


@dataclasses.dataclass(frozen=True)
class ActionOverride(Extension):
    value: float = 0.3
    shape: tuple[int, ...] = (1,)
    name: str = "p3_action_override"

    def action(self, agent_state, ext_state, obs, rng, ctx):
        return _fixed(obs, self.value, self.shape)


@dataclasses.dataclass(frozen=True)
class EvalActionOverride(Extension):
    value: float = -0.2
    shape: tuple[int, ...] = (1,)
    name: str = "p3_eval_action_override"

    def eval_action(self, agent_state, ext_state, obs, rng, ctx):
        return _fixed(obs, self.value, self.shape)


@dataclasses.dataclass(frozen=True)
class ObsFlip(Extension):
    offset: float = 0.0  # the network sees offset - s: s in {-1, 1} or {0, 1}
    name: str = "p3_obs_flip"

    def on_obs(self, obs, ext_state, ctx):
        return self.offset - obs


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
# the chain (0 then 1: V(1) = 1, V(0) = gamma); the bandit (reward a, or
# 1 - a); the contextual bandit (clip(a) s, or [a == s]).
ENV = {
    "value": (continuous.ValueLossOrOptimizerEnv, discrete.ValueLossOrOptimizerEnv),
    "chain": (continuous.RewardDiscountingEnv, discrete.RewardDiscountingEnv),
    "bandit": (
        envs.symmetric(continuous.AdvantagePolicyLossPolicyUpdateEnv),
        discrete.AdvantagePolicyLossPolicyUpdateEnv,
    ),
    "contextual": (envs.SignedActionEnv, discrete.PolicyAndValueEnv),
}
_OVERRIDE = (ActionOverride(), EvalActionOverride())
_DISCRETE_OVERRIDE = (ActionOverride(1, ()), EvalActionOverride(1, ()))
CELLS = {  # cell: probe, extensions (continuous, discrete), log frequency
    "A": ("chain", ((TargetShift(), ActorPull()), (TargetShift(),)), None),
    "B": ("value", ((CriticPull(),), (CriticPull(),)), None),
    "C": ("bandit", (_OVERRIDE, _DISCRETE_OVERRIDE), 500),
    "D": ("contextual", ((ObsFlip(0.0),), (ObsFlip(1.0),)), None),
    "E7": ("value", ((MC,), ()), None),
    "F": ("chain", ((TerminalSafeShift(),), (TerminalSafeShift(),)), None),
    "G": ("value", ((ActorPull(gated=True),), ()), None),
    "H": ("value", ((RewardShift(),), (RewardShift(),)), None),
}


def _p3_read(agent: str, cell: str) -> Callable:
    """V(0), V(1) by the agent's own readout (SAC's chain V(0) without the
    entropy bonus gamma alpha H(pi(.|1)) of the step to 1), the action and
    the network's own reading of +1, phi*, the train and eval returns."""

    def one(n: R.Nets) -> dict:
        v = {"V(0)": R.value(agent, n, 0.0), "V(1)": R.value(agent, n, 1.0)}
        if agent == "SAC" and cell in "AF":
            v["V(0)"] -= GAMMA * R.alpha(n) * R.entropy(R.pi(n, 1.0), R.KEY)
        if agent in ("DQN", "PQN"):
            q = R.q_values(n, 1.0)
            return v | {"Q_net(1,0) - Q_net(1,1)": q[0] - q[1]}
        v |= {"a(0)": R.action(n, 0.0), "a_net(+1)": R.action(n, 1.0, clip=False)}
        if cell == "E7":
            v["phi*(0, a_E)"] = R.critic(n, 0.0, 0.3, params=n.extra)
        return v

    def read(run: runs.Run) -> dict:
        extra = run.state.expert_critic_params if cell == "E7" else None
        out = R.per_seed(one, R.nets(run.state, extra))
        out["train return"] = run.state.collector_state.episodic_mean_return.reshape(-1)
        return out | {"eval return": run.logged("Eval/episodic mean reward")}

    return read


@functools.cache
def p3_readings(cell: str, agent: str) -> Callable:
    probe, exts, every = CELLS[cell]
    kind = int(agent in ("DQN", "PQN"))

    def build() -> Any:
        return agents.make(
            agent, *envs.package(ENV[probe][kind]), extensions=exts[kind]
        )

    return runs.readings(build, _p3_read(agent, cell), log_every=every)


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
        "on_batch": Query("V(0)", 2.0, {"on_batch ignored": 1.0}),
        "E7": Query("phi*(0, a_E)", 1.0, {"MCPretrain unbound or untrained": 0.0}),
    }
    if test in single:
        return (single[test],)
    # E6: the pair separates the flip ignored (+, +), in the actor loss only
    # (-, -) and at collection only (-, +); discrete agents read Q gaps.
    if agent in ("SAC", "PPO"):
        ret = {"flip in the actor loss only": -1.0, "flip at collection only": -1.0}
        net = {"flip ignored": 1.0, "flip at collection only": 1.0}
        return Query("train return", 1.0, ret, True), Query(
            "a_net(+1)", -1.0, net, True
        )
    ret = {"observation ignored": 0.5, "flip in losses only": 0.0}
    flip = {"flip ignored": -1.0, "flip at collection only": -1.0}
    gap = Query("Q_net(1,0) - Q_net(1,1)", 1.0, flip, True)
    return Query("train return", 1.0, ret), gap


SAC_TARGET = "SAC folds on_target only for extensions named in _TARGET_MOD_NAMES (train_SAC.py:90-97, 370-372, 391-413)"
NO_FOLD = "no fold_{} call anywhere in src/ajax, yet every agent but the world models declares the phase (base.py:56, all phases by default)"
SAC_ORDER = "SAC folds post_update before the update (train_SAC.py:1337 vs 1384); right the last call sees the final optimizer steps (852, 1852), today one update earlier (850, 1850)"
P3_DEFECTS = {  # "test-agent" (or "test-*"): the live defect
    "E1-SAC": f"{SAC_TARGET}; right V(1) 1.5, V(0) 1.43; today 1.00, 0.62",
    "E9-SAC": f"{SAC_TARGET}; right V(0) 1.24; today 0.62",
    "E2-DQN": "DQN's critic_loss batch carries q_state, not the differentiated params (train_DQN.py:392-417): the term has no gradient; right V(0) 0.0099, today 1.00",
    "E2-PQN": "PQN's critic_loss batch carries q_state, not the differentiated params (train_PQN.py:217-243): the term has no gradient; right V(0) 0.0099, today 0.997",
    "E3-gated-SAC": "SAC runs actor_loss with ExtensionContext(step=0) and agent_state=None (train_SAC.py:642-653): the gate never opens; right a(0) -0.5, today -0.03 to 0.03",
    "on_batch-*": NO_FOLD.format("on_batch") + "; right V(0) 2.0, today 1.00",
    "E6-SAC": "(contract pending) SAC folds on_obs in the actor loss only (train_SAC.py:516-524), so the policy learns pi(.|-s) against Q(s, .); right train return > 0 and a_net(+1) < 0, today -0.39 to -0.75 and -0.70 to -0.75",
    "E6-*": "(contract pending) on_obs is folded only in SAC's actor loss (train_SAC.py:524), so the flip is ignored here; right the network acts on the flipped input (a_net(+1) < 0; discrete Q_net(1,0) > Q_net(1,1)), today the raw mapping (PPO a_net(+1) 1.3-2.0; DQN and PQN gap -1.0)",
    "E4-*": NO_FOLD.format("action")
    + " (SAC dispatches action extensions by name, only with an expert_policy: _sac_hooks.py:70-71, 339-347); right train return 0.3 (discrete 0), today the policy's own (SAC 0.02-0.08 at 2000 steps, PPO 1.0, DQN 0.9-1.0, PQN 0.975)",
    "E5-*": NO_FOLD.format("eval_action")
    + " (evaluate_and_log runs the actor, log.py:317-347); right eval return -0.2 (discrete 0), today the policy's own (SAC 0.05-0.07, PPO, DQN, PQN 1.0)",
    "metric-UDRL": "UDRL declares eval_metrics (base.py:56 by default) but never evaluates or logs: train_UDRL.py has no evaluate_and_log or compose_eval_metrics call; right 7.0 in the log, today no record (NaN)",
    "E7-unbound-*": "MCPretrain returns the state unchanged when unbound (pretrain.py:146-157) and only SAC binds it (train_SAC.py:1676-1692); right a ValueError, today a run without phi*",
    "E8-order-SAC": SAC_ORDER,
    "E8-order-SafeSAC": SAC_ORDER,
}
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
    "on_batch-SAC": ("H", 1250, ()),
    "on_batch-PPO": ("H", 1250, ()),
    "on_batch-DQN": ("H", 1250, ()),
    "on_batch-PQN": ("H", 80_000, ()),
    "E6-SAC": ("D", 5000, ()),
    "E6-PPO": ("D", 1250, ()),
    "E6-DQN": ("D", 1250, ()),
    "E6-PQN": ("D", 5000, ()),
    "E7-SAC": ("E7", 200, (0.02,)),
}


def _defect(test: str, agent: str) -> str:
    return P3_DEFECTS.get(f"{test}-{agent}", P3_DEFECTS.get(f"{test}-*", ""))


for i, (cell, budget, tol) in P3_CAL.items():
    test, agent = i.rsplit("-", 1)
    query, reads = p3_queries(test, agent), p3_readings(cell, agent)
    CASES[f"p3-{i}"] = Case(f"p3-{i}", query, reads, budget, tol, _defect(test, agent))


def _exact(test: str, names: Any, raises: Any = AssertionError) -> list:
    return [xparam(a, _defect(test, a), raises) for a in names]


PHASE_EXTENSIONS = {
    "on_obs": ObsFlip,
    "on_batch": RewardShift,
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


@pytest.mark.parametrize("agent", _exact("E4", ("SAC", "PPO", "DQN", "PQN")))
def test_p3_e4_action_override_drives_collection(agent: str) -> None:
    """Action 0.3 (discrete 1) collects a train return of exactly 0.3 (0)."""
    r = p3_readings("C", agent)((0, 1), 2000)["train return"]
    assert np.all(np.abs(r - (0.0 if agent in ("DQN", "PQN") else 0.3)) <= 1e-4), r


@pytest.mark.parametrize("agent", _exact("E5", ("SAC", "PPO", "DQN", "PQN")))
def test_p3_e5_eval_action_override_drives_evaluation(agent: str) -> None:
    """Eval action -0.2 (discrete 1) logs an eval return of exactly -0.2 (0)."""
    r = p3_readings("C", agent)((0, 1), 2000)["eval return"]
    assert np.all(np.abs(r - (0.0 if agent in ("DQN", "PQN") else -0.2)) <= 1e-4), r


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


@pytest.mark.parametrize("agent", _exact("metric", TRAINED))
def test_p3_eval_metrics_reach_the_log(agent: str) -> None:
    """A constant metric, 7.0, reaches every seed's evaluation log."""
    run = runs.train(small(agent, 1, (KnownMetric(),)), (0, 1), 1000, 250)
    r = run.logged("P3/known metric")
    assert np.all(np.abs(r - 7.0) <= 1e-4), r


DID_NOT_RAISE = pytest.fail.Exception


@pytest.mark.parametrize(
    "agent", _exact("E7-unbound", ("PPO", "DQN", "PQN"), DID_NOT_RAISE)
)
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


@pytest.mark.parametrize("agent", _exact("E8-count", TRAINED))
def test_p3_e8_post_update_counts_update_iterations(agent: str) -> None:
    """``pretrain`` marks 1000, each ``post_update`` adds 1: 1000 plus the
    update iterations (after learning starts), continued on resume."""
    cfg, want = BOOKKEEPING[agent], 1000
    per = cfg.get("n_steps", cfg.get("horizon", 1))  # env steps per iteration
    for leg, (count, _, _) in enumerate(p3_counter(agent)):
        its = runs.iterations(agent, 2, 1000, per, 1000 * leg)
        want += sum(t >= cfg.get("learning_starts", 0) for t in its)
        np.testing.assert_array_equal(count, want)


@pytest.mark.parametrize("agent", _exact("E8-order", TRAINED))
def test_p3_e8_post_update_runs_after_the_update(agent: str) -> None:
    """As the hook after the update (extensions/base.py:204-209), the last
    ``post_update`` sees the final optimiser steps, on each leg."""
    for _, seen, final in p3_counter(agent):
        np.testing.assert_array_equal(seen, final)


@pytest.mark.parametrize("agent", _exact("E10", BOOKKEEPING, DID_NOT_RAISE))
def test_p3_e10_undeclared_phase_raises_at_construction(agent: str) -> None:
    """An agent declaring every phase but ``action`` rejects an extension
    implementing it at construction, naming the phase."""
    cls = agents.agent_class(agent)
    phases = frozenset(PHASES) - {"action"}
    restricted = type(cls.__name__, (cls,), {"supported_extension_phases": phases})
    with pytest.raises(ValueError, match="action"):
        small(agent, 1, (ActionOverride(),), restricted)


@xfail(
    "SAC's expert prefill writes 7 keys with one env axis (modules/pretrain.py:622-634) into a buffer whose schema has 9 (a_expert, next_a_expert: buffers/utils.py:100-115): chex AssertionError at trace time (tree structure at n_envs 1, shape prefix (1,) != (2,) at n_envs 2); right: training runs"
)
@pytest.mark.parametrize("n_envs", [1, 2])
def test_p3_e11_sac_expert_prefill_trains(n_envs: int) -> None:
    """SAC with a constant expert and a 64-step expert prefill trains."""
    kw: dict = {"expert_policy": constant_expert, "expert_buffer_n_steps": 64}
    runs.train(
        agents.make("SAC", *envs.package(VALUE_C), n_envs=n_envs, **kw), (0,), 500
    )


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
for agent in ("SAC", "SafeSAC"):  # SafeSAC's multi-head critic, online light on
    reads = q7_readings(agent, CHAIN, _phi, (mc(agent == "SafeSAC"),), reward_scale=RS)
    note = "regression steps (250, 500): 250, worst 6e-6; cert 32/32"
    CASES[f"q7-mc-{agent}"] = Case(
        f"q7-mc-{agent}", PHI + V_RANGE, reads, 0, (0.02,) * 4, note=note
    )
CASES["q7-mc-lockstep-SAC"] = Case(
    "q7-mc-lockstep-SAC",
    V_RANGE,
    q7_readings("SAC", CHAIN, _phi, (mc(),), reward_scale=RS, n_envs=256),
    0,
    defect="MCPretrain takes v_min/v_max from the last regression batch only (modules/pretrain.py:229-237); with 256 envs in lockstep the 512 MC rows are 2 timesteps and that batch is the second, every env at observation 1; right (v_min, v_max) = (1.24, 2.0), today (2.0, 2.0), and OnlineBC's weight is 0 everywhere",
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
TD3_LOC = "TD3's cloning pre-training crashes at trace time: the default NLL loss reads pi.loc (cloning.py:339) and TD3's actor returns its Deterministic wrapper (TD3/networks.py:36), which has none; right: the cell runs"
STANDARDISED = "the cloning pre-training standardises its observations (cloning.py:301-304) but seeds the runtime statistics only when obs_norm_info is set (cloning.py:505; None unless normalize_obs_running, interaction.py:1251-1255), so get_pi queries the actor on raw observations (interaction.py:347-350); right a(1) = -0.5, today +0.5 (on {-1, 1} z is within 0.005 of s)"
# Epochs (10, 20): 10, the default, qualifies; cert 32/32. APO's update moves
# its action outwards (at 4 steps per iteration no rung qualified).
CLONED = {"SAC": (0.02, 0.02), "REDQ": (0.02, 0.02), "APO": (0.071, 0.081), "TD3": ()}
for name, shift, queries in (("cloning", 0.0, SIGNED), ("shifted", 2.0, SHIFTED)):
    for agent, tol in CLONED.items():
        defect = TD3_LOC if agent == "TD3" else STANDARDISED if shift else ""
        raises = AttributeError if agent == "TD3" else AssertionError
        CASES[f"q7-{name}-{agent}"] = Case(
            f"q7-{name}-{agent}",
            queries,
            cloning(agent, shift),
            0,
            () if shift else tol,
            defect,
            raises,
            slow=True,
        )


# --- every judged case --------------------------------------------------------


@pytest.mark.parametrize("case", params(CASES))
def test_learns(case: Case) -> None:
    check(case)
