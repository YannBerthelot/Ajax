"""IBRL's target against its authors' (hengyuan-hu/ibrl 3c457da), on fixed inputs and weights.

Their QAgent.update_critic ran once per form on a fixed batch, in float32 and in float64, its draws kept
(benchmarks/reference/conformance_ibrl.py; the repository has no licence, so only its records are
committed). Ajax computes the target from the same target critics, next actions and draws: TD3's target
with IBRL against the two-critic form, REDQ's (alpha 0, five critics, a subset of two) against the state
form, REDQ's subset draw and IBRL's selection draw taped to the authors' value and selection pairs. The
authors' sampled next action reaches the base target as the action it scores (map_next_actions): TD3's
smoothing is test_td3_sfujim's check. Every row matches within the float32-rounding tolerance
(submaps), and two planted faults fail: IBRL drawing its own next action, and selecting on the pair it
values on. Pre-registered before its first run: the per-row target, that tolerance, the authors'
independent pairs kept (UPSTREAM_BUGS 1), their target actor's dropout and 3-step returns left out.
"""

import functools
from types import SimpleNamespace
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import REDQ, TD3
from ajax.agents.REDQ.REDQ import compute_redq_td_target
from ajax.agents.TD3.TD3 import compute_td3_td_target
from ajax.extensions.base import ExtensionContext
from ajax.extensions.target_mods import IBRL
from tests.conformance import kit, originals, submaps

NET = ("16", "relu", "16", "relu")
AGENTS = {"pair": (TD3, {"num_critics": 2}), "state": (REDQ, {"num_critics": 5})}
#: Ajax's critic layers by the authors' MultiFcQ module index.
LAYERS = {"layers_0": "net.0", "layers_2": "net.2", "model": "net.4"}


@functools.cache
def init_fresh_state(form: str) -> Any:
    agent_class, settings = AGENTS[form]
    agent = agent_class(
        "MountainCarContinuous-v0",
        actor_architecture=("16", "relu"),
        critic_architecture=NET,
        normalize_observations=False,
        **settings,
    )
    return agent.init(jax.random.PRNGKey(0), agent.make_context(1)[1])


def take_authors_critics(params: Any, record: dict, dtype: Any) -> Any:
    """params holding the authors' target critics: their kernels as they are,
    their one bias per net and layer spread over the layer's units."""

    def put(path: Any, leaf: Any) -> jax.Array:
        layer = LAYERS[path[-2].key]
        if path[-1].key == "kernel":
            return jnp.asarray(record[f"in.{layer}.weights"], dtype)
        bias = record[f"in.{layer}.biases"][0]
        return jnp.asarray(np.broadcast_to(bias, leaf.shape), dtype)

    return jax.tree_util.tree_map_with_path(put, params)


def compute_target(
    form: str, record: dict, precision: str, fault: Optional[str]
) -> np.ndarray:
    """Ajax's base target with IBRL's bonus, per row, on the record's inputs."""
    dtype = np.float64 if precision == "f64" else np.float32
    state = init_fresh_state(form)
    critics = state.critic_state
    target_params = take_authors_critics(critics.target_params, record, dtype)
    critics = critics.replace(target_params=target_params)

    def rows(name: str) -> jax.Array:
        return jnp.asarray(record[name], dtype)

    next_observations, gamma = rows("batch.next_state"), float(record["batch.gamma"])
    dones, rewards = rows("batch.dones")[:, None], rows("batch.reward")[:, None]
    authors_actions = rows(f"{precision}.rl_action")
    expert_actions = rows(f"{precision}.bc_action")

    def hand_authors_actions(sampled: jax.Array) -> jax.Array:
        return authors_actions

    inputs = {
        "actor_state": state.actor_state,
        "rng": jax.random.PRNGKey(1),
        "next_observations": next_observations,
        "dones": dones,
        "rewards": rewards,
        "gamma": gamma,
        "reward_scale": 1.0,
        "map_next_actions": hand_authors_actions,
    }
    if form == "pair":
        target, _, next_actions = compute_td3_td_target(
            critic_state=critics,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            **inputs,
        )
        subset = None
    else:
        alpha = jnp.asarray(0.0, dtype)
        target, _, next_actions, subset = compute_redq_td_target(
            critic_states=critics, alpha=alpha, subset_size=2, **inputs
        )
    batch = {
        "next_observations": next_observations,
        "raw_next_observations": next_observations,
        "next_a_expert": None,
        "next_actions": None if fault == "own draw" else next_actions,
        "critic_subset": subset,
        "dones": dones,
        "gamma": gamma,
    }
    ibrl = IBRL(expert_policy=lambda observations: expert_actions)
    context = ExtensionContext(
        step=jnp.asarray(0), rng=jax.random.PRNGKey(2), total_steps=1
    )
    agent_state = SimpleNamespace(actor_state=state.actor_state, critic_state=critics)
    target, _ = ibrl.on_target(agent_state, None, batch, target, context)
    return np.asarray(target[:, 0])


def run(
    form: str, record: dict, precision: str, fault: Optional[str] = None
) -> np.ndarray:
    """compute_target in precision; on the state form, REDQ's subset draw taped to the
    authors' value pair and IBRL's selection draw to their selection pair (fault
    "one pair": to the value pair)."""
    if form == "pair":
        with jax.enable_x64(precision == "f64"):
            return compute_target(form, record, precision, fault)
    with kit.recording() as session:  # the draws' keys, the same in either precision
        compute_target(form, record, "f32", None)
    draws = kit.by_role(session.log)
    [subset_role] = [r for r in draws if r[1].endswith("compute_redq_td_target")]
    [selection_role] = [r for r in draws if r[1].endswith("draw_selection_critics")]
    selection = record["draw.value" if fault == "one pair" else "draw.selection"]
    values = {subset_role: [record["draw.value"]], selection_role: [selection]}
    table = kit.key_table({role: draws[role] for role in values}, values)
    with jax.enable_x64(precision == "f64"), kit.replaying(table) as replay:
        out = compute_target(form, record, precision, fault)
    assert not replay.misses
    return out


@pytest.fixture(scope="module")
def records() -> dict:
    found = originals.committed("hengyuan_hu_ibrl").glob("*.npz")
    return {path.stem: dict(np.load(path)) for path in found}


@pytest.fixture(scope="module")
def ours(records: dict) -> dict:
    return {
        (form, precision): run(form, records[form], precision)
        for form in records
        for precision in ("f32", "f64")
    }


def compare(record: dict, ours: dict, form: str, target: np.ndarray) -> submaps.Report:
    theirs = ({"target": record["f32.target"]}, {"target": record["f64.target"]})
    mine = ({"target": ours[form, "f32"]}, {"target": ours[form, "f64"]})
    nulls = submaps.rounding_null(theirs, mine)
    return submaps.compare({"target": target}, theirs[0], nulls, context=form)


@pytest.mark.parametrize("form", ["pair", "state"])
def test_the_target_matches_the_authors(records: dict, ours: dict, form: str) -> None:
    report = compare(records[form], ours, form, ours[form, "f32"])
    assert not report.failures, str(report)


@pytest.mark.parametrize("form, fault", [("pair", "own draw"), ("state", "one pair")])
def test_a_planted_fault_fails_the_check(
    records: dict, ours: dict, form: str, fault: str
) -> None:
    faulty = run(form, records[form], "f32", fault)
    assert compare(records[form], ours, form, faulty).failures
