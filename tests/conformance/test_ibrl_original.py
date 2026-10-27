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

Their acting, QAgent._act_ibrl, ran on a fixed batch of states in both forms and both modes (eval: the
actor's mean; train: its sample, the noise kept). Ajax's IBRL.choose_action makes the same choice from
the same candidates and target critics: in eval mode from the authors' mean, in train mode from TD3's
exploration of that mean, its noise taped to theirs; on the state form its subset draw taped to their
forward_k pair. Every row's executed action matches within the same tolerance, and two planted faults
fail: scoring with the live critic, and choosing on every critic instead of the drawn pair.
Pre-registered (ajax-scratch/ibrl/ACTING_PLAN.md) before its first run.

Both checks run again on their soft choice ("ibrl_soft", act_method and bootstrap alike), against
records made with it: IBRL(soft_choice=True), its choice's uniform draw taped to the event of their
multinomial's exponential race, u = q_1 / (q_0 + q_1) (they take BC iff p_1 / q_1 > p_0 / q_0, i.e. iff
u < p_1). Same tolerance; four planted faults fail: beta ignored (1) and the hard choice, in the target
and in acting. Pre-registered (ajax-scratch/ibrl/SOFT_PLAN.md) before its first run.
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
from ajax.agents.TD3.TD3 import compute_td3_td_target, make_gaussian_exploration
from ajax.extensions.base import ExtensionContext
from ajax.extensions.target_mods import IBRL
from tests.conformance import kit, originals, submaps

NET = ("16", "relu", "16", "relu")
FORMS, MODES, CHOICES = ("pair", "state"), ("eval", "train"), ("hard", "soft")
STDDEV = 0.1  # every IBRL config's acting floor, the recorder's
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


def make_ibrl(expert: Any, choice: str, fault: Optional[str], **bound: Any) -> IBRL:
    """IBRL with the record's choice, or a planted fault's."""
    soft = choice == "soft" and fault != "hard choice"
    beta = 1.0 if fault == "beta ignored" else 10.0
    return IBRL(expert_policy=expert, soft_choice=soft, soft_choice_beta=beta, **bound)


def take_choice_uniforms(exponential: np.ndarray) -> np.ndarray:
    """Per row, the uniform whose draw below p_1 is their race's choice of BC."""
    exponential = exponential.astype(np.float64)
    return (exponential[:, 1] / exponential.sum(1))[:, None]


def compute_target(
    form: str, record: dict, precision: str, fault: Optional[str], choice: str
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
    ibrl = make_ibrl(lambda observations: expert_actions, choice, fault)
    context = ExtensionContext(
        step=jnp.asarray(0), rng=jax.random.PRNGKey(2), total_steps=1
    )
    agent_state = SimpleNamespace(actor_state=state.actor_state, critic_state=critics)
    target, _ = ibrl.on_target(agent_state, None, batch, target, context)
    return np.asarray(target[:, 0])


def named(choice: str, form: str) -> str:
    """A target record's name (an acting record's follows act_)."""
    return f"soft_{form}" if choice == "soft" else form


def run(
    form: str,
    record: dict,
    precision: str,
    fault: Optional[str] = None,
    choice: str = "hard",
) -> np.ndarray:
    """compute_target in precision; on the state form, REDQ's subset draw taped to the
    authors' value pair and IBRL's selection draw to their selection pair (fault
    "one pair": to the value pair); a soft choice's uniform to their race's."""
    with kit.recording() as session:  # the draws' keys, the same in either precision
        compute_target(form, record, "f32", fault, choice)
    draws = kit.by_role(session.log)
    selection = record.get("draw.value" if fault == "one pair" else "draw.selection")
    values = {}
    for role in draws:
        if role[1].endswith("compute_redq_td_target"):
            values[role] = [record["draw.value"]]
        if role[1].endswith("draw_selection_critics"):
            values[role] = [selection]
        if role[1].endswith("choose_expert"):
            values[role] = [take_choice_uniforms(record["draw.exponential"])]
    table = kit.key_table({role: draws[role] for role in values}, values)
    with jax.enable_x64(precision == "f64"), kit.replaying(table) as replay:
        out = compute_target(form, record, precision, fault, choice)
    assert not replay.misses
    return out


@pytest.fixture(scope="module")
def records() -> dict:
    found = originals.committed("hengyuan_hu_ibrl").glob("*.npz")
    return {path.stem: dict(np.load(path)) for path in found}


@pytest.fixture(scope="module")
def ours(records: dict) -> dict:
    return {
        (choice, form, precision): run(
            form, records[named(choice, form)], precision, choice=choice
        )
        for choice in CHOICES
        for form in FORMS
        for precision in ("f32", "f64")
    }


def compare(
    record: dict, ours: dict, choice: str, form: str, target: np.ndarray
) -> submaps.Report:
    theirs = ({"target": record["f32.target"]}, {"target": record["f64.target"]})
    mine = (
        {"target": ours[choice, form, "f32"]},
        {"target": ours[choice, form, "f64"]},
    )
    nulls = submaps.rounding_null(theirs, mine)
    return submaps.compare({"target": target}, theirs[0], nulls, context=form)


@pytest.mark.parametrize("choice", CHOICES)
@pytest.mark.parametrize("form", FORMS)
def test_the_target_matches_the_authors(
    records: dict, ours: dict, form: str, choice: str
) -> None:
    record = records[named(choice, form)]
    report = compare(record, ours, choice, form, ours[choice, form, "f32"])
    assert not report.failures, str(report)


@pytest.mark.parametrize(
    "choice, form, fault",
    [
        ("hard", "pair", "own draw"),
        ("hard", "state", "one pair"),
        ("soft", "pair", "beta ignored"),
        ("soft", "state", "hard choice"),
    ],
)
def test_a_planted_fault_fails_the_check(
    records: dict, ours: dict, choice: str, form: str, fault: str
) -> None:
    record = records[named(choice, form)]
    faulty = run(form, record, "f32", fault, choice)
    assert compare(record, ours, choice, form, faulty).failures


def choose_actions(
    form: str,
    record: dict,
    mode: str,
    precision: str,
    fault: Optional[str],
    choice: str,
) -> np.ndarray:
    """IBRL's executed action per row, on the record's states and target critics: the expert's
    recorded action against the agent's, the authors' mean (eval) or TD3's exploration of it (train)."""
    dtype = np.float64 if precision == "f64" else np.float32
    critics = init_fresh_state(form).critic_state
    authors = take_authors_critics(critics.target_params, record, dtype)
    if fault == "live critic":  # Ajax's own init, the live critic's
        authors = jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), critics.params)
    critics = critics.replace(target_params=authors)

    def rows(name: str) -> jax.Array:
        return jnp.asarray(record[f"{mode}.{precision}.{name}"], dtype)

    states, mean = jnp.asarray(record["batch.state"], dtype), rows("mean")
    action = mean
    if mode == "train":
        policy = SimpleNamespace(mean=lambda: mean)
        explore = make_gaussian_exploration(STDDEV)
        action = explore(policy, jax.random.PRNGKey(3), jnp.asarray(0))
    size = 2 if form == "state" and fault != "every critic" else None
    ibrl = make_ibrl(
        lambda obs: rows("bc_action"), choice, fault, critic_subset_size=size
    )
    step = {"observations": states, "raw_obs": states, "expert_action": None}
    agent_state = SimpleNamespace(critic_state=critics)
    key = jax.random.PRNGKey(4)  # IBRL's own key, as the action stage hands it
    return np.asarray(ibrl.choose_action(agent_state, step, action, key))


def run_acting(
    form: str,
    record: dict,
    mode: str,
    precision: str,
    fault: Optional[str] = None,
    choice: str = "hard",
) -> np.ndarray:
    """choose_actions in precision, TD3's noise taped to the authors' and, on the state form, IBRL's
    subset draw to their forward_k pair; a soft choice's uniform to their race's."""
    with kit.recording() as session:  # the draws' keys, the same in either precision
        choose_actions(form, record, mode, "f32", fault, choice)
    draws = kit.by_role(session.log)
    values = {}
    for role in draws:
        if role[1].endswith("explore"):
            values[role] = [record[f"{mode}.draw.noise"]]
        if role[1].endswith("draw_selection_critics"):
            values[role] = [record[f"{mode}.draw.selection"]]
        if role[1].endswith("choose_expert"):
            exponential = record[f"{mode}.draw.exponential"]
            values[role] = [take_choice_uniforms(exponential)]
    table = kit.key_table({role: draws[role] for role in values}, values)
    with jax.enable_x64(precision == "f64"), kit.replaying(table) as replay:
        out = choose_actions(form, record, mode, precision, fault, choice)
    assert not replay.misses
    return out


@pytest.fixture(scope="module")
def ours_acting(records: dict) -> dict:
    return {
        (choice, form, mode, precision): run_acting(
            form, records[f"act_{named(choice, form)}"], mode, precision, choice=choice
        )
        for choice in CHOICES
        for form in FORMS
        for mode in MODES
        for precision in ("f32", "f64")
    }


def compare_acting(
    record: dict, ours: dict, key: tuple, action: np.ndarray
) -> submaps.Report:
    """key: (choice, form, mode)."""
    mode = key[2]
    theirs = (
        {"action": record[f"{mode}.f32.action"]},
        {"action": record[f"{mode}.f64.action"]},
    )
    mine = ({"action": ours[(*key, "f32")]}, {"action": ours[(*key, "f64")]})
    nulls = submaps.rounding_null(theirs, mine)
    return submaps.compare({"action": action}, theirs[0], nulls, context=key[1])


@pytest.mark.parametrize("choice", CHOICES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("form", FORMS)
def test_the_acting_matches_the_authors(
    records: dict, ours_acting: dict, form: str, mode: str, choice: str
) -> None:
    record = records[f"act_{named(choice, form)}"]
    action = ours_acting[choice, form, mode, "f32"]
    report = compare_acting(record, ours_acting, (choice, form, mode), action)
    assert not report.failures, str(report)


@pytest.mark.parametrize(
    "choice, form, mode, fault",
    [
        ("hard", "pair", "train", "live critic"),
        ("hard", "state", "eval", "every critic"),
        ("soft", "pair", "train", "beta ignored"),
        ("soft", "state", "eval", "hard choice"),
    ],
)
def test_a_planted_acting_fault_fails_the_check(
    records: dict, ours_acting: dict, choice: str, form: str, mode: str, fault: str
) -> None:
    record = records[f"act_{named(choice, form)}"]
    faulty = run_acting(form, record, mode, "f32", fault, choice)
    key = (choice, form, mode)
    assert compare_acting(record, ours_acting, key, faulty).failures
