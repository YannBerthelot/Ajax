"""Tests for the Extension framework (`ajax.extensions.base`)."""

import dataclasses
import importlib
import pkgutil
import typing
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ajax.extensions
from ajax.agents.SAC.action_pipeline import _dispatch_actions
from ajax.controller import ActingState
from ajax.extensions.base import (
    PHASES,
    Extension,
    ExtensionContext,
    ExtensionStack,
)
from ajax.extensions.expert import ExpertValueMetrics, ImitationLoss
from ajax.extensions.target_mods import IBRL


def _context(step: int = 0) -> ExtensionContext:
    return ExtensionContext(
        step=jnp.asarray(step), rng=jax.random.PRNGKey(step), total_steps=1000
    )


@dataclass(frozen=True)
class _FakeAgentState:
    ext_state: tuple = ()
    value: float = 0.0

    def replace(self, **kw):
        from dataclasses import replace as _r

        return _r(self, **kw)


class AddTarget(Extension):
    def __init__(self, delta):
        self.delta = delta

    def on_target(self, agent_state, ext_state, batch, target, context):
        return target + self.delta


class Counter(Extension):
    def init_state(self, agent_state, rng):
        return jnp.asarray(0)

    def post_update(self, agent_state, ext_state, context):
        return agent_state, ext_state + 1


# --------------------------------------------------------------------------
# Extension base class — every phase defaults to a no-op
# --------------------------------------------------------------------------
def test_extension_defaults_are_noops():
    ext = Extension()
    context = _context()
    obs = jnp.ones((3,))
    target = jnp.ones((2,))
    key = jax.random.PRNGKey(0)

    assert ext.init_state(None, key) == ()
    assert ext.on_target(None, (), None, target, context) is target
    assert ext.critic_loss(None, (), None, context) == 0.0
    assert ext.actor_loss(None, (), None, context) == 0.0
    assert ext.action(None, (), obs, key, context) is None
    assert ext.eval_action(None, (), obs, key, context) is None
    assert ext.pretrain("S", "E", context) == ("S", "E")
    assert ext.post_update("S", "E", context) == ("S", "E")
    assert ext.eval_metrics(None, (), key, context) == {}


def test_implemented_phases_introspection():
    class TargetMod(Extension):
        def on_target(self, agent_state, ext_state, batch, target, context):
            return target + 1.0

        def critic_loss(self, agent_state, ext_state, batch, context):
            return 0.5

    assert TargetMod().implemented_phases() == frozenset({"on_target", "critic_loss"})
    assert Extension().implemented_phases() == frozenset()

    class Metric(Extension):
        def eval_metrics(self, agent_state, ext_state, rng, context):
            return {}

    stack = ExtensionStack((TargetMod(), Metric(), Extension()))
    assert stack.implemented_phases() == {"on_target", "critic_loss", "eval_metrics"}
    assert ExtensionStack().implemented_phases() == frozenset()
    # every declared phase is a real method on the base class
    for phase in PHASES:
        assert callable(getattr(Extension, phase))


# --------------------------------------------------------------------------
# ExtensionStack — folds extensions in list order
# --------------------------------------------------------------------------
def test_stack_folds_transforms_and_sums_losses():
    class ConstCriticLoss(Extension):
        def __init__(self, value):
            self.value = value

        def critic_loss(self, agent_state, ext_state, batch, context):
            return self.value

    stack = ExtensionStack(
        [AddTarget(1.0), AddTarget(10.0), ConstCriticLoss(0.25), ConstCriticLoss(0.75)]
    )
    agent, step, rng = _FakeAgentState(ext_state=((),) * 4), 0, jax.random.PRNGKey(0)
    target, metrics = stack.apply_on_target(
        agent, None, jnp.asarray(0.0), step, rng, 100
    )
    assert float(target) == 11.0 and metrics == {}
    total, metrics = stack.apply_critic_loss(agent, None, step, rng, 100)
    assert float(total) == 1.0 and metrics == {}


def test_a_phase_may_return_its_metrics_beside_its_value():
    """A phase returning ``(value, metrics)``: the fold uses the value and
    merges the metrics, those of a phase returning its value alone being
    none; two extensions logging one name raise."""

    class Logged(Extension):
        def __init__(self, name):
            self.name = name

        def on_target(self, agent_state, ext_state, batch, target, context):
            return target + 1.0, {f"{self.name}/target": target}

        def critic_loss(self, agent_state, ext_state, batch, context):
            return 0.5, {f"{self.name}/critic": jnp.asarray(2.0)}

        def actor_loss(self, agent_state, ext_state, batch, context):
            return 0.25, {f"{self.name}/actor": jnp.asarray(3.0)}

    stack = ExtensionStack([Logged("a"), AddTarget(10.0), Logged("b")])
    agent, step, rng = _FakeAgentState(ext_state=((),) * 3), 0, jax.random.PRNGKey(0)
    target, metrics = stack.apply_on_target(agent, None, jnp.asarray(0.0), step, rng, 9)
    assert float(target) == 12.0
    assert {k: float(v) for k, v in metrics.items()} == {
        "a/target": 0.0,
        "b/target": 11.0,
    }
    for fold, name, total in (
        (stack.apply_critic_loss, "critic", 1.0),
        (stack.apply_actor_loss, "actor", 0.5),
    ):
        term, metrics = fold(agent, None, step, rng, 9)
        assert float(term) == total and sorted(metrics) == [f"a/{name}", f"b/{name}"]

    twice = ExtensionStack([Logged("a"), Logged("a")])
    agent = _FakeAgentState(ext_state=((),) * 2)
    with pytest.raises(ValueError, match="a/critic"):
        twice.apply_critic_loss(agent, None, step, rng, 9)


def test_eval_metrics_merge_with_a_key_per_extension():
    class Metric(Extension):
        def __init__(self, key):
            self.key = key

        def eval_metrics(self, agent_state, ext_state, rng, context):
            return {self.key: rng, f"{self.key}/context": context.rng}

    stack, rng = ExtensionStack([Metric("a/x"), Metric("b/y")]), jax.random.PRNGKey(0)
    out = stack.fold_eval_metrics(_FakeAgentState(ext_state=((), ())), 0, rng, 100)
    assert set(out) == {"a/x", "b/y", "a/x/context", "b/y/context"}
    for key, want in zip(("a/x", "b/y"), jax.random.split(rng, 2), strict=True):
        np.testing.assert_array_equal(out[key], want)
        np.testing.assert_array_equal(out[f"{key}/context"], rng)


def test_state_folds_thread_the_agent_state_in_list_order():
    """pretrain and post_update hand each extension the agent state the
    previous one returned, and write back every extension's own state."""

    @dataclass(frozen=True)
    class Digit(Extension):
        digit: int = 0

        def pretrain(self, agent_state, ext_state, context):
            return agent_state.replace(value=10 * agent_state.value + self.digit), -1

        def post_update(self, agent_state, ext_state, context):
            return self.pretrain(agent_state, ext_state, context)[0], ext_state + 1

    stack, rng = ExtensionStack([Digit(1), Digit(2)]), jax.random.PRNGKey(0)
    agent = _FakeAgentState(ext_state=(0, 5))
    pretrained = stack.fold_pretrain(agent, 0, rng, 100)
    assert (pretrained.value, pretrained.ext_state) == (12, (-1, -1))
    updated = stack.apply_post_update(agent, 0, rng, 100)
    assert (updated.value, updated.ext_state) == (12, (1, 6))


def test_sac_action_dispatch_takes_the_last_proposal():
    """SAC folds the ``action`` phase itself (action_pipeline), in stack
    order: each non-None proposal replaces the batch's action, so the last
    wins; ``None`` defers to the agent's own action."""

    class FixedAction(Extension):
        def __init__(self, value):
            self.value = value

        def action(self, agent_state, ext_state, obs, rng, context):
            return jnp.asarray(self.value)

    agent = SimpleNamespace(
        collector_state=SimpleNamespace(timestep=jnp.asarray(0)), ext_state=((),) * 3
    )
    key = jax.random.PRNGKey(0)

    def dispatched(*extensions):
        batch = {"action": jnp.asarray(0)}
        pairs = list(enumerate(extensions))
        acting = ActingState(obs=None, done=None, timestep=jnp.asarray(0))
        out = _dispatch_actions(agent, acting, pairs, batch, "action", key, 10)
        return int(out["action"])

    assert dispatched(FixedAction(1), Extension(), FixedAction(2)) == 2
    assert dispatched(Extension()) == 0


# --------------------------------------------------------------------------
# Stateful extension — init_state + post_update threaded through a scan
# --------------------------------------------------------------------------
def test_stateful_extension_through_scan():
    stack, rng = ExtensionStack([Counter()]), jax.random.PRNGKey(0)
    ext_states = stack.fold_init_states(_FakeAgentState(), rng).ext_state
    assert int(ext_states[0]) == 0

    def body(carry, _):
        agent = _FakeAgentState(ext_state=carry)
        return stack.apply_post_update(agent, 0, rng, 100).ext_state, None

    final, _ = jax.lax.scan(body, ext_states, None, length=5)
    assert int(final[0]) == 5


# --------------------------------------------------------------------------
# Hashability — the stack must be usable as a JIT static argument
# --------------------------------------------------------------------------
def test_stack_hashable_and_jit_static():
    @dataclass(frozen=True)
    class Frozen(Extension):
        scale: float = 1.0

    s1 = ExtensionStack([Frozen(1.0)])
    s2 = ExtensionStack([Frozen(1.0)])
    assert hash(s1) == hash(s2) and s1 == s2

    f_static = jax.jit(lambda x, st: x * len(st), static_argnums=1)
    assert int(f_static(jnp.asarray(3.0), s1)) == 3


def test_extension_context_is_pytree():
    # step / rng are traced leaves; total_steps is static metadata, carried
    # by the treedef.
    leaves, treedef = jax.tree_util.tree_flatten(_context(step=7))
    assert len(leaves) == 2
    assert jax.tree_util.tree_unflatten(treedef, [None, None]).total_steps == 1000


# --------------------------------------------------------------------------
# fold_<phase> helpers — context-built, ext_state-replaced.
# Critical invariant: empty stack returns the input unchanged WITHOUT
# constructing an ExtensionContext (zero JIT-trace cost).
# --------------------------------------------------------------------------
def test_fold_helpers_empty_stack_zero_cost():
    """Empty stack: every fold returns the input unchanged, no context built."""

    stack = ExtensionStack()
    agent = _FakeAgentState()
    target = jnp.ones((2,))
    batch = {"x": 1}
    rng = jax.random.PRNGKey(0)
    step = jnp.asarray(0)

    assert len(stack) == 0 and not stack
    # All fold helpers must short-circuit before any context construction:
    assert stack.fold_init(agent, rng, 100) is agent
    assert stack.fold_init_states(agent, rng) is agent
    assert stack.fold_pretrain(agent, step, rng, 100) is agent
    assert stack.apply_post_update(agent, step, rng, 100) is agent
    folded, metrics = stack.apply_on_target(agent, batch, target, step, rng, 100)
    assert folded is target and metrics == {}
    assert stack.apply_critic_loss(agent, batch, step, rng, 100) == (0.0, {})
    assert stack.apply_actor_loss(agent, batch, step, rng, 100) == (0.0, {})
    assert stack.fold_eval_metrics(agent, step, rng, 100) == {}


def test_fold_helpers_route_through_phase():
    """Non-empty stack: fold helpers thread agent_state.ext_state correctly."""

    class ConstActorLoss(Extension):
        def actor_loss(self, agent_state, ext_state, batch, context):
            return jnp.asarray(0.25)

    stack = ExtensionStack([AddTarget(1.0), AddTarget(10.0)])
    agent = _FakeAgentState(ext_state=((), ()))
    rng = jax.random.PRNGKey(0)
    step = jnp.asarray(0)

    out, _ = stack.apply_on_target(
        agent, batch=None, target=jnp.asarray(0.0), step=step, rng=rng, total_steps=100
    )
    assert float(out) == 11.0

    # post_update threads through the agent_state.replace and returns
    # the updated agent_state with the new ext_state attached.
    stack2 = ExtensionStack([Counter()])
    agent2 = stack2.fold_init_states(_FakeAgentState(), jax.random.PRNGKey(0))
    new_agent = stack2.apply_post_update(agent2, step, rng, 100)
    assert int(new_agent.ext_state[0]) == 1

    # actor_loss fold returns the additive scalar.
    stack3 = ExtensionStack([ConstActorLoss()])
    agent3 = _FakeAgentState(ext_state=((),))
    val, _ = stack3.apply_actor_loss(
        agent3, batch=None, step=step, rng=rng, total_steps=10
    )
    assert float(val) == 0.25


def test_fold_init_builds_the_states_then_pretrains_at_step_zero():
    """``fold_init`` = ``fold_init_states`` then ``fold_pretrain`` at step 0,
    on the two halves of its key; the pretrain sees the fresh state."""

    @dataclass(frozen=True)
    class Seeded(Extension):
        def init_state(self, agent_state, rng):
            return {"key": rng, "seen": jnp.asarray(-1)}

        def pretrain(self, agent_state, ext_state, context):
            return agent_state, {**ext_state, "seen": context.step, "pre": context.rng}

    stack, rng = ExtensionStack([Seeded()]), jax.random.PRNGKey(3)
    out = stack.fold_init(_FakeAgentState(), rng, 100)
    init_key, pretrain_key = jax.random.split(rng)
    want = stack.fold_pretrain(
        stack.fold_init_states(_FakeAgentState(), init_key), 0, pretrain_key, 100
    )
    assert int(out.ext_state[0]["seen"]) == 0
    np.testing.assert_array_equal(out.ext_state[0]["key"], want.ext_state[0]["key"])
    np.testing.assert_array_equal(out.ext_state[0]["pre"], pretrain_key)


def test_fold_init_states_writes_ext_state():
    class Stateful(Extension):
        def init_state(self, agent_state, rng):
            return jnp.asarray(42)

    stack = ExtensionStack([Stateful()])
    agent = _FakeAgentState()
    out = stack.fold_init_states(agent, jax.random.PRNGKey(0))
    assert int(out.ext_state[0]) == 42


# --------------------------------------------------------------------------
# compose_eval_metrics — None-collapse property + merge ordering.
# --------------------------------------------------------------------------
def test_compose_eval_metrics_none_collapse():
    """No extension -> returns None (preserves zero-overhead branch)."""
    from ajax.log import compose_eval_metrics

    assert compose_eval_metrics(ExtensionStack(), 100) is None


# --------------------------------------------------------------------------
# bind_to_agent — default identity + ExtensionStack uniform fold
# --------------------------------------------------------------------------
def test_bind_to_agent_default_identity():
    """Extension.bind_to_agent ignores unknown kwargs and returns self."""
    ext = Extension()
    bound = ext.bind_to_agent(env_args="x", buffer="y", anything=42)
    assert bound is ext


def test_extension_stack_bind_to_agent_passes_kwargs_to_each():
    """ExtensionStack.bind_to_agent threads kwargs through every extension."""
    seen: dict = {}

    class _Spy(Extension):
        def __init__(self, label):
            self.label = label

        def bind_to_agent(self, **agent_context):
            seen[self.label] = dict(agent_context)
            return self

    stack = ExtensionStack([_Spy("a"), _Spy("b")])
    out = stack.bind_to_agent(env_args="ENV", buffer="BUF")
    assert isinstance(out, ExtensionStack) and len(out) == 2
    assert seen["a"] == {"env_args": "ENV", "buffer": "BUF"}
    assert seen["b"] == {"env_args": "ENV", "buffer": "BUF"}


def test_extension_stack_bind_to_agent_empty_is_noop():
    """Empty stack: bind_to_agent returns self unchanged."""
    stack = ExtensionStack()
    assert stack.bind_to_agent(env_args="X") is stack


def test_compose_eval_metrics_folds_the_stack_at_the_collector_timestep():
    """The stack's metrics, its context's step the collector's timestep."""
    from ajax.log import compose_eval_metrics

    class Metric(Extension):
        def eval_metrics(self, agent_state, ext_state, rng, context):
            return {"s/x": 2.0, "s/step": context.step}

    class FakeCollector:
        def __init__(self):
            self.timestep = jnp.asarray(7)

    class _FakeAgent:
        def __init__(self):
            self.collector_state = FakeCollector()
            self.ext_state = ((),)

    eval_metrics = compose_eval_metrics(ExtensionStack([Metric()]), 100)
    out = eval_metrics(_FakeAgent(), jax.random.PRNGKey(0))
    assert float(out["s/x"]) == 2.0 and int(out["s/step"]) == 7


# --------------------------------------------------------------------------
# with_experts — the experts extensions hold, converted at one door
# --------------------------------------------------------------------------
def _ajax_extension_classes() -> list[type[Extension]]:
    for module in pkgutil.iter_modules(ajax.extensions.__path__):
        importlib.import_module(f"ajax.extensions.{module.name}")
    classes: list[type[Extension]] = []
    todo: list[type[Extension]] = [Extension]
    while todo:
        for sub in todo.pop().__subclasses__():
            todo.append(sub)
            if sub.__module__.startswith("ajax.extensions"):
                classes.append(sub)
    return classes


def test_every_extension_field_holding_an_expert_is_declared():
    """A callable field named for an expert is in ``expert_fields``, or
    with_experts would hand it to the extension unconverted."""
    holders = set()
    for cls in _ajax_extension_classes():
        if not dataclasses.is_dataclass(cls):
            assert not cls.expert_fields, cls.__name__
            continue
        hints = typing.get_type_hints(cls)
        held = {
            f.name
            for f in dataclasses.fields(cls)
            if "expert" in f.name and "Callable" in str(hints[f.name])
        }
        assert cls.expert_fields == held, cls.__name__
        holders |= {cls.__name__} if held else set()
    assert {"ImitationLoss", "IBRL", "MCPretrain", "ExpertValueMetrics"} <= holders


def test_with_experts_converts_each_held_expert_and_keeps_identity():
    def expert(obs):
        return obs

    memo: dict = {}

    def convert(e):
        return memo.setdefault(id(e), lambda obs: 0.5 * e(obs))

    other = AddTarget(1.0)
    stack = ExtensionStack(
        [IBRL(expert), ImitationLoss(expert), ExpertValueMetrics(), other]
    )
    ibrl, imitation, metrics, untouched = stack.with_experts(convert)
    assert ibrl.expert_policy is imitation.expert_policy is convert(expert)
    assert metrics.expert_policy is None  # bound later, from the agent
    assert untouched is other
    # one expert, one object: IBRL still reads the agent's stored actions
    bound = ExtensionStack([ibrl]).bind_to_agent(expert_policy=convert(expert))
    assert bound.extensions[0].reads_stored_expert


def test_the_former_fold_names_still_apply_the_phases():
    """The agents not yet declared on a family call fold_<phase>, kept as
    aliases of apply_<phase> until 0.5.0, on the stack, steps and loop."""
    from ajax.agents import loop, steps

    for phase in ("reward", "on_target", "critic_loss", "actor_loss", "post_update"):
        apply = getattr(ExtensionStack, f"apply_{phase}")
        assert getattr(ExtensionStack, f"fold_{phase}") is apply
    for phase in ("on_target", "critic_loss", "post_update"):
        apply = getattr(steps, f"apply_{phase}")
        assert getattr(steps, f"fold_{phase}") is apply
        assert (
            getattr(loop, f"fold_{phase}") is getattr(loop, f"apply_{phase}") is apply
        )


@dataclass(frozen=True)
class _NamedCtx(Extension):
    """A downstream extension written before the rename: each phase names its
    ExtensionContext ctx, as Ajax's did. The stack (and SAC's action
    dispatch) passes it positionally, so the name does not matter."""

    name: str = "named_ctx"

    def init_state(self, agent_state, rng):
        return jnp.asarray(0.0)

    def pretrain(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + ctx.total_steps

    def reward(self, agent_state, ext_state, batch, rewards, ctx):
        return rewards + ctx.step, ext_state, {}

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target + ctx.step

    def critic_loss(self, agent_state, ext_state, batch, ctx):
        return 2.0 * ctx.step

    def actor_loss(self, agent_state, ext_state, batch, ctx):
        return 3.0 * ctx.step

    def action(self, agent_state, ext_state, obs, rng, ctx):
        return ctx.step

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + ctx.step

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {"named_ctx/step": ctx.step}


def test_a_phase_naming_its_context_ctx_still_runs():
    stack, rng, step = ExtensionStack([_NamedCtx()]), jax.random.PRNGKey(0), 2.0
    agent = stack.fold_init(_FakeAgentState(), rng, 100)
    assert float(agent.ext_state[0]) == 100.0  # pretrain read ctx.total_steps
    _, rewards, _ = stack.apply_reward(agent, None, jnp.zeros(3), step, rng, 100)
    target, _ = stack.apply_on_target(agent, None, jnp.asarray(1.0), step, rng, 100)
    critic, _ = stack.apply_critic_loss(agent, None, step, rng, 100)
    actor, _ = stack.apply_actor_loss(agent, None, step, rng, 100)
    after = stack.apply_post_update(agent, step, rng, 100)
    metrics = stack.fold_eval_metrics(agent, step, rng, 100)
    assert (rewards == 2.0).all() and float(target) == 3.0
    assert (float(critic), float(actor)) == (4.0, 6.0)
    assert float(after.ext_state[0]) == 102.0
    assert float(metrics["named_ctx/step"]) == 2.0
    acting = ActingState(obs=None, done=None, timestep=jnp.asarray(step))
    pairs = [(0, _NamedCtx())]
    out = _dispatch_actions(agent, acting, pairs, {"action": 0.0}, "action", rng, 10)
    assert float(out["action"]) == 2.0
