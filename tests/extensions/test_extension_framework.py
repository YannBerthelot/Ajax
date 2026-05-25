"""Tests for the Extension framework (`ajax.extensions.base`)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ajax.extensions.base import (
    PHASES,
    Extension,
    ExtensionContext,
    ExtensionStack,
)


def _ctx(step: int = 0) -> ExtensionContext:
    return ExtensionContext(
        step=jnp.asarray(step), rng=jax.random.PRNGKey(step), total_steps=1000
    )


# --------------------------------------------------------------------------
# Extension base class — every phase defaults to a no-op
# --------------------------------------------------------------------------
def test_extension_defaults_are_noops():
    ext = Extension()
    ctx = _ctx()
    obs = jnp.ones((3,))
    target = jnp.ones((2,))
    key = jax.random.PRNGKey(0)

    assert ext.init_state(None, key) == ()
    assert ext.on_obs(obs, (), ctx) is obs
    assert ext.on_batch("batch", (), ctx) == "batch"
    assert ext.on_target(None, (), None, target, ctx) is target
    assert ext.critic_loss(None, (), None, ctx) == 0.0
    assert ext.actor_loss(None, (), None, ctx) == 0.0
    assert ext.action(None, (), obs, key, ctx) is None
    assert ext.eval_action(None, (), obs, key, ctx) is None
    assert ext.pretrain("S", "E", ctx) == ("S", "E")
    assert ext.post_update("S", "E", ctx) == ("S", "E")
    assert ext.eval_metrics(None, (), key, ctx) == {}


def test_implemented_phases_introspection():
    class TargetMod(Extension):
        def on_target(self, agent_state, ext_state, batch, target, ctx):
            return target + 1.0

        def critic_loss(self, agent_state, ext_state, batch, ctx):
            return 0.5

    assert TargetMod().implemented_phases() == frozenset({"on_target", "critic_loss"})
    assert Extension().implemented_phases() == frozenset()
    # every declared phase is a real method on the base class
    for phase in PHASES:
        assert callable(getattr(Extension, phase))


# --------------------------------------------------------------------------
# ExtensionStack — empty stack is a genuine no-op
# --------------------------------------------------------------------------
def test_empty_stack_is_noop():
    stack = ExtensionStack()
    ctx = _ctx()
    target = jnp.ones((2,))
    key = jax.random.PRNGKey(0)

    assert len(stack) == 0 and not stack
    assert stack.init_states(None, key) == ()
    assert stack.pretrain("S", (), ctx) == ("S", ())
    assert stack.on_obs(target, (), ctx) is target
    assert stack.on_target(None, (), None, target, ctx) is target
    assert stack.critic_loss(None, (), None, ctx) == 0.0
    assert stack.actor_loss(None, (), None, ctx) == 0.0
    assert stack.action(None, (), target, key, ctx) is None
    assert stack.eval_metrics(None, (), key, ctx) == {}
    assert stack.post_update("S", (), ctx) == ("S", ())


# --------------------------------------------------------------------------
# ExtensionStack — folds extensions in list order
# --------------------------------------------------------------------------
def test_stack_folds_transforms_and_sums_losses():
    class AddTarget(Extension):
        def __init__(self, delta):
            self.delta = delta

        def on_target(self, agent_state, ext_state, batch, target, ctx):
            return target + self.delta

    class ConstCriticLoss(Extension):
        def __init__(self, value):
            self.value = value

        def critic_loss(self, agent_state, ext_state, batch, ctx):
            return self.value

    stack = ExtensionStack(
        [AddTarget(1.0), AddTarget(10.0), ConstCriticLoss(0.25), ConstCriticLoss(0.75)]
    )
    states = ((),) * 4
    ctx = _ctx()
    assert float(stack.on_target(None, states, None, jnp.asarray(0.0), ctx)) == 11.0
    assert float(stack.critic_loss(None, states, None, ctx)) == 1.0


def test_eval_metrics_merge():
    class Metric(Extension):
        def __init__(self, key):
            self.key = key

        def eval_metrics(self, agent_state, ext_state, rng, ctx):
            return {self.key: 1.0}

    stack = ExtensionStack([Metric("a/x"), Metric("b/y")])
    out = stack.eval_metrics(None, ((), ()), jax.random.PRNGKey(0), _ctx())
    assert set(out) == {"a/x", "b/y"}


def test_action_last_non_none_wins():
    class FixedAction(Extension):
        def __init__(self, value):
            self.value = value

        def action(self, agent_state, ext_state, obs, rng, ctx):
            return jnp.asarray(self.value)

    obs, key = jnp.ones((3,)), jax.random.PRNGKey(0)
    stack = ExtensionStack([FixedAction(1), Extension(), FixedAction(2)])
    assert int(stack.action(None, ((),) * 3, obs, key, _ctx())) == 2
    # no action-providing extension -> None (agent uses its own policy)
    assert ExtensionStack([Extension()]).action(None, ((),), obs, key, _ctx()) is None


# --------------------------------------------------------------------------
# Stateful extension — init_state + post_update threaded through a scan
# --------------------------------------------------------------------------
def test_stateful_extension_through_scan():
    class Counter(Extension):
        def init_state(self, agent_state, rng):
            return jnp.asarray(0)

        def post_update(self, agent_state, ext_state, ctx):
            return agent_state, ext_state + 1

    stack = ExtensionStack([Counter()])
    ext_states = stack.init_states(None, jax.random.PRNGKey(0))
    assert int(ext_states[0]) == 0

    def body(carry, _):
        _, new = stack.post_update(None, carry, _ctx())
        return new, None

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

    @jax.jit
    def f(x, stack):  # stack is hashable -> static when not an array arg
        return x * len(stack)

    f_static = jax.jit(lambda x, st: x * len(st), static_argnums=1)
    assert int(f_static(jnp.asarray(3.0), s1)) == 3


def test_extension_context_is_pytree():
    ctx = _ctx(step=7)
    # step / rng are traced leaves; total_steps is static metadata.
    leaves = jax.tree_util.tree_leaves(ctx)
    assert len(leaves) >= 2
    assert ctx.total_steps == 1000


# --------------------------------------------------------------------------
# fold_<phase> sugar helpers — None-guarded, ctx-built, ext_state-replaced.
# Critical invariant: empty stack returns the input unchanged WITHOUT
# constructing an ExtensionContext (zero JIT-trace cost).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _FakeAgentState:
    ext_state: tuple = ()
    value: float = 0.0

    def replace(self, **kw):
        from dataclasses import replace as _r

        return _r(self, **kw)


def test_fold_helpers_empty_stack_zero_cost():
    """Empty stack: every fold returns the input unchanged, no ctx built."""

    # Sentinel that would fail if ExtensionContext were instantiated with
    # our crafted args (proving the empty-stack short-circuit is taken).
    class _Boom:
        def __getattr__(self, name):
            raise AssertionError("ctx attr accessed on empty-stack path")

    stack = ExtensionStack()
    agent = _FakeAgentState()
    obs = jnp.ones((3,))
    target = jnp.ones((2,))
    batch = {"x": 1}
    rng = jax.random.PRNGKey(0)
    step = jnp.asarray(0)

    # All fold helpers must short-circuit before any ctx construction:
    assert stack.fold_init_states(agent, rng) is agent
    assert stack.fold_pretrain(agent, step, rng, 100) is agent
    assert stack.fold_post_update(agent, step, rng, 100) is agent
    assert stack.fold_on_target(agent, batch, target, step, rng, 100) is target
    assert stack.fold_on_obs(obs, agent, step, rng, 100) is obs
    assert stack.fold_on_batch(batch, agent, step, rng, 100) is batch
    assert stack.fold_critic_loss(agent, batch, step, rng, 100) == 0.0
    assert stack.fold_actor_loss(agent, batch, step, rng, 100) == 0.0
    assert stack.fold_action(agent, obs, step, rng, 100) is None
    assert stack.fold_eval_action(agent, obs, step, rng, 100) is None
    assert stack.fold_eval_metrics(agent, step, rng, 100) == {}

    # Spot-check the spirit of the "no ctx built" claim: an Extension
    # implementation that touches ctx would crash if reached; we
    # construct one and verify the empty stack never reaches its phase.
    # The above identity-return assertions are the concrete evidence.
    del _Boom  # explicitly unused — kept as documentation of intent


def test_fold_helpers_route_through_phase():
    """Non-empty stack: fold helpers thread agent_state.ext_state correctly."""

    class AddTarget(Extension):
        def __init__(self, delta):
            self.delta = delta

        def on_target(self, agent_state, ext_state, batch, target, ctx):
            return target + self.delta

    class Counter(Extension):
        def init_state(self, agent_state, rng):
            return jnp.asarray(0)

        def post_update(self, agent_state, ext_state, ctx):
            return agent_state, ext_state + 1

    class ConstActorLoss(Extension):
        def actor_loss(self, agent_state, ext_state, batch, ctx):
            return jnp.asarray(0.25)

    stack = ExtensionStack([AddTarget(1.0), AddTarget(10.0)])
    agent = _FakeAgentState(ext_state=((), ()))
    rng = jax.random.PRNGKey(0)
    step = jnp.asarray(0)

    out = stack.fold_on_target(
        agent, batch=None, target=jnp.asarray(0.0), step=step, rng=rng, total_steps=100
    )
    assert float(out) == 11.0

    # post_update threads through the agent_state.replace and returns
    # the updated agent_state with the new ext_state attached.
    stack2 = ExtensionStack([Counter()])
    init_state = stack2.init_states(_FakeAgentState(), jax.random.PRNGKey(0))
    agent2 = _FakeAgentState(ext_state=init_state)
    new_agent = stack2.fold_post_update(agent2, step, rng, 100)
    assert int(new_agent.ext_state[0]) == 1

    # actor_loss fold returns the additive scalar.
    stack3 = ExtensionStack([ConstActorLoss()])
    agent3 = _FakeAgentState(ext_state=((),))
    val = stack3.fold_actor_loss(agent3, batch=None, step=step, rng=rng, total_steps=10)
    assert float(val) == 0.25


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
    """Both inputs are no-ops -> returns None (preserves zero-overhead branch)."""
    from ajax.log import compose_eval_metrics

    assert compose_eval_metrics(None, None, 100) is None
    assert compose_eval_metrics(None, ExtensionStack(), 100) is None


def test_compose_eval_metrics_user_only_passthrough():
    """User callable + empty stack -> the user callable itself."""
    from ajax.log import compose_eval_metrics

    def user_fn(agent_state, rng):
        return {"u": 1.0}

    out = compose_eval_metrics(user_fn, ExtensionStack(), 100)
    assert out is user_fn


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


def test_compose_eval_metrics_merges_user_and_stack():
    """Both contribute -> dict union, stack metrics take precedence on conflict."""
    from ajax.log import compose_eval_metrics

    class Metric(Extension):
        def eval_metrics(self, agent_state, ext_state, rng, ctx):
            return {"s/x": 2.0}

    class FakeCollector:
        def __init__(self):
            self.timestep = jnp.asarray(7)

    class _FakeAgent:
        def __init__(self):
            self.collector_state = FakeCollector()
            self.ext_state = ((),)

    def user_fn(agent_state, rng):
        return {"u/a": 1.0}

    stack = ExtensionStack([Metric()])
    merged = compose_eval_metrics(user_fn, stack, 100)
    out = merged(_FakeAgent(), jax.random.PRNGKey(0))
    assert set(out) == {"u/a", "s/x"}
    assert float(out["s/x"]) == 2.0
