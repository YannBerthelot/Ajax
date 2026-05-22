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
