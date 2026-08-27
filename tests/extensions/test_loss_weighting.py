"""Tests for per-sample loss weighting.

Covers the framework phases (`ajax.extensions.base`), the reduction
helper, and the :class:`InterestWeighting` extension.

The load-bearing property is the **identity**: weighting with a constant
weight must reproduce the unweighted reduction exactly. Everything about
using this for research rests on an unweighted arm being a fair control,
so a drift here would silently invalidate every comparison built on it.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from ajax.extensions.allocation import InterestWeighting
from ajax.extensions.base import (
    PHASES,
    Extension,
    ExtensionContext,
    ExtensionStack,
    weighted_mean,
)


def _ctx(step: int = 0) -> ExtensionContext:
    return ExtensionContext(
        step=jnp.asarray(step), rng=jax.random.PRNGKey(step), total_steps=1000
    )


@dataclass(frozen=True)
class _ConstantWeights(Extension):
    """Minimal weighting extension for framework-level tests."""

    value: float = 2.0
    actor: bool = True
    critic: bool = True
    name: str = "constant_weights"

    def actor_loss_weights(self, agent_state, ext_state, batch, ctx):
        del agent_state, ext_state, ctx
        return jnp.full((batch["n"],), self.value) if self.actor else None

    def critic_loss_weights(self, agent_state, ext_state, batch, ctx):
        del agent_state, ext_state, ctx
        return jnp.full((batch["n"],), self.value) if self.critic else None


# --------------------------------------------------------------------------
# weighted_mean — the reduction contract
# --------------------------------------------------------------------------
def test_none_weights_are_exactly_the_plain_mean():
    losses = jnp.array([1.0, 2.0, 7.0, -3.0])
    assert weighted_mean(losses, None) == losses.mean()


@pytest.mark.parametrize("constant", [1.0, 0.5, 17.0])
def test_uniform_weights_reproduce_the_plain_mean(constant):
    """Self-normalisation: uniform weights carry no learning-rate scale."""
    losses = jnp.array([1.0, 2.0, 7.0, -3.0])
    weights = jnp.full_like(losses, constant)
    assert jnp.allclose(weighted_mean(losses, weights), losses.mean(), atol=1e-6)


def test_weighted_mean_matches_the_closed_form():
    losses = jnp.array([1.0, 3.0])
    weights = jnp.array([1.0, 3.0])
    # (1*1 + 3*3) / (1 + 3) = 2.5
    assert jnp.allclose(weighted_mean(losses, weights), 2.5, atol=1e-6)


def test_weights_broadcast_against_a_trailing_axis():
    """A (batch,) weight vector must align with a (batch, 1) loss."""
    losses = jnp.array([[1.0], [3.0]])
    weights = jnp.array([1.0, 3.0])
    assert jnp.allclose(weighted_mean(losses, weights), 2.5, atol=1e-6)


def test_zero_weights_do_not_produce_nan():
    losses = jnp.array([1.0, 2.0])
    assert jnp.isfinite(weighted_mean(losses, jnp.zeros_like(losses)))


def test_weighted_mean_is_jittable():
    losses = jnp.array([1.0, 3.0])
    weights = jnp.array([1.0, 3.0])
    jitted = jax.jit(weighted_mean)(losses, weights)
    assert jnp.allclose(jitted, 2.5, atol=1e-6)


# --------------------------------------------------------------------------
# Framework phases
# --------------------------------------------------------------------------
def test_weight_phases_default_to_none():
    ext = Extension()
    ctx = _ctx()
    assert ext.actor_loss_weights(None, (), None, ctx) is None
    assert ext.critic_loss_weights(None, (), None, ctx) is None


def test_weight_phases_are_registered_in_PHASES():
    assert "actor_loss_weights" in PHASES
    assert "critic_loss_weights" in PHASES


def test_empty_stack_folds_to_none():
    stack = ExtensionStack(())
    assert stack.fold_actor_loss_weights(None, {}, 0, jax.random.PRNGKey(0), 10) is None
    assert (
        stack.fold_critic_loss_weights(None, {}, 0, jax.random.PRNGKey(0), 10) is None
    )


def test_stack_multiplies_weights_across_extensions():
    stack = ExtensionStack((_ConstantWeights(value=2.0), _ConstantWeights(value=3.0)))
    weights = stack.actor_loss_weights(None, ((), ()), {"n": 4}, _ctx())
    assert jnp.allclose(weights, jnp.full((4,), 6.0))


def test_stack_skips_extensions_that_return_none():
    """A non-weighting extension must not turn the product into None."""
    stack = ExtensionStack((Extension(), _ConstantWeights(value=5.0)))
    weights = stack.critic_loss_weights(None, ((), ()), {"n": 3}, _ctx())
    assert jnp.allclose(weights, jnp.full((3,), 5.0))


def test_stack_of_only_nonweighting_extensions_is_none():
    stack = ExtensionStack((Extension(), Extension()))
    assert stack.actor_loss_weights(None, ((), ()), {"n": 3}, _ctx()) is None


def test_implemented_phases_reports_weighting():
    ext = _ConstantWeights()
    assert "actor_loss_weights" in ext.implemented_phases()
    assert "critic_loss_weights" in ext.implemented_phases()


# --------------------------------------------------------------------------
# InterestWeighting
# --------------------------------------------------------------------------
def test_interest_weighting_applies_only_to_the_enabled_head():
    ext = InterestWeighting(weight_fn=lambda obs: obs[:, 0], apply_to_actor=True)
    batch = {"observations": jnp.array([[1.0, 9.0], [4.0, 9.0]])}
    ctx = _ctx()
    assert jnp.allclose(
        ext.actor_loss_weights(None, (), batch, ctx), jnp.array([1.0, 4.0])
    )
    assert ext.critic_loss_weights(None, (), batch, ctx) is None


def test_interest_weighting_floor_is_added():
    ext = InterestWeighting(
        weight_fn=lambda obs: jnp.zeros(obs.shape[0]),
        apply_to_critic=True,
        floor=0.25,
    )
    batch = {"observations": jnp.zeros((3, 2))}
    assert jnp.allclose(
        ext.critic_loss_weights(None, (), batch, _ctx()), jnp.full((3,), 0.25)
    )


def test_interest_weighting_requires_a_head():
    with pytest.raises(ValueError, match="apply_to_actor or"):
        InterestWeighting(weight_fn=lambda obs: obs)


def test_interest_weighting_rejects_a_negative_floor():
    with pytest.raises(ValueError, match="non-negative"):
        InterestWeighting(weight_fn=lambda obs: obs, apply_to_actor=True, floor=-1.0)


def test_interest_weighting_without_observations_is_none():
    """Loss sites that pass no observations must degrade to unweighted."""
    ext = InterestWeighting(weight_fn=lambda obs: obs, apply_to_actor=True)
    assert ext.actor_loss_weights(None, (), {"actor_params": {}}, _ctx()) is None


def test_interest_weighting_is_hashable():
    """Extensions are JIT static args, so they must hash."""
    fn = lambda obs: obs[:, 0]  # noqa: E731
    assert hash(InterestWeighting(weight_fn=fn, apply_to_actor=True)) is not None


def test_uniform_interest_is_a_no_op_reduction():
    """The experiment-critical identity, end to end through the extension."""
    ext = InterestWeighting(
        weight_fn=lambda obs: jnp.ones(obs.shape[0]), apply_to_actor=True
    )
    batch = {"observations": jnp.zeros((4, 2))}
    weights = ext.actor_loss_weights(None, (), batch, _ctx())
    losses = jnp.array([1.0, 2.0, 7.0, -3.0])
    assert jnp.allclose(weighted_mean(losses, weights), losses.mean(), atol=1e-6)


# --------------------------------------------------------------------------
# End-to-end through PPO
# --------------------------------------------------------------------------
def _ppo(extensions):
    from ajax import PPO

    return PPO(
        env_id="CartPole-v1",
        n_envs=2,
        actor_architecture=("16", "relu"),
        critic_architecture=("16", "relu"),
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        extensions=extensions,
    )


def _actor_params(state):
    """Actor leaves from ``train``'s return (a ``(state, metrics)`` tuple)."""
    if isinstance(state, tuple):
        state = state[0]
    return jax.tree.leaves(state.actor_state.params)


@dataclass(frozen=True)
class _NoOpExtension(Extension):
    """An extension that implements no phase at all."""

    name: str = "noop"


def test_uniform_weighting_matches_an_unweighted_run_exactly():
    """Uniform weights must leave PPO's parameters bit-identical.

    This is the control-arm guarantee: an experiment comparing a weighted
    run against an unweighted one is only interpretable if the weighting
    machinery itself changes nothing when the weights are flat.

    The comparison is against a **no-op extension**, not against a bare
    ``extensions=[]`` agent: attaching any extension at all shifts the
    run (extension-state init consumes randomness), so an empty-stack
    baseline would fold that unrelated shift into this assertion. Note
    the consequence for experiments — an unweighted control arm should
    carry an extension too, or it differs from the treatment arm by more
    than the weights.
    """
    uniform = InterestWeighting(
        weight_fn=lambda obs: jnp.ones(obs.shape[0]),
        apply_to_actor=True,
        apply_to_critic=True,
    )
    unweighted = _ppo([_NoOpExtension()]).train(seed=42, n_timesteps=128)
    weighted = _ppo([uniform]).train(seed=42, n_timesteps=128)

    for a, b in zip(_actor_params(unweighted), _actor_params(weighted), strict=True):
        assert jnp.allclose(a, b, atol=1e-6), "uniform weighting perturbed the policy"


def test_nonuniform_weighting_changes_the_policy():
    """The complement: real weights must actually reach the update."""
    skewed = InterestWeighting(
        weight_fn=lambda obs: jnp.abs(obs[:, 0]) + 0.01, apply_to_actor=True
    )
    vanilla = _ppo([]).train(seed=42, n_timesteps=128)
    weighted = _ppo([skewed]).train(seed=42, n_timesteps=128)

    differs = any(
        not jnp.allclose(a, b, atol=1e-6)
        for a, b in zip(_actor_params(vanilla), _actor_params(weighted), strict=True)
    )
    assert differs, "non-uniform actor weights did not change the policy"


def test_uniform_weights_leave_the_ppo_losses_unchanged():
    """Loss-level identity — deterministic, immune to training chaos.

    The end-to-end test above compares whole training runs, where any
    difference is amplified by feedback. This pins the actual contract at
    the point it is implemented: on one fixed batch, weighting by ones
    must reproduce the unweighted loss to float precision.
    """
    from ajax.agents.PPO.train_PPO import policy_loss_function, value_loss_function

    ones = lambda obs: jnp.ones(obs.shape[0])  # noqa: E731
    agent = _ppo([])
    state = agent.train(seed=0, n_timesteps=64)
    state = state[0] if isinstance(state, tuple) else state
    # ``train`` vmaps over seeds, so every leaf carries a leading seed
    # axis; the loss functions expect a single unbatched network state.
    state = jax.tree.map(lambda leaf: leaf[0], state)

    obs = jnp.asarray(
        jax.random.normal(jax.random.PRNGKey(1), (8, 4)), dtype=jnp.float32
    )
    targets = jax.random.normal(jax.random.PRNGKey(2), (8, 1))
    dones = jnp.zeros((8,), dtype=bool)

    plain, _ = value_loss_function(
        state.critic_state.params, state.critic_state, obs, targets, dones, False
    )
    weighted, _ = value_loss_function(
        state.critic_state.params,
        state.critic_state,
        obs,
        targets,
        dones,
        False,
        sample_weights_fn=ones,
    )
    assert jnp.allclose(plain, weighted, atol=1e-6), "critic loss changed under ones"

    actions = jnp.zeros((8, 1), dtype=jnp.int32)
    log_probs = jnp.full((8, 1), -0.7)
    gae = jax.random.normal(jax.random.PRNGKey(3), (8, 1))
    common = (
        state.actor_state.params,
        state.actor_state,
        obs,
        actions,
        log_probs,
        gae,
        dones,
        False,
        0.2,
        0.01,
        False,
        None,
    )
    plain_a, _ = policy_loss_function(*common)
    weighted_a, _ = policy_loss_function(*common, sample_weights_fn=ones)
    assert jnp.allclose(plain_a, weighted_a, atol=1e-6), "actor loss changed under ones"


# --------------------------------------------------------------------------
# Entropy allocation
# --------------------------------------------------------------------------
def test_entropy_weights_default_to_none():
    ext = Extension()
    assert ext.entropy_weights(None, (), None, _ctx()) is None


def test_entropy_allocation_reports_only_its_phase():
    from ajax.extensions.allocation import EntropyAllocation

    ext = EntropyAllocation(weight_fn=lambda obs: obs[:, 0])
    assert ext.implemented_phases() == frozenset({"entropy_weights"})


def test_entropy_stack_multiplies_and_skips_none():
    from ajax.extensions.allocation import EntropyAllocation

    a = EntropyAllocation(weight_fn=lambda obs: jnp.full(obs.shape[0], 2.0))
    b = EntropyAllocation(weight_fn=lambda obs: jnp.full(obs.shape[0], 3.0))
    stack = ExtensionStack((a, Extension(), b))
    batch = {"observations": jnp.zeros((4, 2))}
    w = stack.entropy_weights(None, ((), (), ()), batch, _ctx())
    assert jnp.allclose(w, jnp.full((4,), 6.0))


def test_uniform_entropy_weights_leave_the_actor_loss_unchanged():
    """The control-arm guarantee for the entropy head, at loss level."""
    from ajax.agents.PPO.train_PPO import policy_loss_function

    ones = lambda obs: jnp.ones(obs.shape[0])  # noqa: E731
    state = _ppo([]).train(seed=0, n_timesteps=64)
    state = state[0] if isinstance(state, tuple) else state
    state = jax.tree.map(lambda leaf: leaf[0], state)

    obs = jnp.asarray(
        jax.random.normal(jax.random.PRNGKey(1), (8, 4)), dtype=jnp.float32
    )
    common = (
        state.actor_state.params,
        state.actor_state,
        obs,
        jnp.zeros((8, 1), dtype=jnp.int32),
        jnp.full((8, 1), -0.7),
        jax.random.normal(jax.random.PRNGKey(3), (8, 1)),
        jnp.zeros((8,), dtype=bool),
        False,
        0.2,
        0.01,
        False,
        None,
    )
    plain, _ = policy_loss_function(*common)
    weighted, _ = policy_loss_function(*common, entropy_weights_fn=ones)
    assert jnp.allclose(plain, weighted, atol=1e-6)


def test_nonuniform_entropy_weights_change_the_actor_loss():
    """And the complement: real entropy weights must reach the objective."""
    from ajax.agents.PPO.train_PPO import policy_loss_function

    skewed = lambda obs: jnp.abs(obs[:, 0]) + 0.01  # noqa: E731
    state = _ppo([]).train(seed=0, n_timesteps=64)
    state = state[0] if isinstance(state, tuple) else state
    state = jax.tree.map(lambda leaf: leaf[0], state)

    obs = jnp.asarray(
        jax.random.normal(jax.random.PRNGKey(1), (8, 4)), dtype=jnp.float32
    )
    common = (
        state.actor_state.params,
        state.actor_state,
        obs,
        jnp.zeros((8, 1), dtype=jnp.int32),
        jnp.full((8, 1), -0.7),
        jax.random.normal(jax.random.PRNGKey(3), (8, 1)),
        jnp.zeros((8,), dtype=bool),
        False,
        0.2,
        0.05,
        False,
        None,
    )
    _, aux_plain = policy_loss_function(*common)
    _, aux_w = policy_loss_function(*common, entropy_weights_fn=skewed)
    assert not jnp.allclose(aux_plain.entropy, aux_w.entropy, atol=1e-6)
