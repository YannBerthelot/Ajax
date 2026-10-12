"""Extension stubs the agent tests share: a counter of the ``post_update``
folds, a constant evaluation metric and an ``on_target`` pass-through (a
phase the world-model agents refuse). Frozen dataclasses, so hashable: an
extension stack is a static argument of the jitted train function."""

from dataclasses import dataclass

import jax.numpy as jnp

from ajax.extensions.base import Extension

METRIC = "stub/metric"


@dataclass(frozen=True)
class CounterExt(Extension):
    """Counts in its state the updates ``post_update`` was folded after."""

    name: str = "counter"

    def init_state(self, agent_state, rng):
        return jnp.asarray(0, dtype=jnp.int32)

    def post_update(self, agent_state, ext_state, ctx):
        return agent_state, ext_state + 1


@dataclass(frozen=True)
class MetricExt(Extension):
    """Adds ``METRIC`` = 1 to every evaluation."""

    name: str = "metric"

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        return {METRIC: jnp.asarray(1.0)}


@dataclass(frozen=True)
class TargetTweak(Extension):
    """Implements ``on_target``, which the world-model agents do not fold:
    they must refuse it at construction."""

    name: str = "target-tweak"

    def on_target(self, agent_state, ext_state, batch, target, ctx):
        return target
