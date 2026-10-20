"""No extension phase is a silent no-op: every Extension of ajax.extensions
implementing ``on_target``, ``critic_loss`` or ``actor_loss`` raises on a
batch without what it reads, rather than return its input or 0 (base.py,
"Phase batches"): an agent not passing a key cannot silence it. The agents'
side is test_phase_contract's; a new extension needs a row in ``BUILD``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import SimpleNamespace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import pytest

import ajax.extensions
from ajax.buffers.dataset import TransitionDataset
from ajax.extensions import (
    adversarial,
    ensemble,
    expert,
    instrumentation,
    offline,
    target_mods,
)
from ajax.extensions.base import Extension, ExtensionContext


def _expert(obs: jax.Array) -> jax.Array:
    return jnp.zeros((*obs.shape[:-1], 1))


#: Each extension configured to act (BiasVorePenalty's default coefficient is 0).
BUILD: dict[type, Callable[[], Extension]] = {
    ensemble.KernelRepulsion: ensemble.KernelRepulsion,
    offline.ConservativeQ: offline.ConservativeQ,
    offline.CalQL: offline.CalQL,
    offline.TargetBC: offline.TargetBC,
    expert.OnlineBC: expert.OnlineBC,
    expert.ImitationLoss: lambda: expert.ImitationLoss(expert_policy=_expert),
    instrumentation.BiasVorePenalty: lambda: instrumentation.BiasVorePenalty(0.3),
    target_mods.IBRL: lambda: target_mods.IBRL(expert_policy=_expert),
    target_mods.LCBGatedBootstrap: lambda: target_mods.LCBGatedBootstrap(_expert),
    target_mods.CriticBlend: lambda: target_mods.CriticBlend(expert_policy=_expert),
    target_mods.MCVarianceCorrection: lambda: target_mods.MCVarianceCorrection(1.0),
    adversarial.AdversarialReward: lambda: adversarial.AdversarialReward(
        TransitionDataset([[0.0]], [[0.0]], [0.0], [True], [False], [[0.0]])
    ),
}


def _phases() -> list[Any]:
    """(class, phase) for each phase an ajax.extensions class implements."""
    cases = []
    for info in pkgutil.iter_modules(ajax.extensions.__path__):
        module = importlib.import_module(f"ajax.extensions.{info.name}")
        for cls in vars(module).values():
            if not (inspect.isclass(cls) and issubclass(cls, Extension)):
                continue
            own = cls.__module__ == module.__name__  # not an import
            for phase in ("reward", "on_target", "critic_loss", "actor_loss"):
                if own and getattr(cls, phase) is not getattr(Extension, phase):
                    id_ = f"{cls.__name__}-{phase}"
                    cases.append(pytest.param(cls, phase, id=id_))
    return cases


STATE = SimpleNamespace(expert_critic_params={})  # phi*, as MCPretrain makes it
CTX = ExtensionContext(step=jnp.asarray(0), rng=jax.random.PRNGKey(0), total_steps=8)


@pytest.mark.parametrize("cls, phase", _phases())
def test_a_phase_raises_on_an_empty_batch(cls: type, phase: str) -> None:
    assert cls in BUILD, f"{cls.__name__} needs a row in BUILD"
    target = (jnp.zeros((2, 1)),) if phase in ("reward", "on_target") else ()
    with pytest.raises((KeyError, ValueError)):
        getattr(BUILD[cls](), phase)(STATE, (), {}, *target, CTX)
