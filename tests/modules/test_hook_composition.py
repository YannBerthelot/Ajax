"""Tests for the composable hook API shared across agents.

Every agent accepts some subset of these ``Optional[Callable]`` hooks at
``__init__`` and stores them on the instance for ``get_make_train`` to
forward into the compiled training function:

    - ``pid_actor_config``
    - ``action_pipeline``
    - ``eval_action_transform``
    - ``target_modifier``            (SAC-family only: SAC, REDQ, ASAC, AVG)
    - ``obs_preprocessor``
    - ``policy_action_transform``    (SAC-family only)

These tests are API-contract tests: construct each agent with each
applicable hook (and ``None``) and confirm construction succeeds and the
attribute is stored. They do not run training — the probing suite covers
end-to-end behaviour with ``None`` hooks.
"""

import pytest

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.PPO.PPO import PPO
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC

SAC_FAMILY_HOOKS = (
    "pid_actor_config",
    "action_pipeline",
    "eval_action_transform",
    "target_modifier",
    "obs_preprocessor",
    "policy_action_transform",
)

# SAC additionally exposes ``runtime_maintenance`` (phi-refresh hook).
SAC_HOOKS = (*SAC_FAMILY_HOOKS, "runtime_maintenance")

PPO_FAMILY_HOOKS = (
    "pid_actor_config",
    "action_pipeline",
    "eval_action_transform",
    "obs_preprocessor",
)

AGENT_HOOKS = {
    SAC: SAC_HOOKS,
    REDQ: SAC_FAMILY_HOOKS,
    ASAC: SAC_FAMILY_HOOKS,
    AVG: SAC_FAMILY_HOOKS,
    PPO: PPO_FAMILY_HOOKS,
    APO: PPO_FAMILY_HOOKS,
}


def _identity_hook(*args, **kwargs):
    """Trivial callable used to exercise hook plumbing."""
    if args:
        return args[0]
    return None


def _instantiate(agent_cls, **kwargs):
    """Build a minimal agent instance on a cheap env."""
    common = {
        "env_id": "Pendulum-v1",
        "n_envs": 1,
        "actor_architecture": ("32", "relu"),
        "critic_architecture": ("32", "relu"),
    }
    # Off-policy agents take a buffer_size; on-policy don't accept it.
    if agent_cls in (SAC, REDQ, ASAC):
        common["buffer_size"] = 1024
        common["batch_size"] = 32
    if agent_cls in (PPO, APO):
        common["n_steps"] = 32
        common["batch_size"] = 32
        common["n_epochs"] = 1
    common.update(kwargs)
    return agent_cls(**common)


@pytest.mark.parametrize("agent_cls", list(AGENT_HOOKS), ids=lambda c: c.__name__)
def test_default_hooks_are_none(agent_cls):
    """By default every hook attribute is None (feature inactive)."""
    agent = _instantiate(agent_cls)
    for hook in AGENT_HOOKS[agent_cls]:
        if hook == "pid_actor_config":
            # PIDActorConfig is a dataclass, not a callable; None by default.
            assert getattr(agent, hook) is None
        else:
            assert (
                getattr(agent, hook) is None
            ), f"{agent_cls.__name__}.{hook} should default to None"


@pytest.mark.parametrize("agent_cls", list(AGENT_HOOKS), ids=lambda c: c.__name__)
def test_hooks_are_stored(agent_cls):
    """Passed-in hooks are retained on the instance (for get_make_train)."""
    hook_kwargs = {
        h: _identity_hook for h in AGENT_HOOKS[agent_cls] if h != "pid_actor_config"
    }
    agent = _instantiate(agent_cls, **hook_kwargs)
    for h, fn in hook_kwargs.items():
        assert (
            getattr(agent, h) is fn
        ), f"{agent_cls.__name__}.{h} should store the hook it was given"


@pytest.mark.parametrize("agent_cls", list(AGENT_HOOKS), ids=lambda c: c.__name__)
def test_get_make_train_forwards_hooks(agent_cls):
    """get_make_train wraps make_train in a partial that carries the hooks.

    Skipped for agents that call ``make_train`` directly from ``train()``
    rather than exposing it via ``get_make_train`` (currently AVG).
    """
    hook_kwargs = {
        h: _identity_hook for h in AGENT_HOOKS[agent_cls] if h != "pid_actor_config"
    }
    agent = _instantiate(agent_cls, **hook_kwargs)
    if not hasattr(agent, "get_make_train"):
        pytest.skip(f"{agent_cls.__name__} does not expose get_make_train")
    make_train_partial = agent.get_make_train()
    # functools.partial stores kwargs on .keywords
    keywords = getattr(make_train_partial, "keywords", {})
    for h, fn in hook_kwargs.items():
        assert (
            h in keywords
        ), f"{agent_cls.__name__}.get_make_train() should forward {h}"
        assert keywords[h] is fn
