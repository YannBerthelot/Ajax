"""Tests for the callable hooks agents still accept.

The Extension framework (``extensions=``) is the surface for research
features. A few ``Optional[Callable]`` hooks remain, each with a live user:
SAC's escape hatches (see CONTRIBUTING.md), TD3's ``action_pipeline``
(AjaxExperiments' TD3 variants), PPO's ``reward_shaping_fn``, and the
DQN / PQN variants (Double DQN, Huber). ``pid_actor_config`` (a network
option, not a callable) is checked alongside.

These are API-contract tests: construct each agent with each hook (and
``None``) and confirm construction succeeds and the attribute is stored and
forwarded to ``make_train``. They do not run training.
"""

import pytest

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.DQN.DQN import DQN
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from ajax.agents.SafeSAC.SafeSAC import SafeSAC
from ajax.agents.TD3.TD3 import TD3

SAC_HOOKS = (
    "pid_actor_config",
    "action_pipeline",
    "eval_action_transform",
    "obs_preprocessor",
    "policy_action_transform",
    "extra_actor_loss_fn",
    "extra_critic_loss_fn",
    "init_transform",
    "auxiliary_update",
)

AGENT_HOOKS = {
    SAC: SAC_HOOKS,
    SafeSAC: SAC_HOOKS,
    TD3: ("pid_actor_config", "action_pipeline"),
    REDQ: ("pid_actor_config",),
    ASAC: ("pid_actor_config",),
    AVG: ("pid_actor_config",),
    PPO: ("pid_actor_config", "reward_shaping_fn"),
    APO: ("pid_actor_config",),
    # Variants (Double DQN, Huber) are exposed as Optional[Callable] hooks.
    DQN: ("td_target_fn", "td_loss_fn"),
    PQN: ("td_loss_fn",),
}


def _identity_hook(*args, **kwargs):
    """Trivial callable used to exercise hook plumbing."""
    if args:
        return args[0]
    return None


def _instantiate(agent_cls, **kwargs):
    """Build a minimal agent instance on a cheap env."""
    if agent_cls is DQN:
        # DQN is discrete-only and takes a single `architecture`.
        common = {
            "env_id": "CartPole-v1",
            "n_envs": 1,
            "architecture": ("32", "relu"),
            "buffer_size": 1024,
            "batch_size": 32,
        }
        common.update(kwargs)
        return agent_cls(**common)
    if agent_cls is PQN:
        # PQN is discrete-only, on-policy (no buffer), single `architecture`.
        common = {
            "env_id": "CartPole-v1",
            "n_envs": 2,
            "architecture": ("32", "relu"),
            "n_steps": 8,
            "num_minibatches": 2,
        }
        common.update(kwargs)
        return agent_cls(**common)
    common = {
        "env_id": "Pendulum-v1",
        "n_envs": 1,
        "actor_architecture": ("32", "relu"),
        "critic_architecture": ("32", "relu"),
    }
    # Off-policy agents take a buffer_size; on-policy don't accept it.
    if agent_cls in (SAC, SafeSAC, TD3, REDQ, ASAC):
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
