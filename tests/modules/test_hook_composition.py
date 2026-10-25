"""Tests for the callable hooks agents still accept.

The Extension framework (``extensions=``) is the surface for research
features. A few ``Optional[Callable]`` hooks remain, each with a live user:
TD3's ``action_pipeline`` (AjaxExperiments' TD3 variants) and the DQN /
PQN variants (Double DQN, Huber).

These are API-contract tests: construct each agent with each hook (and
``None``) and confirm construction succeeds and the hook reaches training:
forwarded to ``make_train``, or a declared agent's field (TD3's pipeline,
which its ``behaviour`` hands collection; PQN's TD loss, which its
``update`` reads). They do not run training.
"""

import pytest

from ajax.agents.DQN.DQN import DQN
from ajax.agents.PQN.PQN import PQN
from ajax.agents.TD3.TD3 import TD3
from tests.agents.TD3.test_td3_metrics import make_context

AGENT_HOOKS = {
    TD3: ("action_pipeline",),
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
    if agent_cls is TD3:
        common["buffer_size"] = 1024
        common["batch_size"] = 32
    common.update(kwargs)
    return agent_cls(**common)


def _hooks(agent):
    """The hooks training receives: make_train's keywords, TD3's pipeline
    (its family's make_train collects with what ``behaviour`` returns) or
    PQN's TD loss (a field its update reads)."""
    if isinstance(agent, TD3):
        return {"action_pipeline": agent.pipeline}
    if isinstance(agent, PQN):
        return {"td_loss_fn": agent.td_loss_fn}
    return agent.get_make_train().keywords


@pytest.mark.parametrize("agent_cls", list(AGENT_HOOKS), ids=lambda c: c.__name__)
def test_default_hooks_are_none(agent_cls):
    """By default every hook training receives is None (feature inactive)."""
    keywords = _hooks(_instantiate(agent_cls))
    for hook in AGENT_HOOKS[agent_cls]:
        assert keywords[hook] is None, (
            f"{agent_cls.__name__}.{hook} should default to None"
        )


@pytest.mark.parametrize("agent_cls", list(AGENT_HOOKS), ids=lambda c: c.__name__)
def test_get_make_train_forwards_hooks(agent_cls):
    """Each hook reaches training: get_make_train's partial carries it,
    TD3's behaviour hands its pipeline to collection, or PQN holds it."""
    hook_kwargs = dict.fromkeys(AGENT_HOOKS[agent_cls], _identity_hook)
    agent = _instantiate(agent_cls, **hook_kwargs)
    keywords = _hooks(agent)
    for h, fn in hook_kwargs.items():
        assert h in keywords, f"{agent_cls.__name__} should forward {h}"
        assert keywords[h] is fn
    if isinstance(agent, TD3):
        assert agent.choose_behaviour(make_context(agent)) is _identity_hook
