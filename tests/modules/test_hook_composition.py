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

Phase 2 of the agent-architecture rework introduced an additional
``extensions=`` surface for SAC (Phase 1 added the framework; Phase 2
migrated SAC). Behaviour-equivalence between the legacy SAC hook flags
and the new extension surface is verified in
``tests/extensions/test_sac_extensions_equivalence.py``. The hook
attributes themselves remain accepted for backward compatibility, so
the API-contract tests below keep passing across the migration.
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

SAC_FAMILY_HOOKS = (
    "pid_actor_config",
    "action_pipeline",
    "eval_action_transform",
    "target_modifier",
    "obs_preprocessor",
    "policy_action_transform",
)

# SAC-specific hook list. Phase 2b migrated several SAC hooks to Extension
# phase methods and removed the matching kwargs from ``SAC.__init__``:
#   - ``target_modifier``       → ``Extension.on_target`` (commit 1408a95)
#   - ``runtime_maintenance``   → ``PhiRefresh.post_update`` (commit af86ca0)
# The other SAC-family agents (REDQ / ASAC / AVG) still accept these as
# callable hooks (their own ``__init__`` signatures are independent), so
# ``SAC_FAMILY_HOOKS`` above is unchanged. The SAC-specific list below
# excludes the migrated hooks.
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

PPO_FAMILY_HOOKS = (
    "pid_actor_config",
    "action_pipeline",
    "eval_action_transform",
    "obs_preprocessor",
)

# PPO additionally exposes init_transform / auxiliary_update / extra_eval_metrics.
PPO_HOOKS = (
    *PPO_FAMILY_HOOKS,
    "init_transform",
    "auxiliary_update",
    "extra_eval_metrics",
    "extra_actor_loss_fn",
    "extra_critic_loss_fn",
)

# DQN is value-based and discrete: no actor-side or SAC-family hooks.
# Its variants (Double DQN, Huber) are exposed as Optional[Callable] hooks.
DQN_HOOKS = (
    "action_pipeline",
    "eval_action_transform",
    "td_target_fn",
    "td_loss_fn",
    "extra_eval_metrics",
)

# PQN is value-based and discrete too; on-policy, so no Double-DQN-style
# target hook -- just exploration, eval transform, the TD loss and the
# extra-eval-metrics hook.
PQN_HOOKS = (
    "action_pipeline",
    "eval_action_transform",
    "td_loss_fn",
    "extra_eval_metrics",
)

AGENT_HOOKS = {
    SAC: SAC_HOOKS,
    SafeSAC: SAC_HOOKS,
    REDQ: SAC_FAMILY_HOOKS,
    ASAC: SAC_FAMILY_HOOKS,
    AVG: SAC_FAMILY_HOOKS,
    PPO: PPO_HOOKS,
    APO: PPO_FAMILY_HOOKS,
    DQN: DQN_HOOKS,
    PQN: PQN_HOOKS,
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
    if agent_cls in (SAC, SafeSAC, REDQ, ASAC):
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
