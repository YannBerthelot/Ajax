"""The contract every agent on the shared training loop keeps, one table
row per agent: it builds its configs on each action space it acts in and
refuses the other; a default run (one scalar seed, one env) keeps no
extension state and no rollout; a run over two seeds and two envs, logged
without a backend, folds two counters in step, adds an extension's metric
to every evaluation and, where the agent exposes it, keeps the last rollout.
The instrumentation extensions, left to auto-detect, measure the critic of
every agent that has one.

The two runs use different env modes (brax ``fast`` and gymnax) where the
agent acts in both. The maths of each agent is tested in its own directory.
"""

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from extension_stubs import METRIC, CounterExt, MetricExt
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer

from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.DQN.DQN import DQN
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.agents.UDRL.UDRL import UDRL
from ajax.extensions.instrumentation import (
    COND_METRIC_KEYS,
    EVAREST_DECOMP_KEYS,
    BiasVoreDecomposition,
    ConditioningMetrics,
)
from ajax.logging.wandb_logging import LoggingConfig
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)

CONTINUOUS, DISCRETE = ("fast", "Pendulum-v1"), ("CartPole-v1",)
NET = ("16", "relu")
OFF_POLICY = {
    "actor_architecture": NET,
    "critic_architecture": NET,
    "learning_starts": 8,
    "batch_size": 8,
    "buffer_size": 128,
}
ON_POLICY = {
    "actor_architecture": NET,
    "critic_architecture": NET,
    "n_steps": 16,
    "batch_size": 8,  # two minibatches a rollout
    "n_epochs": 1,
}


@dataclass(frozen=True)
class Case:
    cls: type
    kwargs: dict[str, Any]
    # The env ids it acts in: the default run uses the first, the other the last.
    envs: tuple[str, ...]
    rollout: int | None = None  # length of the rollout it exposes, if it does
    alpha: bool = False  # has a temperature (alpha_args)
    buffer: bool = False  # has a replay buffer
    critic: bool = True  # has a value or action-value network
    n_timesteps: int = 64
    # Overrides for the extensions' run only.
    extended: dict[str, Any] = field(default_factory=dict)


CASES = {
    "SAC": Case(SAC, OFF_POLICY, CONTINUOUS, alpha=True, buffer=True),
    "ASAC": Case(ASAC, OFF_POLICY, CONTINUOUS, alpha=True, buffer=True),
    "REDQ": Case(
        REDQ,
        {**OFF_POLICY, "num_critic_updates": 2, "num_critics": 4, "subset_size": 2},
        CONTINUOUS,
        alpha=True,
        buffer=True,
    ),
    "TD3": Case(TD3, OFF_POLICY, CONTINUOUS, buffer=True),
    "AVG": Case(
        AVG,
        {"actor_architecture": NET, "critic_architecture": NET},
        CONTINUOUS,
        rollout=1,  # AVG learns from each step as it comes
        alpha=True,
    ),
    "APO": Case(APO, ON_POLICY, CONTINUOUS + DISCRETE, rollout=16, n_timesteps=128),
    "PPO": Case(
        PPO,
        ON_POLICY,
        CONTINUOUS + DISCRETE,
        rollout=16,
        n_timesteps=128,
        # One minibatch: the flat update, whose targets the extensions fold
        # over the whole rollout (with two envs, two would split in time).
        extended={"batch_size": 16},
    ),
    "DQN": Case(
        DQN,
        {
            "architecture": NET,
            "learning_starts": 16,
            "batch_size": 8,
            "buffer_size": 256,
            "target_update_interval": 10,
        },
        DISCRETE,
        buffer=True,
        n_timesteps=128,
    ),
    "PQN": Case(
        PQN,
        {"architecture": NET, "n_steps": 8, "n_epochs": 2, "num_minibatches": 2},
        DISCRETE,
        rollout=8,
        n_timesteps=128,
    ),
    "UDRL": Case(
        UDRL,
        {
            "actor_architecture": NET,
            "n_steps": 8,
            "batch_size": 8,
            "n_updates_per_iter": 4,
            "buffer_capacity": 8,
        },
        CONTINUOUS + DISCRETE,
        n_timesteps=128,
        critic=False,
    ),
}
AGENTS = pytest.mark.parametrize("name", sorted(CASES))


@pytest.mark.parametrize(
    "name, env_id", [(n, e) for n in sorted(CASES) for e in CASES[n].envs]
)
def test_an_agent_builds_its_configs(name: str, env_id: str) -> None:
    case = CASES[name]
    agent = case.cls(env_id=env_id)
    for attr, kind in (
        ("env_args", EnvironmentConfig),
        ("actor_optimizer_args", OptimizerConfig),
        ("critic_optimizer_args", OptimizerConfig),
        ("network_args", NetworkConfig),
    ):
        assert isinstance(getattr(agent, attr), kind), attr
    assert type(agent.agent_config).__name__ == f"{name}Config"
    assert agent.env_args.continuous == (env_id not in DISCRETE)
    assert isinstance(getattr(agent, "alpha_args", None), AlphaConfig) == case.alpha
    # An on-policy agent keeps no replay buffer.
    assert isinstance(getattr(agent, "buffer", None), TrajectoryBuffer) == case.buffer


@pytest.mark.parametrize(
    "name", [n for n in sorted(CASES) if CASES[n].envs in (CONTINUOUS, DISCRETE)]
)
def test_an_agent_refuses_the_action_space_it_cannot_act_in(name: str) -> None:
    case = CASES[name]
    other, kind = ("Pendulum-v1", "discrete")
    if case.envs == CONTINUOUS:
        other, kind = ("CartPole-v1", "continuous")
    with pytest.raises(ValueError, match=f"{name} only supports {kind} action spaces"):
        case.cls(env_id=other)


@AGENTS
def test_a_default_run_keeps_no_extension_state_and_no_rollout(name: str) -> None:
    case = CASES[name]
    agent = case.cls(env_id=case.envs[0], n_envs=1, **case.kwargs)
    state, out = agent.train(seed=42, n_timesteps=case.n_timesteps)
    assert state.ext_state == ()
    assert state.last_rollout is None
    assert out is None  # no logging config: no evaluations


@AGENTS
def test_a_run_folds_its_extensions_and_keeps_its_rollout(name: str) -> None:
    """Two counters advance in step on every seed; every evaluation carries
    the stub's metric; an exposed rollout is (T, n_envs) and finite."""
    case = CASES[name]
    exposed = {} if case.rollout is None else {"expose_recent_rollout": True}
    agent = case.cls(
        env_id=case.envs[-1],
        n_envs=2,
        extensions=[CounterExt("a"), CounterExt("b"), MetricExt()],
        **exposed,
        **(case.kwargs | case.extended),
    )
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    state, out = agent.train(
        seed=[42, 43],
        n_timesteps=case.n_timesteps,
        num_episode_test=1,
        logging_config=logging,
    )
    a, b, _ = (np.asarray(s) for s in state.ext_state)
    assert a.shape == (2,) and (a > 1).all() and (a == b).all(), (a, b)
    evaluated = np.asarray(out["timestep"]) >= 0
    assert evaluated.any(axis=1).all()
    np.testing.assert_array_equal(np.asarray(out[METRIC])[evaluated], 1.0)
    if case.rollout is None:
        assert state.last_rollout is None
        return
    obs = state.last_rollout.obs
    assert obs.shape[:3] == (2, case.rollout, 2), obs.shape
    assert jnp.isfinite(obs).all()


@AGENTS
def test_the_instrumentation_finds_the_critic_on_its_own(name: str) -> None:
    """ConditioningMetrics and BiasVoreDecomposition, with no per-agent
    setting, find the critic (on critic_state, taking the observation or
    the observation and action, or a Q-network on actor_state) and their
    batch (the replay buffer or the exposed rollout): each evaluation of an
    agent with a critic carries all their keys, finite; one without (UDRL)
    carries none."""
    case = CASES[name]
    exposed = {} if case.rollout is None else {"expose_recent_rollout": True}
    kwargs = {"env_id": case.envs[-1], "n_envs": 2, **exposed, **case.kwargs}
    buffer = getattr(case.cls(**kwargs), "buffer", None)
    agent = case.cls(
        **kwargs,
        extensions=[
            ConditioningMetrics(buffer=buffer),
            BiasVoreDecomposition(buffer=buffer),
        ],
    )
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    _, out = agent.train(
        seed=42,
        n_timesteps=case.n_timesteps,
        num_episode_test=1,
        logging_config=logging,
    )
    keys = COND_METRIC_KEYS + EVAREST_DECOMP_KEYS
    if not case.critic:
        assert not set(keys) & set(out), sorted(out)
        return
    evaluated = np.asarray(out["timestep"]) >= 0
    for key in keys:
        assert np.isfinite(np.asarray(out[key])[evaluated]).all(), key
