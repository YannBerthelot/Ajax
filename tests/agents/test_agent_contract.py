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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer

import ajax.log
from ajax.agents.APG.APG import APG
from ajax.agents.APG.train_APG import make_policy_step, rollout_returns
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
from ajax.evaluate import evaluate, get_deterministic_action_and_entropy_fn
from ajax.extensions.base import Extension
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
from tests.agents.extension_stubs import METRIC, CounterExt, MetricExt
from tests.environments import paid_action

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
    # Evaluates its policy (UDRL: no command to evaluate it with).
    evaluates: bool = True
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
        evaluates=False,
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


@AGENTS
def test_the_evaluation_key_is_apart_from_the_training_key(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The initial state's ``eval_rng`` is split from the run's key apart
    from ``rng``, which every training draw descends from: an evaluation
    repeats none of them."""
    import ajax.agents.loop as loop

    seen, fresh_state = [], loop.fresh_state

    def recording(*args: Any) -> Any:
        state = fresh_state(*args)
        record = lambda *keys: seen.append([np.asarray(k) for k in keys])  # noqa: E731
        jax.debug.callback(record, state.rng, state.eval_rng)
        return state

    monkeypatch.setattr(loop, "fresh_state", recording)
    case = CASES[name]
    case.cls(env_id=case.envs[0], n_envs=1, **case.kwargs).train(seed=0, n_timesteps=0)
    (rng, eval_rng), *_ = seen
    assert not np.array_equal(rng, eval_rng)


#: The keys the evaluations of one run hand the policy's evaluation and the
#: extensions' ``eval_metrics``, in order.
_EVAL_KEYS: dict[str, list] = {"policy": [], "extensions": []}


def _record(kind: str) -> Any:
    return lambda key: _EVAL_KEYS[kind].append(np.asarray(key).tobytes())


@dataclass(frozen=True)
class _KeyExt(Extension):
    """Records the key its ``eval_metrics`` is handed."""

    name: str = "key"

    def eval_metrics(self, agent_state, ext_state, rng, ctx):
        jax.debug.callback(_record("extensions"), rng)
        return {}


@AGENTS
def test_each_evaluation_draws_fresh_keys(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The policy's evaluation and the extensions' ``eval_metrics`` draw new
    keys at each evaluation of a run (``eval_rng`` folded with the count of
    evaluations so far): fresh initial states and noise every time."""
    evaluate = ajax.log.evaluate

    def recording(*args: Any, rng: Any, **kwargs: Any) -> Any:
        jax.debug.callback(_record("policy"), rng)
        return evaluate(*args, rng=rng, **kwargs)

    monkeypatch.setattr(ajax.log, "evaluate", recording)
    for keys in _EVAL_KEYS.values():
        keys.clear()
    case = CASES[name]
    agent = case.cls(
        env_id=case.envs[0], n_envs=1, extensions=[_KeyExt()], **case.kwargs
    )
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    _, out = agent.train(
        seed=0, n_timesteps=case.n_timesteps, num_episode_test=1, logging_config=logging
    )
    jax.effects_barrier()
    evaluations = int((np.asarray(out["timestep"]) >= 0).sum())
    assert evaluations >= 2
    kinds = ("policy", "extensions") if case.evaluates else ("extensions",)
    for kind in kinds:
        keys = _EVAL_KEYS[kind]
        assert len(keys) == evaluations, (kind, len(keys))
        assert len(set(keys)) == evaluations, kind


# --------------------------------------------------------------------------
# Continuous agents act on the env's whole box
# --------------------------------------------------------------------------
#: Each continuous agent on a Box(-2, 2) env paying the action it receives:
#: SAC's and TD3's first 64 steps warm up.
ON_THE_BOX = ("APG", "APO", "ASAC", "AVG", "PPO", "REDQ", "SAC", "TD3", "UDRL")
WARM_UP = {"learning_starts": 64, "buffer_size": 128}


def _paid_rows(name: str, state: Any) -> tuple[np.ndarray, np.ndarray]:
    """The stored agent actions and the rewards of every collected row."""
    if name == "UDRL":  # its episode segments
        rows = state.buffer.actions, state.buffer.rewards
    elif CASES[name].buffer:
        experience = state.collector_state.buffer_state.experience
        rows = experience["action"], experience["reward"]
    else:  # the exposed rollout
        rows = state.last_rollout.action, state.last_rollout.reward
    return tuple(np.asarray(x).reshape(-1) for x in rows)  # type: ignore[return-value]


@pytest.mark.parametrize("name", ON_THE_BOX)
def test_an_agent_acts_on_the_whole_box(name: str) -> None:
    """Agent actions in [-1, 1] are mapped onto the env's bounds: every
    collected row is paid 2 * clip(a, -1, 1) for its stored action a, and
    evaluate() the deterministic action's (APG: its closed loop's). SAC's
    and TD3's uniform warm-up covers the box (half their |paid| > 1)."""
    env, params = paid_action.make()
    if name == "APG":
        agent = APG(env, env_params=params, n_envs=4, horizon=1, pid=None)
        seeds, _ = agent.train(seed=0, n_timesteps=0)
        state: Any = jax.tree.map(lambda x: x[0], seeds)
        actor = state.actor_state
        _, rollout = rollout_returns(
            state, actor.params, jax.random.PRNGKey(1), agent.env_args, None, 1, 4
        )
        policy_step = make_policy_step(actor, actor.params, False)
        own, _ = policy_step(None, rollout.obs[0], rollout.resets[0])
        np.testing.assert_allclose(
            rollout.reward[0], paid_action.mapped(own)[:, 0], rtol=1e-6
        )
        return
    case = CASES[name]
    exposed = {"expose_recent_rollout": True} if case.rollout is not None else {}
    kwargs = case.kwargs | exposed | (WARM_UP if case.buffer else {})
    agent = case.cls(env_id=env, env_params=params, n_envs=1, **kwargs)
    state, _ = agent.train(seed=0, n_timesteps=96 if case.buffer else 128)
    actions, rewards = _paid_rows(name, state)
    assert np.abs(actions).max() > 0.01, "no action was collected"
    np.testing.assert_allclose(rewards, paid_action.mapped(actions), rtol=1e-6)
    if name in ("SAC", "TD3"):
        assert (np.abs(rewards[:64]) > 1.0).mean() > 0.25
    if not case.evaluates:
        return
    actor = jax.tree.map(lambda x: x[0], state.actor_state)
    paid = evaluate(env, actor, 2, jax.random.PRNGKey(1), params)[0]
    act = get_deterministic_action_and_entropy_fn(actor, False, True)
    own, _, _ = act(paid_action.OBS[None], None, None)
    np.testing.assert_allclose(paid, paid_action.mapped(own)[0, 0], rtol=1e-6)
