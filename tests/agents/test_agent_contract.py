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

import importlib
import inspect
import pkgutil
import sys
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flashbax.buffers.trajectory_buffer import TrajectoryBuffer

import ajax.agents
from ajax.agents import base
from ajax.agents.APG.APG import APG
from ajax.agents.APG.train_APG import make_policy_step, rollout_returns
from ajax.agents.APO.APO import APO
from ajax.agents.ASAC.ASAC import ASAC
from ajax.agents.AVG.AVG import AVG
from ajax.agents.base import ActorCritic, run_config
from ajax.agents.DQN.DQN import DQN
from ajax.agents.DreamerV3.DreamerV3 import DreamerV3
from ajax.agents.DreamerV3.state import DreamerV3State
from ajax.agents.PPO.PPO import PPO
from ajax.agents.PQN.PQN import PQN
from ajax.agents.REDQ.REDQ import REDQ
from ajax.agents.SAC.SAC import SAC
from ajax.agents.TD3.TD3 import TD3
from ajax.agents.TDMPC2.TDMPC2 import TDMPC2
from ajax.agents.UDRL.state import UDRLConfig
from ajax.agents.UDRL.train_UDRL import EvalCommand
from ajax.agents.UDRL.UDRL import UDRL
from ajax.evaluate import evaluate, get_deterministic_action_and_entropy_fn
from ajax.extensions.base import Extension
from ajax.extensions.capabilities import MissingCapability
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
from tests.agents.DreamerV3.test_dreamerv3_agent import TINY as DREAMER_TINY
from tests.agents.extension_stubs import METRIC, CounterExt, MetricExt
from tests.agents.TDMPC2.common import TINY_AGENT as TDMPC2_TINY
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
    cls: type[ActorCritic]
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

_PG = ("policy/approx_kl", "policy/grad_norm", "value/explained_variance")
#: Training diagnostics each agent logs (docs/guide/logging.md), finite at
#: every evaluation of a run that updates.
DIAGNOSTICS: dict[str, tuple[str, ...]] = {
    "PPO": (*_PG, "value/grad_norm"),
    "APO": (*_PG, "value/grad_norm", "value/value_bias", "value/average_reward"),
    "DQN": ("policy/epsilon", "value/abs_td_error", "value/grad_norm"),
    "PQN": ("policy/epsilon", "value/abs_td_error", "value/grad_norm"),
    "SAC": ("policy/grad_norm", "value/grad_norm"),
    "TD3": (
        "policy/grad_norm",
        "value/grad_norm",
        "value/q_gap",
        "value/target_noise_clip_frac",
    ),
}


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
    # The run config holds the constructor's arguments and the agent's name.
    assert agent.config["env_id"] == env_id
    assert agent.config["algo_name"] == agent.name == name
    assert not {"self", "__class__"} & set(agent.config)


def test_the_run_config_drops_the_instance_and_the_excluded() -> None:
    class Agent:
        def __init__(self, a: int, b: int = 2) -> None:
            self.config = run_config(locals(), "X", exclude=("b",))

    assert Agent(1).config == {"a": 1, "algo_name": "X"}


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
    exposed: dict[str, Any] = (
        {} if case.rollout is None else {"expose_recent_rollout": True}
    )
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
    for metric in DIAGNOSTICS.get(name, ()):
        assert np.isfinite(np.asarray(out[metric])[evaluated]).all(), metric
    if case.rollout is None:
        assert state.last_rollout is None
        return
    assert state.last_rollout is not None
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
    exposed: dict[str, Any] = (
        {} if case.rollout is None else {"expose_recent_rollout": True}
    )
    kwargs = {"env_id": case.envs[-1], "n_envs": 2, **exposed, **case.kwargs}
    buffer = getattr(case.cls(**kwargs), "buffer", None)
    extensions = [
        ConditioningMetrics(buffer=buffer),
        BiasVoreDecomposition(buffer=buffer),
    ]
    if not case.critic:  # refused when built, naming what it lacks
        with pytest.raises(MissingCapability, match="needs critic"):
            case.cls(**kwargs, extensions=extensions)
        return
    agent = case.cls(**kwargs, extensions=extensions)
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    _, out = agent.train(
        seed=42,
        n_timesteps=case.n_timesteps,
        num_episode_test=1,
        logging_config=logging,
    )
    keys = COND_METRIC_KEYS + EVAREST_DECOMP_KEYS
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

    def recording(*args: Any, **kwargs: Any) -> Any:
        state = fresh_state(*args, **kwargs)
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
    """The policy's evaluation (the shared evaluate(), wherever it is
    called from) and the extensions' ``eval_metrics`` draw new keys at each
    evaluation of a run (``eval_rng`` folded with the count of evaluations
    so far): fresh initial states and noise every time."""

    shared = evaluate

    def recording(env: Any, actor_state: Any, num_episodes: int, rng: Any, *a, **kw):
        jax.debug.callback(_record("policy"), rng)
        return shared(env, actor_state, num_episodes, rng, *a, **kw)

    for module in list(sys.modules.values()):
        if getattr(module, "evaluate", None) is shared:
            monkeypatch.setattr(module, "evaluate", recording)
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
    for kind in ("policy", "extensions"):
        keys = _EVAL_KEYS[kind]
        assert len(keys) == evaluations, (kind, len(keys))
        assert len(set(keys)) == evaluations, kind


# --------------------------------------------------------------------------
# Continuous agents act on the env's whole box
# --------------------------------------------------------------------------
#: Each continuous agent on an env paying the action it receives, on a box
#: whose centre is not 0 and whose half-range is not 1, so that a map
#: dropping the offset (2a) or the scale (1 + a) pays another amount:
#: SAC's and TD3's first 64 steps warm up, and every run's 128 steps fill
#: the 128-row buffer (an unwritten row would read reward 0 for action 0).
ON_THE_BOX = ("APG", "APO", "ASAC", "AVG", "PPO", "REDQ", "SAC", "TD3", "UDRL")
BOX = (-1.0, 3.0)  # centre 1, half-range 2
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
    collected row is paid 1 + 2 * clip(a, -1, 1) for its stored action a,
    and evaluate() the deterministic action's (APG: its closed loop's).
    SAC's and TD3's uniform warm-up covers the box (half their paid lies
    over 1 from the centre)."""
    env, params = paid_action.make(BOX)
    if name == "APG":
        apg = APG(env, env_params=params, n_envs=4, horizon=1, pid=None)
        seeds, _ = apg.train(seed=0, n_timesteps=0)
        state: Any = jax.tree.map(lambda x: x[0], seeds)
        actor = state.actor_state
        _, rollout = rollout_returns(
            state, actor.params, jax.random.PRNGKey(1), apg.env_args, None, 1, 4
        )
        policy_step = make_policy_step(actor, actor.params, False)
        own, _ = policy_step(None, rollout.obs[0], rollout.resets[0])
        np.testing.assert_allclose(
            rollout.reward[0], paid_action.mapped(own, BOX)[:, 0], rtol=1e-6
        )
        return
    case = CASES[name]
    exposed = {"expose_recent_rollout": True} if case.rollout is not None else {}
    kwargs = case.kwargs | exposed | (WARM_UP if case.buffer else {})
    agent = case.cls(env_id=env, env_params=params, n_envs=1, **kwargs)
    state, _ = agent.train(seed=0, n_timesteps=128)
    actions, rewards = _paid_rows(name, state)
    assert np.abs(actions).max() > 0.01, "no action was collected"
    np.testing.assert_allclose(rewards, paid_action.mapped(actions, BOX), rtol=1e-6)
    if name in ("SAC", "TD3"):
        assert (np.abs(rewards[:64] - 1.0) > 1.0).mean() > 0.25
    seed = jax.tree.map(lambda x: x[0], state)
    obs, extra = paid_action.OBS[None], {}
    if name == "UDRL":  # it acts on its observation and its command
        config = agent.agent_config
        assert isinstance(config, UDRLConfig)
        command = EvalCommand(config.command_scale_r, config.command_scale_h)
        obs = command.actor_input(obs, command.initial(seed, 1))
        extra = {"agent_state": seed, "command": command}
    key = jax.random.PRNGKey(1)
    paid = evaluate(env, seed.actor_state, 2, key, params, **extra)[0]
    act = get_deterministic_action_and_entropy_fn(seed.actor_state, False, True)
    own, _, _ = act(obs, None, None)
    np.testing.assert_allclose(paid, paid_action.mapped(own, BOX)[0, 0], rtol=1e-6)


@pytest.mark.parametrize("name", ["DreamerV3", "TDMPC2"])
def test_a_world_model_acts_on_the_whole_box(name: str) -> None:
    """The world models step the env through the row collector, here on
    4-step episodes (TD-MPC2 refuses terminations): within an episode, the
    row after each stored raw action a is paid 1 + 2 * clip(a, -1, 1).
    Their evaluations, sampled or planned, are not read here
    (``evaluate_policy`` calls the collector's map)."""
    env, params = paid_action.make(BOX, length=4)
    if name == "DreamerV3":
        dreamer = DreamerV3(env, env_params=params, n_envs=1, **DREAMER_TINY)
        final = cast(DreamerV3State, dreamer.train(seed=0, n_timesteps=20)[0])
        replay = final.replay_state
        action, reward = replay.action[0, 0, :, 0], replay.reward[0, 0]
        paid = ~replay.is_last[0, 0, :-1]  # the rows an episode goes on from
        actions, rewards = action[:-1][paid], reward[1:][paid]
    else:
        tdmpc2 = TDMPC2(env, env_params=params, seed_steps=4, **TDMPC2_TINY)
        ring = tdmpc2.train(seed=0, n_timesteps=20)[0].buffer_state
        written = ring.obs[0, :, 0, 0] != 0  # not the staging round's slots
        actions = ring.action[0, written, :-1, 0].ravel()
        rewards = ring.reward[0, written, 1:].ravel()
    assert rewards.size >= 16 and np.abs(actions).max() > 0.01
    np.testing.assert_allclose(rewards, paid_action.mapped(actions, BOX), rtol=1e-6)


#: Every agent on ActorCritic (TD-MPC2 multi-task, offline, builds no env).
ACTOR_CRITICS = "APG APO ASAC AVG DQN DreamerV3 PPO PQN REDQ SAC TD3 TDMPC2 UDRL"


class _Built(Exception):
    """Stops a constructor once its env is asked for."""


@pytest.mark.parametrize("name", ACTOR_CRITICS.split())
def test_an_agent_forwards_its_episode_length_and_action_repeat(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every agent hands the env builder the episode length and the action
    repeat it is given (brax / playground envs honour both)."""
    seen: dict[str, Any] = {}

    def prepare_env(env_id: Any, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(base, "prepare_env", prepare_env)
    module = importlib.import_module(f"ajax.agents.{name}.{name}")
    with pytest.raises(_Built):
        getattr(module, name)("fast", episode_length=123, action_repeat=3)
    assert (seen["episode_length"], seen["action_repeat"]) == (123, 3)


def test_every_src_agent_is_in_actor_critics() -> None:
    """Test files subclass ActorCritic too, so only ``ajax.`` classes count."""
    for module in pkgutil.iter_modules(ajax.agents.__path__):
        if module.ispkg:
            importlib.import_module(f"ajax.agents.{module.name}.{module.name}")
    found = {
        c.__name__
        for c in base.ActorCritic.__subclasses__()
        if c.__module__.startswith("ajax.")
    }
    assert found == set(ACTOR_CRITICS.split()), "add the new agent to ACTOR_CRITICS"


def test_no_agent_class_lets_its_seeds_update_unequally() -> None:
    """Every agent's seeds update in step, so a resume refuses unequal
    counts; only TD-MPC2's episodic mode lets them differ, per instance
    (test_tdmpc2_episodic)."""
    unequal = {
        n for n in ACTOR_CRITICS.split() if getattr(ajax, n).per_seed_update_counts
    }
    assert not unequal


#: Each agent's gradient clip (its option, its default). PPO clips at its
#: paper's global norm 0.5, and APO at its official code's 10, each over
#: both networks at once (JOINT); PQN at purejaxql's 10; the other
#: references do not clip, and PPO's 0.5 copied onto them once froze SAC's
#: temperature (6b2d529). APG's
#: 1.0 is a recorded deviation (APG.py); the world models clip as their
#: papers do.
CLIPS: dict[str, tuple[str, float | None]] = {
    **dict.fromkeys("ASAC AVG DQN REDQ SAC TD3 UDRL".split(), ("max_grad_norm", None)),
    "PQN": ("max_grad_norm", 10.0),
    "APO": ("max_grad_norm", 10.0),
    "PPO": ("max_grad_norm", 0.5),
    "APG": ("max_grad_norm", 1.0),
    "DreamerV3": ("agc", 0.3),
    "TDMPC2": ("grad_clip_norm", 20.0),
    "TDMPC2MultiTask": ("grad_clip_norm", 20.0),
}
#: The agents whose clip is over the actor's and the critic's gradients at
#: once: their config holds it, and neither network's optimizer clips.
JOINT = {"APO", "PPO"}


@pytest.mark.parametrize(
    "name",
    [
        n
        for n in sorted(CASES)
        if "log_std_bounds" in inspect.signature(CASES[n].cls).parameters
    ],
)
@pytest.mark.parametrize("bounds", [(2.0, -20.0), (1.0, 1.0)])
def test_an_agent_refuses_log_std_bounds_out_of_order(
    name: str, bounds: tuple[float, float]
) -> None:
    """Swapped or equal bounds would clip every log-std to the high one,
    passed to the constructor or replaced on the agent's network config."""
    with pytest.raises(ValueError, match="low < high"):
        CASES[name].cls(env_id="Pendulum-v1", log_std_bounds=bounds)
    agent = CASES[name].cls(env_id="Pendulum-v1")
    with pytest.raises(ValueError, match="low < high"):
        agent.network_args.replace(log_std_bounds=bounds)


@pytest.mark.parametrize("name", ajax.__all__)
def test_an_agent_clips_its_gradients_as_its_paper_does(name: str) -> None:
    """A new agent needs its row in CLIPS. The table's agents carry the
    default to both optimizers, or to their config (JOINT)."""
    option, default = CLIPS[name]
    cls = getattr(ajax, name)
    assert inspect.signature(cls).parameters[option].default == default
    if name in CASES:
        agent = cls(env_id=CASES[name].envs[0])
        own = None if name in JOINT else default
        for args in (agent.actor_optimizer_args, agent.critic_optimizer_args):
            assert args.max_grad_norm == own
        if name in JOINT:
            assert getattr(agent.agent_config, option) == default


class _Stop(Exception):
    """Raised where a test has seen what it needs, before any compilation."""


def _stop(*args: Any, **kwargs: Any) -> Any:
    raise _Stop


def test_train_logs_its_run_config_without_touching_the_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One LoggingConfig reused across agents: each run logs the caller's
    keys and its own, and the caller's dict stays as it was, so PPO's run
    does not inherit SAC's keys."""
    from tests.agents.test_agent_pins import _multitask

    # The module, which its class's name shadows on the package.
    multitask_module = importlib.import_module("ajax.agents.TDMPC2.TDMPC2MultiTask")

    seen: list[dict] = []

    def record(run_id: str, config: LoggingConfig, run_seed: int) -> None:
        seen.append(config.config)

    monkeypatch.setattr(base, "init_logging", record)
    monkeypatch.setattr(multitask_module, "init_logging", record)
    logging = LoggingConfig(config={"user": 1}, use_wandb=False)
    for agent in (SAC(env_id="Pendulum-v1"), PPO(env_id="Pendulum-v1")):
        monkeypatch.setattr(agent, "get_make_train", lambda: _stop)
        with pytest.raises(_Stop):
            agent.train(seed=0, n_timesteps=8, logging_config=logging)
    multitask = _multitask()
    monkeypatch.setattr(multitask, "_init", _stop)
    with pytest.raises(_Stop):
        multitask.train(seed=0, n_timesteps=8, logging_config=logging)
    assert logging.config == {"user": 1}
    assert [c["algo_name"] for c in seen] == ["SAC", "PPO", "TDMPC2MultiTask"]
    assert all(c["user"] == 1 for c in seen)
    assert "buffer_size" in seen[0] and "buffer_size" not in seen[1]
