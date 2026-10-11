"""What SAC's expert machinery feeds its networks and its expert: the
augmented layout at every site (interaction.augment_obs), the expert's
stored actions, raw observations, and the normaliser's columns."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from gymnax.environments import environment, spaces

import ajax.agents.recurrent
import ajax.environments.interaction
import ajax.evaluate
import ajax.extensions.expert
import ajax.extensions.pretrain
from ajax import SAC
from ajax.agents.obs_norm import apply_obs_norm, init_agent_obs_norm
from ajax.extensions.expert import ExpertValueMetrics
from ajax.extensions.pretrain import BellmanPretrain, MCPretrain, PhiRefresh
from ajax.extensions.target_mods import IBRL, ValueBox
from ajax.logging.wandb_logging import LoggingConfig

LIMIT = 8  # the counter's episode length


@struct.dataclass
class _Params(environment.EnvParams):
    max_steps_in_episode: int = LIMIT


class CounterEnv(environment.Environment):
    """Observes its step count t, raw (time limit LIMIT, reward 0): s, s'
    and every row of an episode read differently."""

    @property
    def default_params(self) -> _Params:
        return _Params()

    def step_env(self, key: Any, state: Any, action: Any, params: Any) -> Any:
        # A new state, not state.replace: the collector steps the training-
        # fraction view of the state, which replace would carry along.
        state = environment.EnvState(time=state.time + 1)
        return self.get_obs(state), state, jnp.float32(0.0), jnp.bool_(False), {}

    def reset_env(self, key: Any, params: Any) -> Any:
        state = environment.EnvState(time=0)
        return self.get_obs(state), state

    def get_obs(self, state: Any, params: Any = None, key: Any = None) -> Any:
        return jnp.asarray(state.time, jnp.float32).reshape(1)

    def action_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, (1,), dtype=jnp.float32)

    def observation_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(0.0, float(LIMIT), (1,), dtype=jnp.float32)

    @property
    def num_actions(self) -> int:
        return 1


class CountingExpert:
    """Stateful: plays t / LIMIT at observation t, its state counting its
    calls since the episode's start (so t). Called only with its state,
    unless ``stateless_too`` (then ``expert(obs)`` plays t / LIMIT too)."""

    def __init__(self, stateless_too: bool = False) -> None:
        self.stateless_too = stateless_too

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, *args: jax.Array) -> Any:
        if len(args) == 1 and self.stateless_too:
            return args[0][..., :1] / LIMIT
        state, obs = args
        return obs[..., :1] / LIMIT, state + 1


class RecordingExpert:
    """Records every observation batch it is given; plays ``action`` (or,
    on Pendulum, reads the angular velocity). ``stateful_only``: callable
    only as ``expert(state, obs)``."""

    def __init__(self, action: float | None = None, stateful_only: bool = False):
        self.action, self.stateful_only = action, stateful_only
        self.seen: list[np.ndarray] = []

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, *args: Any) -> Any:
        if self.stateful_only and len(args) != 2:
            raise TypeError("a stateful expert is called with its state")
        obs = args[-1]
        jax.debug.callback(lambda o: self.seen.append(np.asarray(o)), obs)
        if self.action is None:
            action = jnp.clip(-obs[..., -1:] / 8.0, -1.0, 1.0)
        else:
            action = jnp.full((*obs.shape[:-1], 1), self.action, jnp.float32)
        return (action, args[0]) if len(args) == 2 else action


def _record(monkeypatch, module: Any, name: str) -> list:
    """Record the observations (or critic inputs) ``module.name`` is given."""
    seen, original = [], getattr(module, name)
    arg = {"get_pi": ("obs", 2), "predict_value": ("x", 2)}[name]

    def recording(*args: Any, **kw: Any) -> Any:
        x = kw[arg[0]] if arg[0] in kw else args[arg[1]]
        jax.debug.callback(lambda v: seen.append(np.asarray(v)), x)
        return original(*args, **kw)

    monkeypatch.setattr(module, name, recording)
    return seen


def actor_inputs(monkeypatch) -> list:
    """Every input the actor is given: collection, evaluation, the update."""
    seen = _record(monkeypatch, ajax.environments.interaction, "get_pi")
    for module in (ajax.agents.recurrent, ajax.evaluate):
        monkeypatch.setattr(module, "get_pi", ajax.environments.interaction.get_pi)
    return seen


def train(env: Any = "Pendulum-v1", log: bool = False, **kw: Any) -> Any:
    params = env.default_params if not isinstance(env, str) else None
    agent = SAC(
        env,
        env_params=params,
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        **{"expert_buffer_n_steps": 0, **kw},
    )
    logged = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    out = agent.train(
        seed=0,
        n_timesteps=64,
        num_episode_test=3,
        logging_config=logged if log else None,
    )
    jax.effects_barrier()
    return out[0]


def _mc(expert: Any) -> MCPretrain:
    """At least 256 rows: MC pretraining regresses in batches of 256."""
    return MCPretrain(
        expert_policy=expert,
        n_mc_steps=200,
        n_mc_episodes=4,
        n_steps=20,
        online_light_steps=5,
    )


# --- the layout and the stored expert actions --------------------------------


@pytest.mark.parametrize("with_state", [False, True], ids=["action", "state+action"])
def test_every_actor_input_carries_its_own_rows_expert_state_and_action(
    monkeypatch, with_state
):
    """Acting, evaluated and trained (s and s'), every input is [t, (the
    expert's state, t), its action t / LIMIT]: the expert's action at that
    row's own observation, so s' carries a_E(s'), not a_E(s); with the
    state, state then action, as collection stores them."""
    seen = actor_inputs(monkeypatch)
    train(
        CounterEnv(),
        log=True,
        expert_policy=CountingExpert(),
        augment_obs_with_expert_action=True,
        augment_obs_with_expert_state=with_state,
    )
    assert {x.shape[0] for x in seen} == {2, 3, 8}
    for x in seen:
        t = x[..., 0]
        assert x.shape[-1] == 2 + with_state
        np.testing.assert_allclose(x[..., -1], t / LIMIT, err_msg=f"batch {len(x)}")
        if with_state:
            np.testing.assert_allclose(x[..., 1], t, err_msg=f"batch {len(x)}")


def test_expert_value_metrics_query_the_critic_in_the_training_layout(monkeypatch):
    """ExpertValueMetrics' critic inputs are [t, a_E(t), action]."""
    seen = _record(monkeypatch, ajax.extensions.expert, "predict_value")
    train(
        CounterEnv(),
        log=True,
        expert_policy=CountingExpert(),
        augment_obs_with_expert_action=True,
        extensions=(ExpertValueMetrics(),),
    )
    assert seen
    for x in seen:
        np.testing.assert_allclose(x[..., 1], x[..., 0] / LIMIT)


def test_mc_pretraining_fits_phi_star_in_the_training_layout(monkeypatch):
    """phi*'s regression inputs with the training fraction are [t, 0 (the
    fraction), a_E(t), the expert's action]: the expert column after it."""
    seen = _record(monkeypatch, ajax.extensions.pretrain, "predict_value")
    expert = CountingExpert()
    train(
        CounterEnv(),
        expert_policy=expert,
        augment_obs_with_expert_action=True,
        use_train_frac=True,
        extensions=(_mc(expert),),
    )
    assert seen
    for x in seen:
        np.testing.assert_allclose(x[..., 1], 0.0)
        np.testing.assert_allclose(x[..., 2], x[..., 0] / LIMIT)


def test_mc_pretraining_reads_the_experts_stored_actions():
    """MC pretraining augments phi*'s inputs with the actions its rollout
    stored, so a stateful expert is never called without its state and
    sees only Pendulum's raw observations (cos^2 + sin^2 = 1), the
    environment normalising them."""
    expert = RecordingExpert(stateful_only=True)
    train(
        expert_policy=expert,
        augment_obs_with_expert_action=True,
        normalize_observations=True,
        extensions=(_mc(expert),),
    )
    assert expert.seen
    for seen in expert.seen:
        assert seen.shape[-1] == 3
        np.testing.assert_allclose(np.square(seen[..., :2]).sum(-1), 1.0, atol=1e-4)


@pytest.mark.parametrize(
    "stack",
    [
        lambda e: (_mc(e), ValueBox()),
        lambda e: (_mc(e), PhiRefresh(expert_policy=e, interval=16, steps=2)),
        lambda e: (BellmanPretrain(n_steps=4),),
    ],
    ids=["ValueBox", "PhiRefresh", "BellmanPretrain"],
)
def test_extensions_querying_critics_train_with_augmentation(stack):
    """ValueBox (phi* at the acting observation), PhiRefresh (phi* on
    replay) and BellmanPretrain (the critic on replay) give the critics
    the augmented observation they were built for."""
    expert = RecordingExpert(action=-0.9)
    kw = {"expert_buffer_n_steps": 16} if "Bellman" in repr(stack(expert)) else {}
    state = train(
        expert_policy=expert,
        augment_obs_with_expert_action=True,
        extensions=stack(expert),
        **kw,
    )
    assert np.isfinite(np.asarray(jax.tree.leaves(state.critic_state.params)[0])).all()


# --- the observation normaliser ------------------------------------------------


def test_the_normaliser_leaves_appended_columns_raw():
    """Statistics over the agent's observation normalise it; the expert
    action appended after it (and a critic's action) stays raw."""
    info = init_agent_obs_norm(2)
    info = info.replace(
        count=jnp.ones((1, 1)), mean=jnp.ones((1, 2)), var=4 * jnp.ones((1, 2))
    )
    out = apply_obs_norm(jnp.array([[3.0, 5.0, 0.7]]), info)
    np.testing.assert_allclose(out, [[1.0, 2.0, 0.7]], atol=1e-6)


@pytest.mark.parametrize(
    "kw",
    [
        {"normalize_obs_running": True, "use_train_frac": True},
        {"pre_train_n_steps": 64},  # cloning has no training-fraction column
    ],
    ids=["running", "cloning"],
)
def test_normalised_agents_train_with_augmentation(monkeypatch, kw):
    """With the agent's own normaliser (running, or seeded by cloning) and
    augmentation, the statistics cover the agent's observation (with the
    training fraction); the expert's column reaches the actor raw (-0.9)."""
    seen = actor_inputs(monkeypatch)
    train(
        expert_policy=RecordingExpert(action=-0.9),
        augment_obs_with_expert_action=True,
        **kw,
    )
    for x in seen:
        np.testing.assert_allclose(x[..., -1], -0.9)


# --- the evaluation's experts ---------------------------------------------------


def test_the_actor_is_evaluated_with_the_training_experts_column(monkeypatch):
    """The evaluation builds the actor's input with the training expert
    (-0.9); eval_expert_policy (0.5) only rolls out the baseline."""
    seen = actor_inputs(monkeypatch)
    baseline = RecordingExpert(action=0.5)
    train(
        log=True,
        expert_policy=RecordingExpert(action=-0.9),
        eval_expert_policy=baseline,
        augment_obs_with_expert_action=True,
    )
    evaluated = [x for x in seen if len(x) == 3]
    assert evaluated and baseline.seen
    for x in evaluated:
        np.testing.assert_allclose(x[..., -1], -0.9)


@pytest.mark.parametrize("name", ["PhiRefresh", "BellmanPretrain"])
def test_replay_readers_augment_each_observation_with_its_own_expert_action(
    monkeypatch, name
):
    """PhiRefresh and BellmanPretrain augment a replay row's s and s' with
    the expert's stored actions there (augment_from_buffer): every critic
    input, theirs and the update's, is [t, t / LIMIT, action]."""
    sac = _record(monkeypatch, ajax.agents.recurrent, "predict_value")
    phi_star = _record(monkeypatch, ajax.extensions.pretrain, "predict_value")
    expert = CountingExpert(stateless_too=True)
    stack = {
        "PhiRefresh": (
            _mc(expert),
            PhiRefresh(expert_policy=expert, interval=16, steps=2),
        ),
        "BellmanPretrain": (BellmanPretrain(n_steps=4),),
    }[name]
    train(
        CounterEnv(),
        expert_policy=expert,
        expert_buffer_n_steps=16,
        augment_obs_with_expert_action=True,
        extensions=stack,
    )
    assert sac and (phi_star or name == "BellmanPretrain")  # phi*: PhiRefresh's
    for x in sac + phi_star:
        np.testing.assert_allclose(x[..., 1], x[..., 0] / LIMIT)


# --- the raw next observations an extension's expert reads ----------------------


def test_raw_next_observations_are_each_rows_successor(monkeypatch):
    """On the counter, every stored row's raw_next_obs is its raw_obs + 1
    (the final observation at the time limit), the expert prefill's rows
    too, and IBRL's expert reads s' of its batch's rows: never 0, the first
    observation, which no transition leads to."""
    ibrl_expert = RecordingExpert(action=0.5)
    state = train(
        CounterEnv(),
        expert_policy=RecordingExpert(action=0.5),
        expert_buffer_n_steps=8,
        extensions=(IBRL(expert_policy=ibrl_expert),),
    )
    rows = state.collector_state.buffer_state.experience
    raw, raw_next = (np.asarray(rows[k])[..., 0] for k in ("raw_obs", "raw_next_obs"))
    written = raw_next != 0  # every transition's s' is at least 1
    assert written.sum() >= 2 * (8 + 64 // 2)
    np.testing.assert_array_equal(raw_next[written], raw[written] + 1)
    read = np.concatenate([x[..., 0] for x in ibrl_expert.seen if len(x) == 8])
    assert read.size and read.min() >= 1 and read.max() <= LIMIT
