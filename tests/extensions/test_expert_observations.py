"""What SAC's expert machinery feeds its networks and its expert: the
augmented layout at every site (interaction.augment_obs), the expert's
stored actions, raw observations, and the normaliser's columns."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from gymnax.environments import environment, spaces

import ajax.agents.recurrent
import ajax.evaluate
import ajax.extensions.expert
import ajax.extensions.pretrain
import ajax.extensions.target_mods
import ajax.log
from ajax import SAC
from ajax.environments.obs_norm import apply_obs_norm, init_agent_obs_norm
from ajax.extensions.expert import ExpertValueMetrics, ExpertWarmup
from ajax.extensions.observe import (
    ExpertActionInput,
    ExpertStateInput,
    TrainingProgressInput,
)
from ajax.extensions.pretrain import (
    BellmanPretrain,
    CloningPretrain,
    ExpertPrefill,
    MCPretrain,
    PhiRefresh,
)
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
    calls since the episode's start (so t; with ``state_dim`` d, the j-th
    column j t). Called only with its state, unless ``stateless_too`` (then
    ``expert(obs)`` plays t / LIMIT too)."""

    def __init__(self, stateless_too: bool = False, state_dim: int = 1) -> None:
        self.stateless_too, self.state_dim = stateless_too, state_dim

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, self.state_dim))

    def __call__(self, *args: jax.Array) -> Any:
        if len(args) == 1 and self.stateless_too:
            return args[0][..., :1] / LIMIT
        state, obs = args
        return obs[..., :1] / LIMIT, state + jnp.arange(1.0, self.state_dim + 1)


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


class StatelessRecordingExpert:
    """RecordingExpert without a state (no init_state): an extension calls
    an expert of its own, not the agent's, without one."""

    def __init__(self, action: float) -> None:
        self._expert = RecordingExpert(action=action)
        self.seen = self._expert.seen

    def __call__(self, obs: jax.Array) -> jax.Array:
        return self._expert(obs)


def _record(monkeypatch, module: Any, name: str) -> list:
    """Record the critic inputs module.name (a predict_value) is given."""
    seen, original = [], getattr(module, name)

    def recording(*args: Any, **kw: Any) -> Any:
        x = kw["x"] if "x" in kw else args[2]
        jax.debug.callback(lambda v: seen.append(np.asarray(v)), x)
        return original(*args, **kw)

    monkeypatch.setattr(module, name, recording)
    return seen


#: Two programs computing the same float32 values may round apart in the last
#: place (CI's XLA orders sums unlike a laptop's). Scaling an actor's
#: parameters by 1 + 1e-6 moves an evaluation's return and entropy by under
#: 1e-6 (measured); a changed input column moves them by far more.
ROUNDING: dict[str, Any] = {"rtol": 1e-5, "atol": 1e-6}
EVALUATION = ("Eval/episodic mean reward", "Eval/episodic entropy")


@struct.dataclass
class ActsOn:
    """The actor handed prepare(obs) in place of each observation: the input
    an evaluation should hand it, built by hand."""

    actor: Any
    prepare: Any = struct.field(pytree_node=False)
    obs_norm_info: Any = None  # prepare normalises, where it should
    hidden_state: Any = None

    @property
    def params(self) -> Any:
        return self.actor.params

    def apply(self, params: Any, obs: jax.Array) -> Any:
        return self.actor.apply(params, self.prepare(obs))


def appending(*columns: float) -> Any:
    """A prepare appending the constant columns to each observation."""

    def prepare(obs: jax.Array) -> jax.Array:
        constants = jnp.broadcast_to(jnp.asarray(columns), (len(obs), len(columns)))
        return jnp.concatenate([obs, constants], axis=-1)

    return prepare


def assert_last_evaluation_acts_on(run: tuple, prepare: Any) -> None:
    """The run's last logged evaluation is its final actor's (the one
    evaluated) on the same episodes when handed prepare(obs)."""
    agent, state, evaluations = run
    state = jax.tree.map(lambda x: x[0], state)
    key = ajax.log.evaluation_keys(state, int(evaluations["timestep"][0, -1]))[0]
    env = agent.env_args
    actor = ActsOn(state.actor_state, prepare)
    out = ajax.evaluate.evaluate(env.env, actor, 3, key, env.env_params)
    logged = [evaluations[name][0, -1] for name in EVALUATION]
    np.testing.assert_allclose(logged, [out[0].mean(), out[1]], **ROUNDING)


def assert_trained_on_column(
    network_state: Any, column: int, value: float, of_column: Any = None
) -> None:
    """The network trained on inputs holding value (times column of_column,
    when given) in column at every row: by the chain rule through x w + b,
    its first layer's kernel row there has value times the bias's gradient
    (of_column's row's), and so Adam's running means of them."""
    optimiser = jax.tree.leaves(network_state.opt_state, is_leaf=_is_adam)
    means = next(state for state in optimiser if _is_adam(state)).mu
    layer = means["params"]["encoder"]["network"]["layers_0"]
    kernel, bias = np.asarray(layer["kernel"]), np.asarray(layer["bias"])
    reference = bias if of_column is None else kernel[of_column]
    assert np.abs(reference).max() > 0, "the network never trained"
    np.testing.assert_allclose(kernel[column], value * reference, **ROUNDING)


def _is_adam(node: Any) -> bool:
    return hasattr(node, "mu")  # Adam's state: its running means of gradients


def assert_same_run(run: tuple, reference: tuple) -> None:
    """run collected, trained and evaluated as reference: the same stored
    rows and actions (collection), networks (the update, whose targets read
    the actor at s') and logged evaluations."""
    (_, state, evaluations), (_, expected, expected_evaluations) = run, reference
    rows = state.collector_state.buffer_state.experience
    expected_rows = expected.collector_state.buffer_state.experience
    for name in ("obs", "next_obs", "action"):
        np.testing.assert_allclose(
            rows[name], expected_rows[name], err_msg=name, **ROUNDING
        )
    for network in ("actor_state", "critic_state"):
        trained = (getattr(state, network).params, getattr(expected, network).params)
        chex.assert_trees_all_close(*trained, **ROUNDING)
    for name in EVALUATION:
        np.testing.assert_allclose(
            evaluations[name], expected_evaluations[name], err_msg=name, **ROUNDING
        )


def train(env: Any = "Pendulum-v1", log: bool = False, **kw: Any) -> Any:
    return run(env, log, **kw)[1]


def run(env: Any = "Pendulum-v1", log: bool = False, **kw: Any) -> tuple:
    """A short SAC run: the agent, its state and its logged evaluations."""
    params = env.default_params if not isinstance(env, str) else None
    agent = SAC(
        env,
        env_params=params,
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        **kw,
    )
    logged = LoggingConfig(log_frequency=32, backends=())
    state, evaluations = agent.train(
        seed=0,
        n_timesteps=64,
        num_episode_test=3,
        logging_config=logged if log else None,
    )
    jax.effects_barrier()
    return agent, state, evaluations


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


class CounterObservingItsExpert(CounterEnv):
    """The counter observing at step t what CountingExpert's columns hold
    there: t, (its state, t,) its action t / LIMIT."""

    def __init__(self, with_state: bool) -> None:
        self.with_state = with_state

    def get_obs(self, state: Any, params: Any = None, key: Any = None) -> Any:
        t = jnp.asarray(state.time, jnp.float32)
        return jnp.stack([t, t, t / LIMIT] if self.with_state else [t, t / LIMIT])

    def observation_space(self, params: Any = None) -> spaces.Box:
        return spaces.Box(0.0, float(LIMIT), (2 + self.with_state,), dtype=jnp.float32)


@pytest.mark.parametrize("with_state", [False, True], ids=["action", "state+action"])
def test_every_actor_input_carries_its_own_rows_expert_state_and_action(with_state):
    """Acting, evaluated and trained (s and s'), every input is [t, (the
    expert's state, t), its action t / LIMIT]: the expert's action at that
    row's own observation, so s' carries a_E(s'), not a_E(s); with the
    state, state then action, as collection stores them. Stored so, trained
    on so, evaluated so, and the run goes as the counter observing those
    columns itself."""
    expert = CountingExpert()
    state = (ExpertStateInput.for_expert(expert),) if with_state else ()
    features = (*state, ExpertActionInput())
    # The actor steps from the first update, not after SAC's 2000 steps.
    options = {"log": True, "policy_update_start": 8}
    augmented = run(CounterEnv(), expert_policy=expert, extensions=features, **options)

    def own_columns(obs: jax.Array) -> jax.Array:
        t = obs[..., :1]
        return jnp.concatenate([t, t, t / LIMIT] if with_state else [t, t / LIMIT], -1)

    final = jax.tree.map(lambda x: x[0], augmented[1])
    rows = final.collector_state.buffer_state.experience
    for name in ("obs", "next_obs"):
        stored = rows[name][:, : 64 // 2]
        np.testing.assert_allclose(stored, own_columns(stored), err_msg=name)
    assert_trained_on_column(final.actor_state, -1, 1 / LIMIT, of_column=0)
    if with_state:
        assert_trained_on_column(final.actor_state, 1, 1.0, of_column=0)
    assert_last_evaluation_acts_on(augmented, own_columns)
    reference = run(CounterObservingItsExpert(with_state), **options)
    assert_same_run(augmented, reference)


def test_expert_value_metrics_query_the_critic_in_the_training_layout(monkeypatch):
    """ExpertValueMetrics' critic inputs are [t, a_E(t), action]."""
    seen = _record(monkeypatch, ajax.extensions.expert, "predict_value")
    train(
        CounterEnv(),
        log=True,
        expert_policy=CountingExpert(),
        extensions=(ExpertValueMetrics(), ExpertActionInput()),
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
        extensions=(_mc(expert), TrainingProgressInput(), ExpertActionInput()),
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
        normalize_observations=True,
        extensions=(_mc(expert), ExpertActionInput()),
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
    bellman = "Bellman" in repr(stack(expert))
    prefill = (ExpertPrefill(n_steps=16),) if bellman else ()
    extensions = (*stack(expert), ExpertActionInput(), *prefill)
    state = train(expert_policy=expert, extensions=extensions)
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


def test_cloning_statistics_leave_the_training_progress_column_raw():
    """Cloning standardises its dataset but not the training-progress
    column: constant 0 before training, its std would be 1e-6."""
    from ajax.agents.cloning import bc_statistics

    obs = jnp.stack([jnp.linspace(-2.0, 2.0, 64), jnp.zeros(64)], axis=-1)
    mean, std = bc_statistics(obs, pinned=(1,))
    np.testing.assert_allclose(mean, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(std[1], 1.0)
    assert std[0] > 1.0  # the other columns are standardised


def own_statistics(kind: str, expert: Any, *extensions: Any) -> dict:
    """The agent's own observation statistics: running, or seeded by
    cloning (CloningPretrain, ahead of the other pretraining)."""
    if kind == "running":
        return {"normalize_observations": True, "extensions": extensions}
    return {"extensions": (CloningPretrain(expert, n_steps=64), *extensions)}


@pytest.mark.parametrize("kind", ["running", "cloning"])
def test_normalised_agents_train_with_augmentation(kind):
    """With the agent's own normaliser (running, or seeded by cloning) and
    augmentation, the statistics cover the agent's observation (with the
    training fraction: 4 columns); the expert's column reaches the actor
    unnormalised, in agent units (its -0.9 torque of Pendulum's [-2, 2]:
    -0.45): stored so, and trained on so after the normaliser."""
    expert = RecordingExpert(action=-0.9)
    features = (TrainingProgressInput(), ExpertActionInput())
    own = own_statistics(kind, expert, *features)
    # The actor steps from the first update, not after SAC's 2000 steps.
    state = jax.tree.map(
        lambda x: x[0], train(expert_policy=expert, policy_update_start=8, **own)
    )
    assert state.actor_state.obs_norm_info.mean.shape[-1] == 4
    rows = state.collector_state.buffer_state.experience
    for name in ("obs", "next_obs"):
        np.testing.assert_allclose(rows[name][:, : 64 // 2, -1], -0.45, err_msg=name)
    assert_trained_on_column(state.actor_state, 4, -0.45)


@pytest.mark.parametrize("kind", ["running", "cloning"])
def test_phi_star_is_queried_as_it_was_fitted(kind):
    """With the agent's own observation statistics (running, or seeded by
    cloning), phi* is still fitted on observations as stored; ValueBox's
    expert value and MCVarianceCorrection's target read it the same way."""
    from ajax.extensions.target_mods import (
        box_compute_state,
        mc_correction_modify_target,
    )
    from ajax.networks.networks import predict_value

    expert = RecordingExpert(action=-0.5)
    state = train(expert_policy=expert, **own_statistics(kind, expert, _mc(expert)))
    first = lambda tree: jax.tree.map(lambda x: x[0], tree)  # noqa: E731
    critic, phi = first(state.critic_state), first(state.expert_critic_params)
    assert critic.obs_norm_info is not None
    obs = jnp.array([[1.0, 0.0, 5.0], [0.0, 1.0, -6.0]])
    a = jnp.full((2, 1), -0.5)
    x = jnp.concatenate([obs, a], axis=-1)
    fitted = predict_value(critic.replace(obs_norm_info=None), phi, x).min(0)
    through_stats = predict_value(critic, phi, x).min(0)
    assert not np.allclose(fitted, through_stats, atol=1e-3)  # the statistics matter
    v_box = box_compute_state(obs, a, critic, phi, jnp.float32(-1e9), None)[2]
    target = mc_correction_modify_target(
        jnp.zeros((2, 1)), critic, phi, obs, a, jnp.full((2,), 1e9), 0.0
    )
    np.testing.assert_allclose(v_box, fitted, rtol=1e-6)
    np.testing.assert_allclose(target, fitted, rtol=1e-6)


READERS = {  # each reader of phi*, by the function applying it
    "OnlineBC": "compute_online_bc_loss",
    "ExpertValueMetrics": "eval_metrics",
    "online light pretraining": "light_loss_fn",
    "CriticBlend": "on_target",
    "MCVarianceCorrection": "mc_correction_modify_target",
    "EDGE": "choose_by_score",
    "ValueBox": "box_compute_state",
}


@pytest.fixture(scope="module")
def phi_star_readers() -> dict:
    """The functions that applied phi* through phi_star_critic_state while
    SAC trained with the agent's own statistics and every reader of phi*,
    and whether the critic state they stripped held statistics."""
    import sys

    import ajax.extensions.exploration
    from ajax.extensions.expert import OnlineBC
    from ajax.extensions.exploration import EDGEExploration
    from ajax.extensions.target_mods import CriticBlend, MCVarianceCorrection
    from ajax.networks.networks import phi_star_critic_state

    readers: dict = {}

    def recording(critic_state: Any) -> Any:
        caller = sys._getframe(1).f_code.co_name
        readers[caller] = critic_state.obs_norm_info is not None
        return phi_star_critic_state(critic_state)

    expert = RecordingExpert(action=-0.5)
    stack: tuple = (_mc(expert), OnlineBC(), CriticBlend(expert_policy=expert))
    stack += (MCVarianceCorrection(threshold=0.0), ExpertValueMetrics())
    stack += (EDGEExploration(gate="lcb"), ValueBox())
    modules: tuple = (ajax.extensions.expert, ajax.extensions.target_mods)
    modules += (ajax.extensions.pretrain, ajax.extensions.exploration)
    with pytest.MonkeyPatch.context() as patch:
        for module in modules:
            patch.setattr(module, "phi_star_critic_state", recording)
        train(
            log=True,
            expert_policy=expert,
            normalize_observations=True,
            extensions=stack,
        )
    return readers


@pytest.mark.parametrize("reader", list(READERS))
def test_every_phi_star_reader_strips_the_agents_statistics(phi_star_readers, reader):
    """With the agent's own statistics on, each reader applies phi* through
    phi_star_critic_state, which drops them (phi* is fitted without)."""
    assert phi_star_readers.get(READERS[reader]) is True, phi_star_readers


# --- the evaluation's experts ---------------------------------------------------


def test_the_actor_is_evaluated_with_the_training_experts_column():
    """The evaluation builds the actor's input with the training expert
    (-0.9, in agent units -0.45); eval_expert_policy (0.5) only rolls out
    the baseline."""
    baseline = RecordingExpert(action=0.5)
    evaluated = run(
        log=True,
        expert_policy=RecordingExpert(action=-0.9),
        eval_expert_policy=baseline,
        extensions=(ExpertActionInput(),),
    )
    assert baseline.seen
    assert_last_evaluation_acts_on(evaluated, appending(-0.45))


@pytest.mark.parametrize("name", ["PhiRefresh", "BellmanPretrain"])
def test_replay_readers_augment_each_observation_with_its_own_expert_action(
    monkeypatch, name
):
    """PhiRefresh and BellmanPretrain augment a replay row's s and s' with
    the expert's stored actions there (augment_from_batch): every critic
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
    prefill = ExpertPrefill(n_steps=16)
    extensions = (*stack, ExpertActionInput(), prefill)
    train(CounterEnv(), expert_policy=expert, extensions=extensions)
    assert sac and (phi_star or name == "BellmanPretrain")  # phi*: PhiRefresh's
    for x in sac + phi_star:
        np.testing.assert_allclose(x[..., 1], x[..., 0] / LIMIT)


# --- the raw next observations an extension's expert reads ----------------------


def test_raw_next_observations_are_each_rows_successor(monkeypatch):
    """On the counter, every stored row's raw_next_obs is its raw_obs + 1
    (the final observation at the time limit), the expert prefill's rows
    too, and IBRL's expert reads s' of its batch's rows: never 0, the first
    observation, which no transition leads to."""
    ibrl_expert = StatelessRecordingExpert(action=0.5)
    state = train(
        CounterEnv(),
        expert_policy=RecordingExpert(action=0.5),
        extensions=(IBRL(expert_policy=ibrl_expert), ExpertPrefill(n_steps=8)),
    )
    rows = state.collector_state.buffer_state.experience
    raw, raw_next = (np.asarray(rows[k])[..., 0] for k in ("raw_obs", "raw_next_obs"))
    written = raw_next != 0  # every transition's s' is at least 1
    assert written.sum() >= 2 * (8 + 64 // 2)
    np.testing.assert_array_equal(raw_next[written], raw[written] + 1)
    read = np.concatenate([x[..., 0] for x in ibrl_expert.seen if len(x) == 8])
    assert read.size and read.min() >= 1 and read.max() <= LIMIT


class _ShiftedIntegratingExpert:
    """Stateful, called only with its state, its action reading it: an
    integrator of the observations since the episode's start, playing
    u / LIMIT - 3 - I / 200 at observation u with I its integral before the
    call, outside the action box so the critic inputs it builds stand
    apart. On the counter its action at u is shifted_integrated(u)."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        u = obs[..., :1]
        return u / LIMIT - 3.0 - state / 200.0, state + u


def shifted_integrated(u: Any) -> Any:
    return u / LIMIT - 3.0 - u * (u - 1) / 400.0


@pytest.mark.parametrize(
    "name", ["IBRL", "LCBGatedBootstrap", "CriticBlend", "PhiRefresh"]
)
def test_an_extension_reads_a_stateful_agent_experts_stored_actions(monkeypatch, name):
    """An extension given the agent's stateful expert reads the actions
    collection stored for it, from its tracked state: its critic inputs at
    s' in the replay batches it reads (8 rows: the update's, PhiRefresh's
    refresh) are [s', shifted_integrated(s')], which a zero, reset or
    pre-call state breaks. Such inputs must exist (PhiRefresh refreshed)."""
    from ajax.extensions.target_mods import CriticBlend, LCBGatedBootstrap

    module = (
        ajax.extensions.pretrain
        if name == "PhiRefresh"
        else ajax.extensions.target_mods
    )
    seen = _record(monkeypatch, module, "predict_value")
    expert = _ShiftedIntegratingExpert()
    stack = {
        "IBRL": (IBRL(expert_policy=expert),),
        "LCBGatedBootstrap": (LCBGatedBootstrap(expert_policy=expert),),
        "CriticBlend": (_mc(expert), CriticBlend(expert_policy=expert)),
        "PhiRefresh": (
            _mc(expert),
            PhiRefresh(expert_policy=expert, interval=16, steps=2),
        ),
    }[name]
    train(CounterEnv(), expert_policy=expert, extensions=stack)
    rows = np.concatenate([x for x in seen if len(x) == 8])  # replay batches
    expert_rows = rows[rows[:, -1] < -1.5]
    assert len(expert_rows) >= 8
    np.testing.assert_allclose(
        expert_rows[:, -1], shifted_integrated(expert_rows[:, 0]), atol=1e-5
    )


SETTINGS = "the same expert as its expert_policy"


@pytest.mark.parametrize("agent", ["SAC", "TD3"])
def test_an_extension_refuses_a_stateful_expert_without_its_state(agent):
    """IBRL on SAC without that expert, and ImitationLoss on TD3 (which
    stores no expert action), would call a stateful expert without its
    state: refused at construction, naming the settings that keep only the
    extension's use of the expert. A stateless expert stays allowed."""
    from ajax.agents.TD3.TD3 import TD3
    from ajax.extensions.expert import ImitationLoss

    stateful, stateless = _ShiftedIntegratingExpert(), _ShiftedExpert()
    with pytest.raises(ValueError, match=SETTINGS):
        if agent == "SAC":
            SAC("Pendulum-v1", extensions=(IBRL(expert_policy=stateful),))
        else:
            TD3("Pendulum-v1", extensions=(ImitationLoss(expert_policy=stateful),))
    if agent == "SAC":
        SAC("Pendulum-v1", extensions=(IBRL(expert_policy=stateless),))
    else:
        TD3("Pendulum-v1", extensions=(ImitationLoss(expert_policy=stateless),))


def test_the_named_settings_keep_only_the_extensions_use_of_the_expert():
    """Following the refusal (the stateful expert given to SAC with no
    prefill and no warm-up mixing) trains exactly as the extension using
    the same expert statelessly, with SAC holding no expert."""

    def act(obs: jax.Array) -> jax.Array:
        return jnp.tanh(jnp.sum(obs, axis=-1, keepdims=True))

    class Stateful:
        def init_state(self, n_envs: int) -> jax.Array:
            return jnp.zeros((n_envs, 1))

        def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
            return act(obs), state

    alone = train(extensions=(IBRL(expert_policy=act),))
    expert = Stateful()
    stored = train(expert_policy=expert, extensions=(IBRL(expert_policy=expert),))
    for name in ("actor_state", "critic_state"):
        a, b = (jax.tree.leaves(getattr(s, name).params) for s in (alone, stored))
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)


class _ShiftedExpert:
    """Stateless, plays t / LIMIT - 3 at observation t: outside the action
    box, so its actions tell the critic inputs it builds from any other."""

    def __call__(self, obs: jax.Array) -> jax.Array:
        return obs[..., :1] / LIMIT - 3.0


@pytest.mark.parametrize("name", ["IBRL", "PhiRefresh"])
def test_an_extensions_expert_reads_the_raw_next_observation_of_each_row(
    monkeypatch, name
):
    """The extension's critic input at s' is [s' of the row, a_E(raw s' of
    the same row)]: with a_E(t) = t / LIMIT - 3, every such input satisfies
    a = s' / LIMIT - 3, which a raw s' read from another row (another
    sampling key) breaks. IBRL reads it from the update's target batch,
    PhiRefresh from its own replay batch."""
    module = ajax.extensions.target_mods if name == "IBRL" else ajax.extensions.pretrain
    seen = _record(monkeypatch, module, "predict_value")
    expert = _ShiftedExpert()
    stack = {
        "IBRL": (IBRL(expert_policy=expert),),
        "PhiRefresh": (
            _mc(expert),
            PhiRefresh(expert_policy=expert, interval=16, steps=2),
        ),
    }[name]
    train(CounterEnv(), extensions=stack)
    rows = np.concatenate([x.reshape(-1, x.shape[-1]) for x in seen])
    expert_rows = rows[rows[:, -1] < -1.5]
    assert len(expert_rows) >= 8
    np.testing.assert_allclose(expert_rows[:, -1], expert_rows[:, 0] / LIMIT - 3.0)


@pytest.mark.parametrize(
    "state_dim, train_frac", [(1, True), (2, False)], ids=["1d+fraction", "2d"]
)
def test_the_expert_prefill_stores_the_experts_state_as_collection_does(
    state_dim, train_frac
):
    """With ExpertStateInput, every stored row, the expert prefill's (its
    first LIMIT per env) and collection's, is [t, (the training fraction,)
    the expert's state t] at s and [t + 1, (...), t + 1] at s', the
    episode's last s' included (the state before the reset)."""
    expert = CountingExpert(state_dim=state_dim)
    progress = (TrainingProgressInput(),) if train_frac else ()
    features = (*progress, ExpertStateInput.for_expert(expert))
    expert_steps = (ExpertWarmup(fraction=0.7), ExpertPrefill(n_steps=LIMIT))
    extensions = (*features, *expert_steps)
    state = train(CounterEnv(), expert_policy=expert, extensions=extensions)
    rows = state.collector_state.buffer_state.experience
    obs, nxt = (np.asarray(rows[k]) for k in ("obs", "next_obs"))
    obs, nxt = (x.reshape(-1, *x.shape[-2:]) for x in (obs, nxt))  # (envs, rows, .)
    assert obs.shape[-1] == nxt.shape[-1] == 1 + train_frac + state_dim
    written = nxt[..., 0] != 0  # every transition's s' is at least 1
    assert written[:, :LIMIT].all()  # the prefill's rows come first
    for x in (obs, nxt):
        for j in range(state_dim):
            got = x[..., 1 + train_frac + j][written]
            np.testing.assert_allclose(got, (j + 1) * x[..., 0][written])
        if train_frac:  # 0 in the prefill, a fraction after
            np.testing.assert_allclose(x[:, :LIMIT, 1], 0.0)
            assert (x[..., 1][written] <= 1.0).all()


class IntegratingExpert:
    """Stateful, its action reading its state: an integrator of the
    observations since the episode's start, playing u / LIMIT - I / 200
    at observation u with I its integral before the call (so, on the
    counter, A(u) = u / LIMIT - u (u - 1) / 400)."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
        return obs[..., :1] / LIMIT - state / 200.0, state + obs[..., :1]


def integrated_action(u: Any) -> Any:
    return u / LIMIT - u * (u - 1) / 400.0


def test_stored_expert_actions_come_from_the_experts_tracked_state():
    """Every stored row, the prefill's and collection's, holds the action
    the integrating expert took at s from its state there, a_E = A(s), and
    at s' from its state after the step, A(s + 1): a zero, reset or
    pre-call state stores another."""
    prefill = (ExpertPrefill(n_steps=LIMIT),)
    state = train(CounterEnv(), expert_policy=IntegratingExpert(), extensions=prefill)
    rows = state.collector_state.buffer_state.experience
    s, s_next, a, a_next = (
        np.asarray(rows[k])[..., 0].reshape(-1)
        for k in ("raw_obs", "next_obs", "a_expert", "next_a_expert")
    )
    written = s_next != 0
    assert written.sum() >= 2 * LIMIT
    np.testing.assert_allclose(a[written], integrated_action(s[written]), atol=1e-6)
    np.testing.assert_allclose(
        a_next[written], integrated_action(s[written] + 1), atol=1e-6
    )


def test_mc_pretraining_fits_phi_star_with_the_experts_state_columns(monkeypatch):
    """With ExpertStateInput (and ExpertActionInput, and ValueBox querying
    phi* at acting time), phi*'s inputs are the online critic's:
    [t, 0 (the fraction), the integrating expert's state t (t - 1) / 2, its
    action A(t), action]."""
    seen = _record(monkeypatch, ajax.extensions.pretrain, "predict_value")
    expert = IntegratingExpert()
    train(
        CounterEnv(),
        expert_policy=expert,
        extensions=(
            _mc(expert),
            ValueBox(),
            TrainingProgressInput(),
            ExpertStateInput.for_expert(expert),
            ExpertActionInput(),
        ),
    )
    assert seen
    for x in seen:
        t = x[..., 0]
        assert x.shape[-1] == 5
        np.testing.assert_allclose(x[..., 1], 0.0)
        np.testing.assert_allclose(x[..., 2], t * (t - 1) / 2)
        np.testing.assert_allclose(x[..., 3], integrated_action(t), atol=1e-6)


@pytest.mark.parametrize("own", ["stateless", "another stateful"])
def test_mc_pretraining_refuses_another_expert_under_state_augmentation(own):
    """phi*'s state columns are the agent's expert's: MCPretrain rolling out
    another expert (a stateless one, or another stateful one of the same
    width) is refused under ExpertStateInput."""
    stateless = lambda obs: jnp.full((*obs.shape[:-1], 1), 0.5)  # noqa: E731
    expert = stateless if own == "stateless" else IntegratingExpert()
    agent_expert = IntegratingExpert()
    extensions = (_mc(expert), ExpertStateInput.for_expert(agent_expert))
    with pytest.raises(ValueError, match="MC rollout must be that expert's"):
        train(CounterEnv(), expert_policy=agent_expert, extensions=extensions)
