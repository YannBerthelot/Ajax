import dataclasses
from functools import partial
from typing import Any

import distrax
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs import create as create_brax_env
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.classic_control.pendulum import Pendulum

from ajax import evaluate as evaluation
from ajax.agents.UDRL.UDRL import EvalCommand
from ajax.controller import Controller
from ajax.environments.interaction import step as environment_step
from ajax.evaluate import ValueEstimate, evaluate
from ajax.extensions.base import Extension
from ajax.extensions.expert import ExpertGains
from ajax.extensions.observe import ExpertActionInput, TrainingProgressInput
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import (
    EnvironmentConfig,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import EnvNormalizationInfo, NormalizationInfo
from ajax.wrappers import InitialStateWrapper
from tests.environments.identity_env import IdentityEnv
from tests.extensions.test_expert_observations import (
    appending,
    assert_last_evaluation_acts_on,
    assert_same_run,
    assert_trained_on_column,
    run,
)


def _sac_actor(env_config: EnvironmentConfig):
    """A freshly initialised squashed Gaussian actor (64-wide nets) on
    env_config, on the key SAC's init drew it from before B4 (the second of
    five): the regression values below are its."""
    optimizer_args = OptimizerConfig(learning_rate=3e-4)
    network_args = NetworkConfig(
        actor_architecture=("64", "relu", "64"),
        critic_architecture=("64", "relu", "64"),
        squash=True,
    )
    _, init_key, *_ = jax.random.split(jax.random.PRNGKey(0), 5)
    actor_state, _ = get_initialized_actor_critic(
        init_key,
        env_config,
        optimizer_args,
        optimizer_args,
        network_args,
        continuous=True,
        action_value=True,
        squash=True,
        num_critics=2,
    )
    return actor_state


@pytest.mark.parametrize("env_id", ["fast", "Pendulum-v1"])
def test_evaluate_returns_scalar_metrics(env_id):
    if env_id == "fast":
        env, env_params = create_brax_env("fast", batch_size=1), None
    else:
        env, env_params = gymnax.make(env_id)
    env_config = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=1, continuous=True
    )
    rewards, avg_entropy, *_ = evaluate(
        env=env,
        actor_state=_sac_actor(env_config),
        num_episodes=3,
        rng=jax.random.PRNGKey(42),
        env_params=env_params,
        recurrent=False,
    )
    assert rewards.shape == ()
    assert avg_entropy.shape == ()


def _eval_pendulum():
    env, env_params = gymnax.make("Pendulum-v1")
    env_cfg = EnvironmentConfig(
        env=env, env_params=env_params, n_envs=1, continuous=True
    )
    return env, env_params, _sac_actor(env_cfg)


def test_evaluate_regression_pendulum():
    """Pins eval numerical output on Pendulum-v1 so refactors (e.g. while_loop -> scan)
    don't silently change the reward computation."""
    env, env_params, actor = _eval_pendulum()
    rewards, entropy, _, _, length, _ = evaluate(
        env=env,
        actor_state=actor,
        num_episodes=5,
        rng=jax.random.PRNGKey(42),
        env_params=env_params,
        recurrent=False,
    )
    # Pendulum truncates at max_steps=200 for every episode with this policy.
    assert float(length) == 200.0
    assert abs(float(rewards) - (-1319.96)) < 5.0
    assert abs(float(entropy) - 0.4189) < 1e-2


def test_evaluate_done_masking_no_accumulation_past_done():
    """Rewards must not accumulate past per-env `done`. With Pendulum (fixed
    200-step episodes), running eval for longer than 200 steps must yield the
    same reward as running it for exactly 200 steps — this is the scan
    equivalent of while_loop's early exit, and the correctness condition the
    while_loop->scan conversion must preserve."""
    env, env_params, actor = _eval_pendulum()
    kwargs = {
        "env": env,
        "actor_state": actor,
        "num_episodes": 4,
        "rng": jax.random.PRNGKey(7),
        "env_params": env_params,
        "recurrent": False,
    }
    # Baseline (natural length — 200 for Pendulum).
    r_base, *_ = evaluate(**kwargs)
    # Over-long length (2x). If done-masking is correct, rewards match.
    r_long, *_ = evaluate(max_eval_steps=400, **kwargs)
    assert float(r_base) == pytest.approx(float(r_long), abs=1e-4)


def _zero_torque(obs):
    return jnp.zeros(obs.shape[:-1] + (1,))


@dataclasses.dataclass(frozen=True)
class _StatefulZeroTorque:
    """``_zero_torque`` in the stateful convention, its state unused."""

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, state, obs):
        return _zero_torque(obs), state


def test_the_expert_rollout_calls_an_expert_in_its_own_convention():
    """A stateless expert is called ``expert(obs)``, a stateful one
    ``expert(state, obs)``, as in the agent's rollout: one expert written
    both ways earns one return."""
    env, env_params, actor = _eval_pendulum()
    kwargs = {
        "env": env,
        "actor_state": actor,
        "num_episodes": 2,
        "rng": jax.random.PRNGKey(3),
        "env_params": env_params,
    }
    stateless = float(evaluate(expert_policy=_zero_torque, **kwargs)[-1])
    stateful = float(evaluate(expert_policy=_StatefulZeroTorque(), **kwargs)[-1])
    assert stateless < 0.0  # Pendulum's cost: every return is negative
    assert stateless == pytest.approx(stateful)


def _full_torque(obs):
    return jnp.ones(obs.shape[:-1] + (1,))


def test_the_baseline_return_is_the_evaluation_experts():
    """evaluate rolls out eval_expert_policy for the baseline,
    expert_policy building only the actor's input."""
    env, env_params, actor = _eval_pendulum()
    kwargs = {
        "env": env,
        "actor_state": actor,
        "num_episodes": 2,
        "rng": jax.random.PRNGKey(3),
        "env_params": env_params,
    }
    zero = float(evaluate(expert_policy=_zero_torque, **kwargs)[-1])
    full = float(evaluate(expert_policy=_full_torque, **kwargs)[-1])
    both = evaluate(
        expert_policy=_zero_torque, eval_expert_policy=_full_torque, **kwargs
    )
    assert zero != pytest.approx(full)
    assert float(both[-1]) == pytest.approx(full)


def test_setup_environment_accepts_prebuilt_gymnax_env_without_params():
    """Regression: a prebuilt gymnax env carries env_params=None (prepare_env)
    and the flattening check must fall back to the env's default params
    instead of handing None to observation_space (CartPole reads
    params.x_threshold there)."""
    from gymnax.environments.classic_control.cartpole import CartPole

    from ajax.evaluate import setup_environment

    env, mode, continuous = setup_environment(CartPole(), None, 2, None, 0.99)
    assert mode == "gymnax" and not continuous
    assert env.observation_space(env.default_params).shape == (4,)


_SEEN: list = []  # (batch size, observations) of every expert call


@dataclasses.dataclass(frozen=True)
class _RecordingExpert:
    """An expert recording what it is given; its action depends on the
    observation's scale (the angular velocity, raw in [-8, 8]). Stateful
    (``expert(state, obs)``), and callable as ``expert(obs)`` too, as SAC's
    update calls it to augment observations."""

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, *args):
        obs = args[-1]
        jax.debug.callback(lambda o: _SEEN.append(np.asarray(o)), obs)
        action = jnp.clip(-obs[..., 2:3] / 8.0, -1.0, 1.0)
        return (action, args[0]) if len(args) == 2 else action


@pytest.mark.parametrize("augment", [False, True], ids=["baseline", "augmented"])
def test_the_expert_sees_raw_observations_in_collection_and_evaluation(augment):
    """With observation normalisation on, collection and evaluation (the
    expert's own rollout, and with augmentation the agent's) give the expert
    Pendulum's raw observations: cos^2 + sin^2 = 1, which normalised ones
    break, and no training-fraction column. The update calls it on none: it
    reads the actions collection stored."""
    from ajax import SAC
    from ajax.logging.wandb_logging import LoggingConfig

    _SEEN.clear()
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        expert_policy=_RecordingExpert(),
        normalize_observations=True,
        extensions=(TrainingProgressInput(), ExpertActionInput()) if augment else (),
    )
    logged = LoggingConfig(log_frequency=32, backends=())
    agent.train(seed=0, n_timesteps=64, num_episode_test=3, logging_config=logged)
    jax.effects_barrier()
    # By batch size: collection (2 envs) and evaluation (3 episodes); never
    # the update (8 rows).
    assert {seen.shape[0] for seen in _SEEN} == {2, 3}
    for seen in _SEEN:
        assert seen.shape[-1] == 3
        np.testing.assert_allclose(np.square(seen[..., :2]).sum(-1), 1.0, atol=1e-4)


def test_the_actor_acts_and_trains_on_one_augmented_layout():
    """The expert's action (a constant -0.9 torque here, -0.45 in agent
    units) sits in the same column of every input the actor is given,
    acting, evaluated or trained, after Pendulum's three observations and
    the training fraction: in the rows collection stored, in what the actor
    trained on, and in its last evaluation (the fraction then 1)."""
    trained = run(
        log=True,
        expert_policy=lambda obs: jnp.full((*obs.shape[:-1], 1), -0.9),
        extensions=(TrainingProgressInput(), ExpertActionInput()),
        policy_update_start=8,  # the actor steps from the first update
    )
    state = jax.tree.map(lambda x: x[0], trained[1])
    rows = state.collector_state.buffer_state.experience
    for name in ("obs", "next_obs"):
        stored = rows[name][:, : 64 // 2]
        np.testing.assert_allclose(stored[..., 4], -0.45, err_msg=name)
        assert ((0.0 <= stored[..., 3]) & (stored[..., 3] <= 1.0)).all()  # the fraction
    assert_trained_on_column(state.actor_state, 4, -0.45)
    assert_last_evaluation_acts_on(trained, appending(1.0, -0.45))


@dataclasses.dataclass(frozen=True)
class _AlternatingExpert:
    """A stateful expert called only with its state, ``expert(state, obs)``,
    as a controller with an integrator is: it plays -0.9 and -0.8 in turn."""

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, state, obs):
        return jnp.where(state % 2 == 0, -0.9, -0.8), state + 1


class _PendulumObservingTheExpert(Pendulum):
    """Pendulum observing after its own three observations what the
    alternating expert plays at the episode's step t, in agent units: -0.45
    at even t, -0.4 at odd t."""

    def get_obs(self, state, params=None, key=None):
        played = jnp.where(state.time % 2 == 0, -0.45, -0.4)
        return jnp.append(super().get_obs(state), played)

    def observation_space(self, params):
        return spaces.Box(-jnp.inf, jnp.inf, (4,), jnp.float32)


def test_the_update_reads_the_stateful_experts_actions_from_the_buffer():
    """The update augments the observation and the next one with the
    actions collection stored, which the expert took from its state; it
    never calls the expert without its state. Every input the actor is
    given holds what the expert played there, -0.9 or -0.8 in turn (in
    agent units, -0.45 or -0.4): the run goes as Pendulum observing them
    itself."""
    # The actor steps from the first update, not after SAC's 2000 steps.
    options = {"log": True, "policy_update_start": 8}
    augmented = run(
        Pendulum(),
        expert_policy=_AlternatingExpert(),
        extensions=(ExpertActionInput(),),
        **options,
    )
    rows = augmented[1].collector_state.buffer_state.experience
    for name in ("obs", "next_obs"):
        played = np.asarray(rows[name])[0, :, : 64 // 2, -1].astype(float)
        played = np.round(played, 4)
        assert set(played.ravel().tolist()) == {-0.45, -0.4}, name
    assert_same_run(augmented, run(_PendulumObservingTheExpert(), **options))


# --------------------------------------------------------------------------
# Every output of evaluate() against its closed form
# --------------------------------------------------------------------------
STD = 0.5


@struct.dataclass
class _LinearActor:
    """An actor state whose policy at observation [t, c] is N(t / 10, STD)."""

    params: Any = None
    obs_norm_info: Any = None
    hidden_state: Any = None

    def apply(self, params, obs):
        mean = obs[..., :1] / 10
        return distrax.Normal(mean, jnp.full(mean.shape, STD))


def _fix_c(key, state, params, c):
    return state.replace(c=jnp.float32(c))


def _pays_one(obs):
    return jnp.ones(obs.shape[:-1] + (1,))


@pytest.mark.parametrize(
    "c, length, paid",
    [(0.5, 3, [0.0, 0.2, 0.4]), (-0.5, 5, [0.0, 0.2, 0.4, 0.6, 0.8])],
    ids=["terminated", "truncated"],
)
def test_every_output_of_evaluate_takes_its_closed_form(c, length, paid):
    """On the identity task with c fixed, the actor's mean t / 10 is paid
    2 t / 10 on Box(-2, 2) for ``length`` steps: its return is their sum,
    its entropy N(., STD)'s, its average reward the mean of the first 10^4
    steps of their repetition (the episodes restart), the expert's return
    one per step. Without a critic there is no value bias (NaN)."""
    env = InitialStateWrapper(IdentityEnv(), partial(_fix_c, c=c))
    out = evaluate(
        env,
        _LinearActor(),
        2,
        jax.random.PRNGKey(0),
        env.default_params,
        avg_reward_mode=True,
        expert_policy=_pays_one,
    )
    entropy = 0.5 * np.log(2 * np.pi * np.e * STD**2)
    average = np.mean(np.resize(paid, 10_000))
    expected = (sum(paid), entropy, average, np.nan, length, length)
    names = ("return", "entropy", "average reward", "bias", "length", "expert")
    # The average reward reduces 10^4 float32 rewards in an order XLA picks
    # per platform. Summed one by one, the least precise order, it is off by
    # 4e-6 (relative; measured).
    loose = {"average reward": (1e-4, 0.0)}
    for name, got, want in zip(names, out, expected, strict=True):
        rtol, atol = loose.get(name, (1e-5, 0.0))
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol, err_msg=name)


def _reward_std_2() -> EnvNormalizationInfo:
    """A training env's frozen reward statistics of std 2 (variance 16 / 4),
    for its 2 envs."""
    return EnvNormalizationInfo(
        reward=NormalizationInfo(
            var=jnp.full((2, 1), 4.0),
            count=jnp.full((2, 1), 4.0),
            mean=jnp.zeros((2, 1)),
            mean_2=jnp.full((2, 1), 16.0),
        )
    )


def _ten_times_the_action(agent_state, obs, action):
    """A plant critic: Q(s, a) = 10 a, t at the linear actor's action t / 10."""
    return 10 * action[..., 0]


@pytest.mark.parametrize(
    "c, length", [(0.5, 3), (-0.5, 5)], ids=["terminated", "truncated"]
)
def test_the_value_bias_is_the_critic_estimate_minus_the_realised_return(c, length):
    """The plant critic estimates t at step t; the actor is paid 0.2 t (as
    above), which the critic sees normalised by the training env's frozen
    reward scale (std 2) and scaled by 3, so 0.3 t. The
    value bias is the mean over the episode's steps of t minus the return
    realised from t, the rewards 0.3 k (k >= t) discounted by 0.9; the steps
    after the episode's end (the time limit of 5 restarts it) do not count."""
    env = InitialStateWrapper(IdentityEnv(), partial(_fix_c, c=c))
    bias = evaluate(
        env,
        _LinearActor(),
        2,
        jax.random.PRNGKey(0),
        env.default_params,
        norm_info=_reward_std_2(),
        value_estimate=ValueEstimate(
            _ten_times_the_action, gamma=0.9, reward_scale=3.0
        ),
    )[3]
    rewards = [0.3 * k for k in range(length)]
    realised = [
        sum(0.9 ** (k - t) * rewards[k] for k in range(t, length))
        for t in range(length)
    ]
    expected = np.mean([t - realised[t] for t in range(length)])
    np.testing.assert_allclose(bias, expected, rtol=1e-5)


class _GainExpertPayingZero:
    """A gain expert (``pid_gain_policy``) whose action is 0 whatever the
    gains: the actor's action, t / 10, is the gains' log-scale only."""

    learnable_fields = ("kp",)
    anchor_gains = jnp.array([1.0])

    def init_state(self, n_envs):
        return jnp.zeros((n_envs, 1))

    def __call__(self, state, obs):
        return self.step_with_gains(state, obs, self.anchor_gains)

    def step_with_gains(self, state, obs, gains):
        return jnp.zeros(obs.shape[:-1] + (1,)), state


def test_the_value_bias_reads_a_gain_policy_critic_at_the_gains_action():
    """Under gain mode collection stores the actor's action (the
    gains'), not the expert's executed one, so the critic is read at it: the
    plant critic estimates t at step t, the expert's action 0 pays 0, and the
    bias is the mean of t over the 3 steps, 1 (at the executed action, 0)."""
    env = InitialStateWrapper(IdentityEnv(), partial(_fix_c, c=0.5))
    bias = evaluate(
        env,
        _LinearActor(),
        2,
        jax.random.PRNGKey(0),
        env.default_params,
        expert_policy=_GainExpertPayingZero(),
        controller=Controller(transforms=(ExpertGains(),)),
        value_estimate=ValueEstimate(_ten_times_the_action, gamma=0.9),
    )[3]
    np.testing.assert_allclose(bias, 1.0, rtol=1e-6)


@struct.dataclass
class _CommandReader(_LinearActor):
    """An actor acting its command's scaled desired return, column 2 of
    its input [t, c, d_r / 10, d_h / 100]."""

    def apply(self, params, obs):
        mean = obs[..., 2:3]
        return distrax.Normal(mean, jnp.full(mean.shape, STD))


@struct.dataclass
class _Targets:
    """The command targets a UDRL state carries."""

    command_target_return: Any = 1.0
    command_target_horizon: Any = 5.0


@pytest.mark.parametrize(
    "c, length", [(0.5, 3), (-0.5, 5)], ids=["terminated", "truncated"]
)
def test_a_commanded_actor_acts_on_its_decaying_command(c, length):
    """UDRL's command from desired return 1, read by the actor as d_r / 10
    and paid twice that on Box(-2, 2): each step pays d_r / 5, which the
    command then loses, so the episode returns 1 - 0.8 ** length."""
    env = InitialStateWrapper(IdentityEnv(), partial(_fix_c, c=c))
    returns, _, _, _, steps, _ = evaluate(
        env,
        _CommandReader(),
        2,
        jax.random.PRNGKey(0),
        env.default_params,
        agent_state=_Targets(),
        command=EvalCommand(scale_r=0.1, scale_h=0.01),
    )
    np.testing.assert_allclose(returns, 1 - 0.8**length, rtol=1e-5)
    np.testing.assert_allclose(steps, length, rtol=1e-6)


@pytest.mark.parametrize(
    "c, length", [(0.5, 3), (-0.5, 5)], ids=["terminated", "truncated"]
)
def test_a_commanded_actor_decays_its_command_by_normalised_rewards(c, length):
    """With reward normalisation (frozen std 2) UDRL's targets, and so its
    command, are in normalised units: from the target 1 as it is, each step
    pays d_r / 5, d_r / 10 normalised, which the command loses. d_r falls by
    0.9 a step and the episode returns 2 (1 - 0.9 ** length)."""
    env = InitialStateWrapper(IdentityEnv(), partial(_fix_c, c=c))
    returns = evaluate(
        env,
        _CommandReader(),
        2,
        jax.random.PRNGKey(0),
        env.default_params,
        norm_info=_reward_std_2(),
        agent_state=_Targets(),
        command=EvalCommand(scale_r=0.1, scale_h=0.01),
    )[0]
    np.testing.assert_allclose(returns, 2 * (1 - 0.9**length), rtol=1e-5)


@dataclasses.dataclass(frozen=True)
class _HandsEnvZero(Extension):
    """A chooser handing the expert evaluation's steps in env 0."""

    name: str = "hands_env_zero"

    def eval_action(self, agent_state, ext_state, obs, rng, context):
        handed = jnp.arange(obs["executed_action"].shape[0]) == 0
        obs["expert_mask"] = handed
        return jnp.where(handed[:, None], obs["expert_action"], obs["executed_action"])


@struct.dataclass
class _OneStatelessExtension:
    ext_state: Any = ((),)


def test_a_chooser_replaces_the_executed_action_and_logs_its_share():
    """The chooser hands env 0 the expert's action, 1 on Box(-2, 2) (0.5 in
    agent units), which the identity task pays 1 a step; env 1 keeps the
    actor's mean t / 10, paid 0.2 t. A seventh output, the share of the
    evaluated steps it handed the expert: env 0's, half of them."""
    env = InitialStateWrapper(IdentityEnv(), partial(_fix_c, c=0.5))
    out = evaluate(
        env,
        _LinearActor(),
        2,
        jax.random.PRNGKey(0),
        env.default_params,
        expert_policy=_pays_one,
        agent_state=_OneStatelessExtension(),
        controller=Controller(choosers=((0, "hands_env_zero#0", _HandsEnvZero()),)),
    )
    np.testing.assert_allclose(out[0], (3 * 1.0 + 0.6) / 2, rtol=1e-5)
    np.testing.assert_allclose(out[6], 0.5)


def test_a_chooser_leaves_the_rollouts_keys(monkeypatch) -> None:
    """A chooser draws on keys derived from the step's (the action stage's):
    the environment steps on the keys it steps on without one."""
    stepped: dict[str, list] = {"plain": [], "chosen": []}
    env = InitialStateWrapper(IdentityEnv(), partial(_fix_c, c=0.5))
    chooser = (0, "hands_env_zero#0", _HandsEnvZero())
    for name, controller in (
        ("plain", Controller()),
        ("chosen", Controller(choosers=(chooser,))),
    ):

        def spied(keys: Any, *args: Any, name: str = name) -> Any:
            jax.debug.callback(lambda k: stepped[name].append(np.asarray(k)), keys)
            return environment_step(keys, *args)

        monkeypatch.setattr(evaluation, "step", spied)
        jax.clear_caches()  # a trace kept from before would step unseen
        evaluate(
            env,
            _LinearActor(),
            2,
            jax.random.PRNGKey(0),
            env.default_params,
            expert_policy=_pays_one,
            agent_state=_OneStatelessExtension(),
            controller=controller,
        )
        jax.effects_barrier()
    jax.clear_caches()  # no trace keeps the spy past the test
    assert stepped["plain"]
    np.testing.assert_array_equal(
        np.stack(stepped["plain"]), np.stack(stepped["chosen"])
    )
