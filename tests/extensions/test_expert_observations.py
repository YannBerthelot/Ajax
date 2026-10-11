"""What SAC's expert machinery feeds its networks and its expert: the
augmented layout at every site (interaction.augment_obs), the expert's
stored actions, raw observations, and the normaliser's columns."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import environment, spaces

import ajax.agents.recurrent
import ajax.environments.interaction
import ajax.evaluate
import ajax.extensions.expert
import ajax.extensions.pretrain
from ajax import SAC
from ajax.extensions.pretrain import MCPretrain
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
    """Stateful, called only with its state: plays t / LIMIT at observation
    t, its state counting its calls since the episode's start (so t)."""

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, state: jax.Array, obs: jax.Array) -> tuple:
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
