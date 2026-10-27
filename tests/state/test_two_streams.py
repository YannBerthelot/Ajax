"""Updates and collection draw on two streams of keys (ajax.random_keys):
the agent state's rng and the collector's. An update that draws a key more
leaves the collection's stream, and so its env steps, action noise and
warm-up, as they were; forced resets draw on a key derived from the
collection's and move neither stream."""

from typing import Any

import jax
import numpy as np
import pytest

import ajax

NET = {"actor_architecture": ("8", "relu"), "critic_architecture": ("8", "relu")}
PENDULUM = {"env_id": "Pendulum-v1", "n_envs": 2, **NET}
REPLAY = {**PENDULUM, "buffer_size": 64, "batch_size": 4, "learning_starts": 8}
ON_POLICY = {"env_id": "Pendulum-v1", "n_envs": 2, "n_steps": 8, "batch_size": 8}
ON_POLICY.update({"n_epochs": 1})
ON_POLICY.update(
    {"actor_architecture": ("8", "tanh"), "critic_architecture": ("8", "tanh")}
)
UDRL = {"env_id": "Pendulum-v1", "n_envs": 2, "actor_architecture": ("8", "relu")}
UDRL.update(
    {"n_steps": 8, "batch_size": 8, "buffer_capacity": 8, "n_updates_per_iter": 4}
)
DQN = {"env_id": "CartPole-v1", "n_envs": 2, "architecture": ("8", "relu")}
DQN.update({"buffer_size": 64, "batch_size": 4, "learning_starts": 8})

#: name: (the agent, its arguments, the method of its update, the env steps).
AGENTS: dict[str, tuple[Any, dict[str, Any], str, int]] = {
    "SAC": (ajax.SAC, {**REPLAY, "policy_update_start": 0}, "update", 24),
    "PPO": (ajax.PPO, ON_POLICY, "update", 48),
    "DQN": (ajax.DQN, DQN, "update", 24),
    "AVG": (ajax.AVG, {**PENDULUM, "learning_starts": 8}, "update", 24),
    "UDRL": (ajax.UDRL, UDRL, "iterate", 48),
}


def drawing_one_key_more(cls: Any, method: str) -> Any:
    """cls whose method draws one key more from the updates' stream, first."""

    def draw_then_run(self: Any, agent_state: Any, *args: Any, **kwargs: Any) -> Any:
        _, agent_state = agent_state.get_keys_and_new_state()
        return getattr(cls, method)(self, agent_state, *args, **kwargs)

    return type(f"{cls.__name__}DrawingOneKeyMore", (cls,), {method: draw_then_run})


def final_state(cls: Any, kwargs: dict[str, Any], steps: int) -> Any:
    state, _ = cls(**kwargs).train(seed=0, n_timesteps=steps)
    return state


@pytest.mark.parametrize("name", AGENTS)
def test_an_update_drawing_a_key_more_leaves_the_collections_keys(name: str) -> None:
    cls, kwargs, method, steps = AGENTS[name]
    plain = final_state(cls, kwargs, steps)
    more = final_state(drawing_one_key_more(cls, method), kwargs, steps)
    # The updates' stream moved; the collection's did not.
    assert not np.array_equal(plain.rng, more.rng)
    np.testing.assert_array_equal(plain.collector_state.rng, more.collector_state.rng)


def test_the_warmup_collects_the_same_rows_whatever_the_updates_draw() -> None:
    """SAC's uniform warm-up actions come from the collection's keys: the
    replay's rows before learning_starts are the same."""
    cls, kwargs, _, steps = AGENTS["SAC"]
    rows = [
        final_state(agent, kwargs, steps).collector_state.buffer_state.experience
        for agent in (cls, drawing_one_key_more(cls, "update"))
    ]
    # Each leaf is (seeds, envs, rows, ...).
    warmup = kwargs["learning_starts"] // kwargs["n_envs"]
    for plain, more in zip(
        jax.tree.leaves(rows[0]), jax.tree.leaves(rows[1]), strict=True
    ):
        np.testing.assert_array_equal(plain[:, :, :warmup], more[:, :, :warmup])
    # The actor's actions after it, which its updates changed, differ.
    actions = [experience["action"][:, :, warmup:] for experience in rows]
    assert not np.array_equal(*actions)


def test_forced_resets_move_neither_stream() -> None:
    """PPO with brax's num_resets_per_eval resets every env on a key derived
    from the collection's: both streams end where they end without it."""
    resets = {"num_resets_per_eval": 1, "num_evals": 3}
    plain = final_state(ajax.PPO, ON_POLICY, 48)
    reset = final_state(ajax.PPO, {**ON_POLICY, **resets}, 48)
    assert not np.array_equal(
        plain.collector_state.last_obs, reset.collector_state.last_obs
    )
    np.testing.assert_array_equal(plain.rng, reset.rng)
    np.testing.assert_array_equal(plain.collector_state.rng, reset.collector_state.rng)
