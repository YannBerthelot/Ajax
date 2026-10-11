"""The extensions that call an expert of their own give it the environment's
own observations, as SAC's collection and evaluation give SAC's expert."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ajax import SAC
from ajax.extensions.pretrain import MCPretrain
from ajax.extensions.target_mods import CriticBlend


class _RecordingExpert:
    """A Pendulum expert recording every observation batch it is given; its
    action reads the last column, the angular velocity (raw in [-8, 8]).
    Callable stateless, ``expert(obs)``, and stateful, ``expert(state, obs)``."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def init_state(self, n_envs: int) -> jax.Array:
        return jnp.zeros((n_envs, 1))

    def __call__(self, *args):
        obs = args[-1]
        jax.debug.callback(lambda o: self.seen.append(np.asarray(o)), obs)
        action = jnp.clip(-obs[..., -1:] / 8.0, -1.0, 1.0)
        return (action, args[0]) if len(args) == 2 else action


def _mc(expert: _RecordingExpert) -> MCPretrain:
    """The expert critic CriticBlend and PhiRefresh read (256 rows at least:
    MC pretraining regresses in batches of 256)."""
    return MCPretrain(
        expert_policy=expert,
        n_mc_steps=200,
        n_mc_episodes=4,
        n_steps=20,
        online_light_steps=5,
    )


def _seen(expert: _RecordingExpert, *extensions, **kwargs) -> list[np.ndarray]:
    """Every observation batch ``expert`` is given while a tiny SAC, which
    has no expert of its own, trains on Pendulum with ``extensions``."""
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        extensions=extensions,
        **kwargs,
    )
    agent.train(seed=0, n_timesteps=64)
    jax.effects_barrier()
    assert 8 in {seen.shape[0] for seen in expert.seen}  # the update called it
    return expert.seen


def test_critic_blend_gives_its_expert_every_observation_column():
    """CriticBlend's expert reads s' with all of Pendulum's three columns
    (it dropped the last, the angular velocity, augmented or not)."""
    expert = _RecordingExpert()
    for seen in _seen(expert, _mc(expert), CriticBlend(expert_policy=expert)):
        assert seen.shape[-1] == 3, seen.shape
