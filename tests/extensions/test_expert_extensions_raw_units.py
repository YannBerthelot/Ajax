"""The extensions that call an expert of their own give it the environment's
own observations, as SAC's collection and evaluation give SAC's expert."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.extensions.pretrain import MCPretrain, PhiRefresh
from ajax.extensions.target_mods import IBRL, CriticBlend, LCBGatedBootstrap


class _RecordingExpert:
    """A stateless Pendulum expert recording every observation batch it is
    given; its action reads the last column, the angular velocity (raw in
    [-8, 8]). Stateless: an extension calls an expert of its own (not the
    agent's) without a state, and refuses a stateful one."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def __call__(self, obs):
        jax.debug.callback(lambda o: self.seen.append(np.asarray(o)), obs)
        return jnp.clip(-obs[..., -1:] / 8.0, -1.0, 1.0)


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


STACKS = {  # each extension, MC pretraining first where it reads phi*
    "IBRL": lambda e: (IBRL(expert_policy=e),),
    "LCBGatedBootstrap": lambda e: (LCBGatedBootstrap(expert_policy=e),),
    "CriticBlend": lambda e: (_mc(e), CriticBlend(expert_policy=e)),
    "PhiRefresh": lambda e: (_mc(e), PhiRefresh(expert_policy=e, interval=16, steps=2)),
}


@pytest.mark.parametrize("name", list(STACKS))
def test_the_extensions_expert_reads_raw_observations(name):
    """With the environment normalising observations and the agent
    conditioning on the training fraction, every observation batch the
    extension's expert is given, current or next, is Pendulum's own: three
    columns with cos^2 + sin^2 = 1, which normalised ones break."""
    expert = _RecordingExpert()
    kw = {"normalize_observations": True, "use_train_frac": True}
    for seen in _seen(expert, *STACKS[name](expert), **kw):
        assert seen.shape[-1] == 3, seen.shape
        np.testing.assert_allclose(np.square(seen[..., :2]).sum(-1), 1.0, atol=1e-4)


@pytest.mark.parametrize(
    "name, stored",
    [("IBRL", True), ("PhiRefresh", True), ("MCPretrain", False), ("metrics", False)],
)
def test_the_buffer_keeps_raw_next_observations_only_for_a_declared_reader(
    name, stored
):
    """raw_next_obs is allocated when an extension declares it reads it
    (Extension.buffer_fields), not for one that merely carries an expert."""
    from ajax.extensions.expert import ExpertValueMetrics

    expert = _RecordingExpert()
    stacks = {
        "IBRL": (IBRL(expert_policy=expert),),
        "PhiRefresh": (_mc(expert), PhiRefresh(expert_policy=expert)),
        "MCPretrain": (_mc(expert),),
        "metrics": (ExpertValueMetrics(),),
    }
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        expert_policy=expert,
        expert_buffer_n_steps=0,
        extensions=stacks[name],
    )
    state, _ = agent.train(seed=0, n_timesteps=0)
    experience = state.collector_state.buffer_state.experience
    assert ("raw_next_obs" in experience) == stored


def test_an_agent_rejects_an_extension_reading_a_field_it_does_not_store():
    from ajax import REDQ

    with pytest.raises(ValueError, match="raw_next_obs"):
        REDQ("Pendulum-v1", extensions=(IBRL(expert_policy=_RecordingExpert()),))


def test_an_offline_agent_rejects_an_extension_reading_a_buffer_field():
    """TDMPC2MultiTask runs the shared check too: it stores no extra buffer
    field, so PhiRefresh (a post_update it does fold) is rejected."""
    from ajax import TDMPC2MultiTask
    from ajax.agents.TDMPC2.dataset import TaskEpisodes, pool_tasks

    rng, length = np.random.default_rng(0), 10
    data = pool_tasks(
        [
            TaskEpisodes(
                rng.normal(size=(2, length + 1, 2)).astype(np.float32),
                np.zeros((2, length + 1, 1), np.float32),
                np.zeros((2, length + 1), np.float32),
                length,
                "task",
            )
        ]
    )
    with pytest.raises(ValueError, match="raw_next_obs"):
        TDMPC2MultiTask(
            data, extensions=(PhiRefresh(expert_policy=_RecordingExpert()),)
        )
