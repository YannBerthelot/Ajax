"""The replay family's evaluation metric: the share of the replay's written
rows an expert acted in (replay/expert_share), for every replay agent whose
replay keeps the flags (a run with an expert), and none for the others."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from ajax.agents.families import measure_replay
from ajax.buffers.utils import mean_of_written_rows
from ajax.extensions.expert import ExpertWarmup
from ajax.logging.wandb_logging import LoggingConfig
from tests.agents.test_agent_contract import CASES


def _replay(flags: list[list[float]], index: int, full: bool) -> SimpleNamespace:
    experience = {"is_expert": jnp.asarray(flags)[..., None]}
    return SimpleNamespace(
        experience=experience,
        current_index=jnp.asarray(index),
        is_full=jnp.asarray(full),
    )


def test_the_share_reads_the_written_rows_only() -> None:
    """Two envs, capacity 4: two rows written each, then the full replay."""
    flags = [[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    np.testing.assert_allclose(
        mean_of_written_rows(_replay(flags, 2, False), "is_expert"), 0.75
    )
    np.testing.assert_allclose(
        mean_of_written_rows(_replay(flags, 1, True), "is_expert"), 7 / 8
    )
    np.testing.assert_allclose(
        mean_of_written_rows(_replay(flags, 0, False), "is_expert"), 0.0
    )


def test_a_replay_without_flags_logs_nothing() -> None:
    collector = SimpleNamespace(buffer_state=SimpleNamespace(experience={"action": 0}))
    assert measure_replay(SimpleNamespace(collector_state=collector)) == {}


def _push(obs: jnp.ndarray) -> jnp.ndarray:
    return jnp.full(obs.shape[:-1] + (1,), 0.5)


@pytest.mark.parametrize("name", ["SAC", "TD3", "REDQ", "ASAC", "CrossQ", "DAgger"])
def test_every_replay_agent_with_an_expert_logs_its_replays_share(name: str) -> None:
    """The warm-up mix (DAgger: its beta, 1 in its first iteration) has the
    expert act in some rows: each evaluation logs their share, in [0, 1]."""
    case = CASES[name]
    kwargs = dict(case.kwargs)
    kwargs.update({"expert_policy": _push})
    if name != "DAgger":
        kwargs.update({"extensions": (ExpertWarmup(fraction=1.0),)})
    agent = case.cls(env_id="Pendulum-v1", n_envs=2, **kwargs)
    logging = LoggingConfig(config={}, log_frequency=16, use_wandb=False)
    _, logged = agent.train(
        seed=0, n_timesteps=case.n_timesteps, logging_config=logging
    )
    share = np.asarray(logged["replay/expert_share"])
    assert share.size and np.all((share > 0) & (share <= 1)), share


def test_a_run_without_an_expert_logs_no_share() -> None:
    case = CASES["TD3"]
    agent = case.cls(env_id="Pendulum-v1", n_envs=2, **case.kwargs)
    logging = LoggingConfig(config={}, log_frequency=16, use_wandb=False)
    _, logged = agent.train(seed=0, n_timesteps=32, logging_config=logging)
    assert "replay/expert_share" not in logged
