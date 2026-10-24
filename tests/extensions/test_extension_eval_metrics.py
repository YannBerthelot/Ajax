"""The schedules and shares the expert-guided extensions log at each
evaluation: pure functions of the timestep and the agent state, equal to
the values their phases use."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.extensions.base import ExtensionContext
from ajax.extensions.expert import JSRLCurriculum
from ajax.extensions.exploration import EDGEExploration
from ajax.extensions.pretrain import MCPretrain
from ajax.extensions.target_mods import CriticBlend, LCBGatedBootstrap, ValueBox
from ajax.logging.wandb_logging import LoggingConfig

KEY = jax.random.PRNGKey(0)


def expert(obs: jax.Array) -> jax.Array:
    return jnp.clip(-obs[..., -1:] / 8.0, -1.0, 1.0)


def _metrics(ext: Any, t: int = 25, bounds: dict | None = None) -> dict:
    """``ext``'s evaluation metrics at timestep ``t`` of a 100-step run, one
    env in two inside the box and a quarter driven by the expert."""
    collector = SimpleNamespace(
        timestep=jnp.asarray(t),
        last_expert_frac=jnp.asarray(0.25),
        last_in_box=jnp.array([[1.0], [0.0]]),
    )
    agent_state = SimpleNamespace(collector_state=collector, **(bounds or {}))
    context = ExtensionContext(step=jnp.asarray(t), rng=KEY, total_steps=100)
    out = ext.eval_metrics(agent_state, ext.init_state(agent_state, KEY), KEY, context)
    return {k: float(v) for k, v in out.items()}


def test_jsrl_logs_its_guide_horizon_and_expert_share() -> None:
    jsrl = JSRLCurriculum(episode_length=200, decay_frac=0.5)
    assert _metrics(jsrl) == {"jsrl/horizon": 100.0, "jsrl/expert_frac": 0.25}
    assert _metrics(jsrl, t=60)["jsrl/horizon"] == 0.0


def test_critic_blend_and_lcb_log_their_schedules() -> None:
    blend = CriticBlend(expert_policy=expert, critic_warmup_frac=0.5)
    assert _metrics(blend) == {"critic_blend/expert_weight": 0.5}
    assert _metrics(blend, t=60) == {"critic_blend/expert_weight": 0.0}
    lcb = LCBGatedBootstrap(expert_policy=expert, lcb_beta_init=2.0, lcb_beta_decay_k=1)
    assert _metrics(lcb) == {"lcb_bootstrap/beta": 1.5}  # 2 (1 - 25/100)


@pytest.mark.parametrize(
    "gate, schedule",
    [
        ("fixed", {"edge/decay": 0.5}),  # 1 - 25 / (0.5 * 100)
        ("boltzmann", {"edge/decay": 0.5}),
        ("lcb", {"edge/beta": 0.75}),  # 1 (1 - 25/100)
        ("argmax_lcb", {"edge/beta": 0.75}),
        ("thompson", {}),
    ],
)
def test_edge_logs_the_schedule_of_its_gate(gate: str, schedule: dict) -> None:
    edge = EDGEExploration(gate=gate, decay_frac=0.5, lcb_beta_decay_k=1.0)
    assert _metrics(edge) == {"edge/live_expert_frac": 0.25, **schedule}


def test_value_box_and_mc_pretrain_log_the_box() -> None:
    bounds: dict[str, jax.Array] = {
        "expert_v_min": jnp.asarray(-1.0),
        "expert_v_max": jnp.asarray(3.0),
    }
    assert _metrics(ValueBox(), bounds=bounds) == {
        "value_box/threshold": 0.0,  # -1 + (3 - -1) 25 / 100
        "value_box/in_box_frac": 0.5,
        "value_box/expert_frac": 0.25,
    }
    logged = _metrics(MCPretrain(expert_policy=expert), bounds=bounds)
    assert (logged["mc_pretrain/v_min"], logged["mc_pretrain/v_max"]) == (-1.0, 3.0)


def test_a_logged_sac_run_carries_the_extensions_metrics() -> None:
    """The metrics reach the evaluations of a run, finite, the box's
    bounds those MC pretraining set on the state."""
    stack = (
        MCPretrain(
            expert_policy=expert,
            n_mc_steps=200,
            n_mc_episodes=4,
            n_steps=20,
            online_light_steps=5,
        ),
        CriticBlend(expert_policy=expert),
        LCBGatedBootstrap(expert_policy=expert),
        EDGEExploration(gate="fixed"),
        ValueBox(),
    )
    agent = SAC(
        "Pendulum-v1",
        n_envs=2,
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
        learning_starts=16,
        batch_size=8,
        expert_policy=expert,
        expert_buffer_n_steps=0,
        extensions=stack,
    )
    logging = LoggingConfig(config={}, log_frequency=32, use_wandb=False)
    state, out = agent.train(  # SAC's state, with MCPretrain's range
        seed=0, n_timesteps=64, num_episode_test=1, logging_config=logging
    )
    evaluated = np.asarray(out["timestep"]).reshape(-1) >= 0
    names = ["critic_blend/expert_weight", "lcb_bootstrap/beta", "edge/decay"]
    names += ["value_box/threshold", "value_box/in_box_frac", "value_box/expert_frac"]
    for name in (*names, "mc_pretrain/v_min", "mc_pretrain/v_max"):
        values = np.asarray(out[name]).reshape(-1)[evaluated]
        assert evaluated.any() and np.isfinite(values).all(), name
    np.testing.assert_array_equal(
        np.asarray(out["mc_pretrain/v_max"]).reshape(-1)[evaluated],
        np.asarray(cast(Any, state).expert_v_max).reshape(-1)[0],
    )
