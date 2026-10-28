"""ConservativeQ (CQL), CalQL (Cal-QL) and TargetBC (ReBRAC): CQL's gap, the
floor on the policy actions' values, the stored return to go and next action,
the target penalty, and the refusals. Whether they make a
critic conservative is the probes' (tests/probing/test_offline)."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ajax import SAC
from ajax.buffers.dataset import TransitionDataset, fill_replay, next_actions
from ajax.extensions.base import ExtensionContext
from ajax.extensions.capabilities import MissingCapability
from ajax.extensions.offline import CalQL, ConservativeQ, TargetBC
from ajax.extensions.pretrain import ExpertPrefill
from ajax.networks.networks import predict_value

KEY = jax.random.PRNGKey(0)
CTX = ExtensionContext(step=jnp.asarray(0), rng=KEY)
ROWS = 8


def sac_batch(return_to_go: float) -> dict[str, Any]:
    """A SAC critic-loss batch at its initial networks, on Pendulum."""
    agent = SAC(
        "Pendulum-v1",
        actor_architecture=("8", "relu"),
        critic_architecture=("8", "relu"),
    )
    state = jax.tree.map(lambda x: x[0], agent.train(seed=0, n_timesteps=0)[0])
    obs = jax.random.normal(KEY, (ROWS, 3))
    return {
        "observations": obs,
        "next_observations": obs[::-1],
        "actions": jnp.zeros((ROWS, 1)),
        "critic_params": state.critic_state.params,
        "critic_state": state.critic_state,
        "actor_state": state.actor_state,
        "return_to_go": jnp.full((ROWS, 1), return_to_go),
    }


def test_without_a_floor_calql_is_cql_and_a_floor_above_every_value_binds_all() -> None:
    """A -inf return to go floors nothing: CalQL's term is CQL's on the same
    draws. One above every value raises every policy action's."""
    batch = sac_batch(-jnp.inf)
    cql, _ = ConservativeQ(alpha=2.0, n_actions=4).critic_loss(None, (), batch, CTX)
    calql, metrics = CalQL(alpha=2.0, n_actions=4).critic_loss(None, (), batch, CTX)
    np.testing.assert_allclose(float(calql), float(cql), rtol=1e-6)
    assert float(metrics["calql/bound_frac"]) == 0.0
    high, metrics = CalQL(alpha=2.0, n_actions=4).critic_loss(
        None, (), sac_batch(1e3), CTX
    )
    assert float(metrics["calql/bound_frac"]) == 1.0 and float(high) > float(cql)


def test_the_gap_falls_by_the_datas_own_values() -> None:
    """CQL's gap is the logsumexp over sampled actions minus Q at the data's
    actions (Kumar et al., 2020, eq. 4). Only that last term reads the batch's
    actions, so changing them moves the gap by minus the change in the data's
    mean value (a critic away from its initial, nearly flat, one)."""
    batch = sac_batch(-jnp.inf)
    noise = lambda p: p + 0.3 * jax.random.normal(KEY, p.shape)  # noqa: E731
    batch["critic_params"] = jax.tree.map(noise, batch["critic_params"])
    cql = ConservativeQ(alpha=1.0, n_actions=4)

    def gap_and_value(actions: jax.Array) -> tuple[float, float]:
        at = {**batch, "actions": actions}
        _, metrics = cql.critic_loss(None, (), at, CTX)
        x = jnp.concatenate([at["observations"], actions], -1)
        value = predict_value(at["critic_state"], at["critic_params"], x).mean()
        return float(metrics["cql/gap"]), float(value)

    (gap_0, q_0), (gap_1, q_1) = (
        gap_and_value(batch["actions"]),
        gap_and_value(jnp.full((ROWS, 1), 0.8)),
    )
    assert abs(q_1 - q_0) > 1e-2
    np.testing.assert_allclose(gap_1 - gap_0, -(q_1 - q_0), rtol=1e-4)


def test_the_replay_keeps_a_datasets_return_and_minus_inf_for_collected_rows() -> None:
    agent = SAC("Pendulum-v1", buffer_size=64, batch_size=8, extensions=(CalQL(),))
    state, _ = agent.train(seed=0, n_timesteps=4)
    replay = jax.tree.map(lambda x: x[0], state.collector_state.buffer_state)
    np.testing.assert_array_equal(replay.experience["return_to_go"][0, :4], -np.inf)
    obs, ends = np.zeros((ROWS, 3)), np.ones(ROWS, bool)
    rows = TransitionDataset(
        obs, np.zeros((ROWS, 1)), np.arange(ROWS), ends, ~ends, obs
    )
    empty = agent.train(seed=0, n_timesteps=0)[0].collector_state.buffer_state
    filled = fill_replay(agent.buffer, jax.tree.map(lambda x: x[0], empty), rows)
    np.testing.assert_array_equal(
        filled.experience["return_to_go"][0, :ROWS, 0], rows.return_to_go
    )


def test_expert_prefill_rows_keep_minus_inf() -> None:
    agent = SAC(
        "Pendulum-v1",
        buffer_size=64,
        batch_size=8,
        extensions=(CalQL(), ExpertPrefill(n_steps=4)),
        expert_policy=lambda obs: jnp.zeros((*obs.shape[:-1], 1)),
    )
    state, _ = agent.train(seed=0, n_timesteps=0)
    replay = jax.tree.map(lambda x: x[0], state.collector_state.buffer_state)
    np.testing.assert_array_equal(replay.experience["return_to_go"][0, :4], -np.inf)


def test_the_terms_refuse_bad_settings_and_agents_without_their_inputs() -> None:
    for bad in ({"alpha": -1.0}, {"n_actions": 0}):
        with pytest.raises(ValueError, match="alpha >= 0 and n_actions >= 1"):
            ConservativeQ(**bad)
    from ajax import ASAC, AVG, TD3

    for term in (ConservativeQ(), CalQL()):
        with pytest.raises(MissingCapability, match="policy_density"):
            TD3("Pendulum-v1", extensions=(term,))  # a point mass, no density
    with pytest.raises(MissingCapability, match="discounted"):
        ASAC("Pendulum-v1", extensions=(CalQL(),))  # an average-reward critic
    with pytest.raises(MissingCapability, match="replay"):
        AVG("Pendulum-v1", extensions=(CalQL(),))  # no replay to hold a dataset


def test_target_bc_lowers_the_bootstrap_by_the_next_actions_distance() -> None:
    """y - gamma (1 - d) beta sum (next_actions - a')^2; a row whose a' is
    unknown (NaN) or that ended keeps its target."""
    batch = {
        "next_actions": jnp.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]]),
        "next_action": jnp.array([[0.0, 0.0], [jnp.nan, 0.0], [0.0, 0.0]]),
        "dones": jnp.array([[0.0], [0.0], [1.0]]),
        "gamma": 0.9,
    }
    target, metrics = TargetBC(beta=2.0).on_target(
        None, (), batch, jnp.ones((3, 1)), CTX
    )
    np.testing.assert_allclose(target[:, 0], [1.0 - 0.9 * 2.0 * 0.25, 1.0, 1.0])
    np.testing.assert_allclose(float(metrics["target_bc/known_frac"]), 2 / 3)


def test_a_rows_next_action_is_its_episodes_next_rows() -> None:
    """NaN on an episode's last row, terminal or cut."""
    ends = np.array([False, True, False, False, True])
    action = np.arange(5.0)[:, None]
    rows = TransitionDataset(
        np.zeros((5, 1)), action, np.zeros(5), ends, np.zeros(5, bool), np.zeros((5, 1))
    )
    np.testing.assert_array_equal(
        next_actions(rows)[:, 0], [1.0, np.nan, 3.0, 4.0, np.nan]
    )


def _episodes(rows: int = 32, length: int = 8) -> TransitionDataset:
    """Pendulum-shaped rows in episodes of length steps: all but each
    episode's last have a next action."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(rows, 3))
    ends = np.arange(rows) % length == length - 1
    actions = rng.uniform(-1, 1, (rows, 1))
    cut = np.zeros(rows, bool)
    return TransitionDataset(
        obs, actions, rng.normal(size=rows), ends, cut, np.roll(obs, -1, 0)
    )


@pytest.mark.parametrize("name", ["SAC", "TD3", "REDQ", "TQC", "CrossQ"])
def test_target_bc_reads_the_next_actions_every_target_scores(name: str) -> None:
    """Every agent whose target scores a next action hands it on: offline on
    multi-step episodes the penalty is positive where a' is known."""
    from ajax.logging.wandb_logging import LoggingConfig
    from tests.agents.test_agent_contract import CASES

    case = CASES[name]
    kwargs = dict(case.kwargs)
    kwargs.update({"learning_starts": 0, "extensions": (TargetBC(),)})
    agent = case.cls(env_id="Pendulum-v1", n_envs=2, **kwargs)
    logging = LoggingConfig(log_frequency=4, backends=())
    _, logged = agent.train_offline(_episodes(), 8, seed=0, logging_config=logging)
    assert np.all(np.asarray(logged["target_bc/known_frac"]) > 0)
    assert np.all(np.asarray(logged["target_bc/penalty"]) > 0)


def test_target_bc_refuses_a_target_without_a_next_action_or_a_replay() -> None:
    """IQL bootstraps on V(s'), at no next action; AVG keeps no replay, which
    a dataset's next action reaches the target through."""
    from ajax import AVG, IQL

    with pytest.raises(MissingCapability, match="next_action_target"):
        IQL("Pendulum-v1", extensions=(TargetBC(),))
    with pytest.raises(MissingCapability, match="replay"):
        AVG("Pendulum-v1", extensions=(TargetBC(),))
