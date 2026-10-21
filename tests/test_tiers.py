"""The tier guard (tests/conftest.py) marks the slow tier by path and reads
the gradient updates a fast test's train programs applied."""

from types import SimpleNamespace

import numpy as np

from ajax import SAC
from tests import conftest


def test_the_slow_tier_is_tests_probing_but_its_harness_checks() -> None:
    assert conftest.is_slow_tier("/repo/tests/probing/test_memory.py")
    assert conftest.is_slow_tier("/repo/tests/probing/world_models/test_x.py")
    assert not conftest.is_slow_tier("/repo/tests/probing/test_framework.py")
    assert not conftest.is_slow_tier("/repo/tests/agents/test_agent_contract.py")


def test_the_updates_are_read_off_any_train_result() -> None:
    state = SimpleNamespace(n_updates=np.array([3, 5]))
    assert conftest.updates_of((state, None)) == 5
    assert conftest.updates_of(state) == 5
    assert conftest.updates_of(SimpleNamespace()) == 0  # no counter


def test_a_run_reports_the_steps_its_networks_took() -> None:
    """SAC counts its updates on its train states alone (no ``n_updates``):
    timesteps 13 to 16 update, 4 steps on the one env."""
    net = ("8", "relu")
    agent = SAC(
        env_id="Pendulum-v1",
        n_envs=1,
        learning_starts=13,
        buffer_size=32,
        batch_size=4,
        actor_architecture=net,
        critic_architecture=net,
    )
    state, _ = agent.train(seed=0, n_timesteps=16)
    assert conftest._updates == int(np.max(np.asarray(state.critic_state.step))) == 4


def test_a_fast_test_fails_past_the_line_of_gradient_steps() -> None:
    assert conftest.training_past_the_line(conftest.MAX_UPDATES) is None
    failure = conftest.training_past_the_line(conftest.MAX_UPDATES + 1)
    assert failure is not None and "tests/probing" in failure


def test_a_fast_test_past_a_budget_is_reported(monkeypatch) -> None:
    over: list = []
    monkeypatch.setattr(conftest, "_over", over)
    trained = {"updates": conftest.MAX_UPDATES + 1, "seconds": 1.0}
    slow = {"updates": 0, "seconds": conftest.MAX_SECONDS + 1}
    within = {"updates": conftest.MAX_UPDATES, "seconds": 1.0}
    for nodeid, measured in (("a", trained), ("b", slow), ("c", within)):
        report = SimpleNamespace(
            nodeid=nodeid, user_properties=[("fast_tier", measured)]
        )
        conftest.pytest_runtest_logreport(report)  # type: ignore[arg-type]
    assert [nodeid for nodeid, _ in over] == ["a", "b"]
