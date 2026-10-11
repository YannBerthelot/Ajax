"""REDQ's target entropy: ``-action_dim`` by default, or with
``target_entropy_per_dim="mbpo"`` MBPO's per-task value, the default of
REDQ's official code (watchernyu/REDQ@7b5d1bf, ``redq/algos/core.py:10-13``,
read at ``redq/algos/redq_sac.py:57-58``)."""

import pytest

from ajax.agents.REDQ.REDQ import MBPO_TARGET_ENTROPY, REDQ


def test_the_default_stays_minus_the_action_dim() -> None:
    assert REDQ("hopper", buffer_size=1_000).agent_config.target_entropy == -3.0


@pytest.mark.parametrize("env_id", sorted(MBPO_TARGET_ENTROPY))
def test_mbpo_takes_the_tables_value_for_each_brax_task(env_id: str) -> None:
    agent = REDQ(env_id, buffer_size=1_000, target_entropy_per_dim="mbpo")
    assert agent.agent_config.target_entropy == MBPO_TARGET_ENTROPY[env_id]


def test_mbpo_refuses_a_task_the_table_lacks() -> None:
    with pytest.raises(ValueError, match="MBPO's table has no 'Pendulum-v1'"):
        REDQ("Pendulum-v1", target_entropy_per_dim="mbpo")


def test_another_name_is_refused() -> None:
    with pytest.raises(ValueError, match="a number or 'mbpo'"):
        REDQ("Pendulum-v1", target_entropy_per_dim="auto")  # type: ignore[arg-type]
